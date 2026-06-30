"""
slope_computer.py — Centroid Displacements → Wavefront Slopes
==============================================================
Converts pixel centroid displacements (dx, dy) into physical angular
slopes in radians, and assembles the slope vector used by the
wavefront reconstructor.

Physics
-------
Spot displacement on detector:
    Δx_pix = f_lens * (∂W/∂x) / (λ * pixel_size)
            = f_lens * s_x / pixel_size   [pixels]

Rearranging for the wavefront gradient (slope in rad/m or rad/pix):
    s_x_rad = Δx_pix * pixel_size / f_lens   [radians of tilt]

The slope vector s ∈ ℝ^(2 N_sa_valid) is ordered as:
    [s_x_0, s_x_1, ..., s_x_N, s_y_0, ..., s_y_N]
where indices run over VALID sub-apertures only.

Output
------
``SlopeResult.s_x``   : float32 [N_frames, N_sa_valid]  x-slopes [rad]
``SlopeResult.s_y``   : float32 [N_frames, N_sa_valid]  y-slopes [rad]
``SlopeResult.slopes``: float32 [N_frames, 2*N_sa_valid] stacked slope vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SlopeResult:
    """Holds the computed slopes for all frames."""
    s_x:    np.ndarray   # [N_frames, N_sa_valid]  x-slopes [rad]
    s_y:    np.ndarray   # [N_frames, N_sa_valid]  y-slopes [rad]
    slopes: np.ndarray   # [N_frames, 2*N_sa_valid] concatenated [sx | sy]
    valid:  np.ndarray   # [N_frames, N_sa_valid]   frame-wise validity
    # Metadata
    n_frames:    int
    n_sa_valid:  int
    pixel_size:  float   # [m]
    focal_length: float  # [m]
    timestamps_ms: np.ndarray  # [N_frames]

    # Conversion factor from pixels to radians stored for traceability
    pix_to_rad: float   # = pixel_size / focal_length

    @property
    def rms_slope(self) -> np.ndarray:
        """RMS slope magnitude per frame [rad], shape [N_frames]."""
        mag = np.sqrt(self.s_x**2 + self.s_y**2)
        # Only average over valid sub-apertures
        mask = self.valid  # [N_frames, N_sa]
        rms = np.array([
            np.sqrt(np.mean(mag[i, mask[i]]**2)) if mask[i].any() else 0.0
            for i in range(self.n_frames)
        ])
        return rms.astype(np.float32)

    @property
    def mean_x_slope(self) -> np.ndarray:
        """Mean x-slope per frame (= x-tilt component) [rad]."""
        return np.nanmean(
            np.where(self.valid, self.s_x, np.nan), axis=1
        ).astype(np.float32)

    @property
    def mean_y_slope(self) -> np.ndarray:
        """Mean y-slope per frame (= y-tilt component) [rad]."""
        return np.nanmean(
            np.where(self.valid, self.s_y, np.nan), axis=1
        ).astype(np.float32)


class SlopeComputer:
    """
    Converts centroid displacements to angular wavefront slopes.

    Parameters
    ----------
    pixel_size : float
        Physical pixel size [m].
    focal_length : float
        MLA lenslet focal length [m].
    lenslet_size : float
        Physical lenslet size [m] — used for slope-to-wavefront-gradient
        conversion if needed.
    valid_mask : np.ndarray, shape [N_sa_total], dtype bool
        Boolean mask of which sub-apertures are inside the pupil.
    clip_sigma : float | None
        If set, clip slope outliers beyond clip_sigma * std per frame.
        Useful for bad-centroid rejection.
    """

    def __init__(self,
                 pixel_size:    float,
                 focal_length:  float,
                 lenslet_size:  float,
                 valid_mask:    np.ndarray,
                 clip_sigma:    Optional[float] = 4.0):

        self.pixel_size    = pixel_size
        self.focal_length  = focal_length
        self.lenslet_size  = lenslet_size
        self.valid_mask    = valid_mask.astype(bool)  # [N_sa_total]
        self.clip_sigma    = clip_sigma

        # Pixel displacement → slope conversion factor
        # s [rad] = Δpix * pixel_size / focal_length
        self.pix_to_rad = pixel_size / focal_length

        self._n_sa_total = len(valid_mask)
        self._n_sa_valid = int(valid_mask.sum())
        self._valid_idx  = np.where(valid_mask)[0]

        log.info(
            "SlopeComputer: %d/%d sub-apertures valid  "
            "pix→rad factor = %.3e",
            self._n_sa_valid, self._n_sa_total, self.pix_to_rad
        )

    # ── Core conversion ──────────────────────────────────────────────────────

    def compute(self, centroid_result: dict) -> SlopeResult:
        """
        Compute slopes from the output dict of CentroidPipeline.run().

        Parameters
        ----------
        centroid_result : dict
            Output of ``CentroidPipeline.run()``.

        Returns
        -------
        SlopeResult
        """
        dx    = centroid_result["dx"]    # [N_frames, N_sa_total] pixels
        dy    = centroid_result["dy"]    # [N_frames, N_sa_total] pixels
        valid = centroid_result["valid"] # [N_frames, N_sa_total] bool
        ts    = centroid_result["timestamps_ms"]

        n_frames = dx.shape[0]

        # Extract valid sub-apertures only
        dx_valid = dx[:, self._valid_idx]    # [N_frames, N_sa_valid]
        dy_valid = dy[:, self._valid_idx]
        v_valid  = valid[:, self._valid_idx]  # [N_frames, N_sa_valid]

        # Convert pixels → radians
        s_x = (dx_valid * self.pix_to_rad).astype(np.float32)
        s_y = (dy_valid * self.pix_to_rad).astype(np.float32)

        # Outlier rejection per frame
        if self.clip_sigma is not None:
            s_x, s_y, v_valid = self._clip_outliers(s_x, s_y, v_valid)

        # Assemble concatenated slope vector [sx | sy] per frame
        slopes = np.concatenate([s_x, s_y], axis=1)  # [N_frames, 2*N_sa_valid]

        return SlopeResult(
            s_x          = s_x,
            s_y          = s_y,
            slopes       = slopes,
            valid        = v_valid,
            n_frames     = n_frames,
            n_sa_valid   = self._n_sa_valid,
            pixel_size   = self.pixel_size,
            focal_length = self.focal_length,
            timestamps_ms= ts,
            pix_to_rad   = self.pix_to_rad,
        )

    def _clip_outliers(self,
                       s_x:   np.ndarray,
                       s_y:   np.ndarray,
                       valid: np.ndarray,
                       ) -> tuple:
        """
        Per-frame outlier rejection: zero out slopes more than
        clip_sigma * std away from frame mean.  Mark those as invalid.
        """
        sigma = self.clip_sigma
        for i in range(s_x.shape[0]):
            for s in (s_x[i], s_y[i]):
                m = np.mean(s[valid[i]])
                std = np.std(s[valid[i]]) + 1e-12
                outlier = np.abs(s - m) > sigma * std
                valid[i] &= ~outlier
                s[outlier] = 0.0
        return s_x, s_y, valid

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def slope_map_2d(self,
                     s_x_frame: np.ndarray,
                     s_y_frame: np.ndarray,
                     valid_frame: np.ndarray,
                     ) -> tuple:
        """
        Reshape 1D valid-SA slopes back to 2D grid for visualisation.

        Parameters
        ----------
        s_x_frame, s_y_frame : float [N_sa_valid]
        valid_frame          : bool  [N_sa_valid]

        Returns
        -------
        sx_2d, sy_2d : float [n_sa_y, n_sa_x]  (NaN outside pupil)
        """
        grid = None
        # We need grid to reshape — get it from calling code context.
        # This is a utility method; the caller supplies the correct grid.
        raise NotImplementedError(
            "Call slope_map_2d_from_grid() supplying the SubApertureGrid"
        )

    def slope_map_2d_from_grid(self,
                                s_x_frame:   np.ndarray,
                                s_y_frame:   np.ndarray,
                                valid_frame: np.ndarray,
                                grid) -> tuple:
        """
        Reshape slopes to 2D grid [n_sa_y, n_sa_x].

        Parameters
        ----------
        s_x_frame, s_y_frame : [N_sa_valid]
        valid_frame          : [N_sa_valid]
        grid                 : SubApertureGrid

        Returns
        -------
        sx_2d, sy_2d : float32 [n_sa_y, n_sa_x]  NaN outside pupil
        """
        ny, nx = grid.n_sa_y, grid.n_sa_x
        sx_2d = np.full((ny, nx), np.nan, dtype=np.float32)
        sy_2d = np.full((ny, nx), np.nan, dtype=np.float32)

        valid_idx = self._valid_idx   # indices into [N_sa_total]
        for k, i in enumerate(valid_idx):
            row = i // nx
            col = i  % nx
            if valid_frame[k]:
                sx_2d[row, col] = s_x_frame[k]
                sy_2d[row, col] = s_y_frame[k]

        return sx_2d, sy_2d

    def print_summary(self, slope_result: SlopeResult):
        sr = slope_result
        rms = sr.rms_slope
        print(
            f"\nSlope Summary\n"
            f"  Frames        : {sr.n_frames}\n"
            f"  Valid SA / Total : {sr.n_sa_valid}/{self._n_sa_total}\n"
            f"  pix → rad     : {sr.pix_to_rad:.3e}\n"
            f"  RMS slope     : {rms.mean():.3e} ± {rms.std():.3e} rad\n"
            f"  Max |sx|      : {np.abs(sr.s_x).max():.3e} rad\n"
            f"  Max |sy|      : {np.abs(sr.s_y).max():.3e} rad\n"
        )
