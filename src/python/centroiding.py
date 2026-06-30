"""
centroiding.py — SH-WFS Spot Centroiding (Python + C bridge)
=============================================================
Provides three centroiding backends:

  1. ``PythonCentroider``  — pure NumPy, easy to inspect and debug
  2. ``CCentroider``       — ctypes bridge to libcentroid.so (fast)
  3. ``CentroidPipeline``  — orchestrates either backend over a frame sequence

Coordinate convention
---------------------
All centroid coordinates are in **full-frame pixels**, 0-indexed from the
top-left corner.  Sub-aperture indices follow row-major order matching
the sub-aperture grid defined in ``build_grid()``.

Array shapes
------------
``centroids`` : float32 array  [N_frames, N_sa, 2]  (col, row) = (x, y)
``valid``     : bool   array   [N_frames, N_sa]
``flux``      : float32 array  [N_frames, N_sa]
"""

from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Sub-aperture grid ────────────────────────────────────────────────────────

@dataclass
class SubApertureGrid:
    """
    Defines the lenslet grid on the detector.

    Attributes
    ----------
    n_sa_x, n_sa_y : int
        Number of sub-apertures in x and y.
    pix_per_sa : int
        Pixels allocated to each sub-aperture (square).
    x_start, y_start : int array [N_sa]
        Top-left pixel of each sub-aperture (row-major order).
    cx_ref, cy_ref : float array [N_sa]
        Geometric centre of each sub-aperture in full-frame pixels.
    valid : bool array [N_sa]
        True for sub-apertures inside the pupil mask.
    n_sa_total : int
        Total sub-apertures (n_sa_x * n_sa_y).
    n_sa_valid : int
        Sub-apertures inside the pupil.
    """
    n_sa_x:     int
    n_sa_y:     int
    pix_per_sa: int
    x_start:    np.ndarray   # [N_sa] int32
    y_start:    np.ndarray   # [N_sa] int32
    cx_ref:     np.ndarray   # [N_sa] float32  geometric centres
    cy_ref:     np.ndarray   # [N_sa] float32
    valid:      np.ndarray   # [N_sa] bool

    @property
    def n_sa_total(self) -> int:
        return self.n_sa_x * self.n_sa_y

    @property
    def n_sa_valid(self) -> int:
        return int(self.valid.sum())

    @property
    def valid_indices(self) -> np.ndarray:
        return np.where(self.valid)[0]


def build_grid(n_sa_x: int,
               n_sa_y: int,
               pix_per_sa: int,
               pupil_cx: float,
               pupil_cy: float,
               pupil_radius: float) -> SubApertureGrid:
    """
    Build a SubApertureGrid from MLA parameters and pupil geometry.

    Parameters
    ----------
    n_sa_x, n_sa_y  : MLA sub-aperture count
    pix_per_sa       : pixels per sub-aperture (square)
    pupil_cx, cy     : pupil centre in full-frame pixels
    pupil_radius     : pupil radius in pixels (sub-apertures whose centre
                       lies outside this circle are masked)
    """
    n = n_sa_x * n_sa_y
    x_start = np.empty(n, dtype=np.int32)
    y_start = np.empty(n, dtype=np.int32)
    cx_ref  = np.empty(n, dtype=np.float32)
    cy_ref  = np.empty(n, dtype=np.float32)
    valid   = np.zeros(n, dtype=bool)

    idx = 0
    for j in range(n_sa_y):
        for i in range(n_sa_x):
            xs = i * pix_per_sa
            ys = j * pix_per_sa
            cx = xs + 0.5 * (pix_per_sa - 1)
            cy = ys + 0.5 * (pix_per_sa - 1)

            x_start[idx] = xs
            y_start[idx] = ys
            cx_ref[idx]  = cx
            cy_ref[idx]  = cy

            r = np.hypot(cx - pupil_cx, cy - pupil_cy)
            valid[idx] = (r <= pupil_radius)
            idx += 1

    return SubApertureGrid(
        n_sa_x=n_sa_x, n_sa_y=n_sa_y, pix_per_sa=pix_per_sa,
        x_start=x_start, y_start=y_start,
        cx_ref=cx_ref, cy_ref=cy_ref,
        valid=valid,
    )


# ── Pure-Python centroider ───────────────────────────────────────────────────

class PythonCentroider:
    """
    NumPy-based centroiding — all three algorithms.
    Slower than the C backend but useful for validation and debugging.

    Parameters
    ----------
    grid : SubApertureGrid
    method : str
        One of ``"cog"``, ``"threshold_cog"``, ``"windowed_cog"``
    threshold_sigma : float
        For T-CoG: threshold = bg_mean + threshold_sigma * bg_std
    window_sigma_px : float
        For W-CoG: Gaussian window sigma in pixels
    border_px : int
        Border width for background estimation
    min_flux : float
        Minimum integrated (thresholded) flux for a valid centroid
    """

    METHODS = ("cog", "threshold_cog", "windowed_cog")

    def __init__(self,
                 grid:             SubApertureGrid,
                 method:           str   = "threshold_cog",
                 threshold_sigma:  float = 3.0,
                 window_sigma_px:  float = 2.5,
                 border_px:        int   = 2,
                 min_flux:         float = 10.0):
        assert method in self.METHODS, f"method must be one of {self.METHODS}"
        self.grid            = grid
        self.method          = method
        self.threshold_sigma = threshold_sigma
        self.window_sigma_px = window_sigma_px
        self.border_px       = border_px
        self.min_flux        = min_flux

        # Pre-build pixel coordinate grids for each sub-aperture
        # Shape: [psa, psa] for local coords; full-frame offsets added per SA
        p = grid.pix_per_sa
        local_col = np.arange(p, dtype=np.float32)          # [p]
        local_row = np.arange(p, dtype=np.float32)          # [p]
        self._lc, self._lr = np.meshgrid(local_col, local_row)  # [p,p]

    # ── Per-frame centroiding ────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Centroid all sub-apertures in one frame.

        Parameters
        ----------
        frame : float32 array [H, W]

        Returns
        -------
        cx, cy : float32 [N_sa]   centroid positions (full-frame pixels)
        flux   : float32 [N_sa]   integrated flux per sub-aperture
        valid  : bool    [N_sa]   True where centroid succeeded
        """
        n = self.grid.n_sa_total
        cx    = self.grid.cx_ref.copy()
        cy    = self.grid.cy_ref.copy()
        flux  = np.zeros(n, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)

        for idx in range(n):
            if not self.grid.valid[idx]:
                continue

            xs = self.grid.x_start[idx]
            ys = self.grid.y_start[idx]
            p  = self.grid.pix_per_sa

            tile = frame[ys:ys+p, xs:xs+p]   # [p, p]

            # Full-frame coordinate grids
            col_ff = self._lc + xs    # [p, p]
            row_ff = self._lr + ys    # [p, p]

            cx_i, cy_i, f_i, v_i = self._centroid_tile(
                tile, col_ff, row_ff,
                xs + 0.5*(p-1), ys + 0.5*(p-1)
            )
            cx[idx]    = cx_i
            cy[idx]    = cy_i
            flux[idx]  = f_i
            valid[idx] = v_i and self.grid.valid[idx]

        return cx.astype(np.float32), cy.astype(np.float32), \
               flux.astype(np.float32), valid

    def _centroid_tile(self, tile, col_ff, row_ff, cx_geom, cy_geom):
        """Dispatch to the correct algorithm for a single sub-aperture tile."""
        if self.method == "cog":
            return self._cog(tile, col_ff, row_ff, cx_geom, cy_geom)
        elif self.method == "threshold_cog":
            return self._tcog(tile, col_ff, row_ff, cx_geom, cy_geom)
        else:
            return self._wcog(tile, col_ff, row_ff, cx_geom, cy_geom)

    def _background(self, tile: np.ndarray) -> Tuple[float, float]:
        """Estimate background from border pixels."""
        bp = self.border_px
        if bp <= 0:
            return 0.0, 1.0
        mask = np.zeros(tile.shape, dtype=bool)
        mask[:bp,  :] = True
        mask[-bp:, :] = True
        mask[:,  :bp] = True
        mask[:, -bp:] = True
        border_vals = tile[mask]
        if len(border_vals) < 2:
            return 0.0, 1.0
        return float(border_vals.mean()), float(border_vals.std()) + 1e-9

    def _cog(self, tile, col_ff, row_ff, cx_geom, cy_geom):
        total = float(tile.sum())
        if total <= self.min_flux:
            return cx_geom, cy_geom, total, False
        cx = float((tile * col_ff).sum()) / total
        cy = float((tile * row_ff).sum()) / total
        return cx, cy, total, True

    def _tcog(self, tile, col_ff, row_ff, cx_geom, cy_geom):
        bg_mean, bg_std = self._background(tile)
        thr = bg_mean + self.threshold_sigma * bg_std
        t = np.maximum(tile - thr, 0.0)
        total = float(t.sum())
        if total <= self.min_flux:
            return cx_geom, cy_geom, total, False
        cx = float((t * col_ff).sum()) / total
        cy = float((t * row_ff).sum()) / total
        return cx, cy, total, True

    def _wcog(self, tile, col_ff, row_ff, cx_geom, cy_geom):
        # Pass 1: T-CoG for initial estimate
        cx0, cy0, _, v0 = self._tcog(tile, col_ff, row_ff, cx_geom, cy_geom)
        if not v0:
            cx0, cy0 = cx_geom, cy_geom

        # Pass 2: Gaussian-windowed CoG
        sig = self.window_sigma_px
        r2  = (col_ff - cx0)**2 + (row_ff - cy0)**2
        w   = np.exp(-0.5 * r2 / (sig * sig))

        bg_mean, bg_std = self._background(tile)
        thr = bg_mean + self.threshold_sigma * bg_std
        t   = np.maximum(tile - thr, 0.0)
        wt  = w * t
        total = float(wt.sum())
        if total <= self.min_flux:
            return cx0, cy0, total, False
        cx = float((wt * col_ff).sum()) / total
        cy = float((wt * row_ff).sum()) / total
        return cx, cy, total, True


# ── C-backed centroider ──────────────────────────────────────────────────────

class CCentroider:
    """
    High-speed centroider backed by libcentroid.so via ctypes.
    Falls back transparently to PythonCentroider if the shared library
    is not found.

    Parameters match PythonCentroider.
    """

    _LIB_NAME = "libcentroid.so"
    _LIB_DIRS = [
        Path(__file__).parent.parent / "c",   # src/c/
        Path.cwd() / "src" / "c",
    ]

    def __init__(self,
                 grid:            SubApertureGrid,
                 method:          str   = "threshold_cog",
                 threshold_sigma: float = 3.0,
                 window_sigma_px: float = 2.5,
                 border_px:       int   = 2,
                 min_flux:        float = 10.0):

        self.grid            = grid
        self.method          = method
        self.threshold_sigma = threshold_sigma
        self.window_sigma_px = window_sigma_px
        self.border_px       = border_px
        self.min_flux        = min_flux

        self._lib = self._load_lib()

        if self._lib is not None:
            self._setup_lib_types()
            self._sas_c = self._build_sa_structs()
            log.info("CCentroider: using C backend (libcentroid.so)")
        else:
            log.warning("CCentroider: C library not found — falling back to Python")
            self._fallback = PythonCentroider(
                grid=grid, method=method,
                threshold_sigma=threshold_sigma,
                window_sigma_px=window_sigma_px,
                border_px=border_px, min_flux=min_flux
            )

    # ── ctypes plumbing ──────────────────────────────────────────────────────

    def _load_lib(self) -> Optional[ctypes.CDLL]:
        for d in self._LIB_DIRS:
            p = d / self._LIB_NAME
            if p.exists():
                try:
                    return ctypes.CDLL(str(p))
                except OSError as e:
                    log.warning("Failed to load %s: %s", p, e)
        return None

    def _setup_lib_types(self):
        """Declare ctypes argument/return types for the C functions."""
        lib = self._lib

        # Match struct layout from centroid.h
        class SubAperture_C(ctypes.Structure):
            _fields_ = [
                ("x_start", ctypes.c_int),
                ("y_start", ctypes.c_int),
                ("width",   ctypes.c_int),
                ("height",  ctypes.c_int),
                ("valid",   ctypes.c_int),
            ]

        class Centroid_C(ctypes.Structure):
            _fields_ = [
                ("cx",         ctypes.c_double),
                ("cy",         ctypes.c_double),
                ("total_flux", ctypes.c_double),
                ("valid",      ctypes.c_int),
            ]

        class CentroidConfig_C(ctypes.Structure):
            _fields_ = [
                ("method",       ctypes.c_int),
                ("threshold",    ctypes.c_float),
                ("min_flux",     ctypes.c_float),
                ("window_sigma", ctypes.c_double),
                ("border_px",    ctypes.c_int),
                ("use_openmp",   ctypes.c_int),
            ]

        self._SubAperture_C    = SubAperture_C
        self._Centroid_C       = Centroid_C
        self._CentroidConfig_C = CentroidConfig_C

        lib.batch_centroid.restype  = None
        lib.batch_centroid.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # image
            ctypes.c_int,                    # img_width
            ctypes.c_int,                    # img_height
            ctypes.POINTER(SubAperture_C),   # sas
            ctypes.c_int,                    # n_sa
            ctypes.POINTER(CentroidConfig_C),# cfg
            ctypes.POINTER(Centroid_C),      # results
        ]

    def _build_sa_structs(self):
        """Build the C SubAperture array from the Python grid."""
        n  = self.grid.n_sa_total
        SA = self._SubAperture_C * n
        sas = SA()
        p   = self.grid.pix_per_sa
        for i in range(n):
            sas[i].x_start = int(self.grid.x_start[i])
            sas[i].y_start = int(self.grid.y_start[i])
            sas[i].width   = p
            sas[i].height  = p
            sas[i].valid   = int(self.grid.valid[i])
        return sas

    def _build_config(self) -> "CentroidConfig_C":
        method_map = {"cog": 0, "threshold_cog": 1, "windowed_cog": 2}
        cfg = self._CentroidConfig_C()
        cfg.method       = method_map.get(self.method, 1)
        cfg.threshold    = float(self.threshold_sigma)
        cfg.min_flux     = float(self.min_flux)
        cfg.window_sigma = float(self.window_sigma_px)
        cfg.border_px    = int(self.border_px)
        cfg.use_openmp   = 1
        return cfg

    # ── Public interface ─────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._lib is None:
            return self._fallback.process_frame(frame)

        # Ensure contiguous float32 C-order
        img = np.ascontiguousarray(frame, dtype=np.float32)
        h, w = img.shape

        n = self.grid.n_sa_total
        Res = self._Centroid_C * n
        results = Res()
        cfg = self._build_config()

        self._lib.batch_centroid(
            img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(w), ctypes.c_int(h),
            self._sas_c,
            ctypes.c_int(n),
            ctypes.byref(cfg),
            results,
        )

        cx    = np.array([r.cx         for r in results], dtype=np.float32)
        cy    = np.array([r.cy         for r in results], dtype=np.float32)
        flux  = np.array([r.total_flux for r in results], dtype=np.float32)
        valid = np.array([bool(r.valid) and self.grid.valid[i]
                          for i, r in enumerate(results)], dtype=bool)

        return cx, cy, flux, valid


# ── Pipeline ─────────────────────────────────────────────────────────────────

class CentroidPipeline:
    """
    Runs centroiding over an entire frame sequence and manages reference
    centroid computation.

    Usage
    -----
    >>> pipeline = CentroidPipeline(grid, cfg, use_c=True)
    >>> pipeline.set_reference("geometric")
    >>> results = pipeline.run(frame_loader)
    """

    def __init__(self,
                 grid:    SubApertureGrid,
                 method:  str   = "threshold_cog",
                 threshold_sigma: float = 3.0,
                 window_sigma_px: float = 2.5,
                 border_px: int = 2,
                 min_flux:  float = 10.0,
                 use_c: bool = True):

        self.grid = grid
        kwargs = dict(
            grid=grid, method=method,
            threshold_sigma=threshold_sigma,
            window_sigma_px=window_sigma_px,
            border_px=border_px, min_flux=min_flux,
        )
        if use_c:
            self._centroider = CCentroider(**kwargs)
        else:
            self._centroider = PythonCentroider(**kwargs)

        # Reference centroids (set via set_reference)
        self.ref_cx: Optional[np.ndarray] = None
        self.ref_cy: Optional[np.ndarray] = None

    def set_reference(self,
                      method: str = "geometric",
                      flat_frame: Optional[np.ndarray] = None,
                      avg_frames: Optional[np.ndarray] = None):
        """
        Compute reference centroid positions.

        Parameters
        ----------
        method      : "geometric" | "flat_frame" | "time_average"
        flat_frame  : float32 [H,W] — used if method="flat_frame"
        avg_frames  : float32 [N,H,W] — used if method="time_average"
        """
        if method == "geometric":
            self.ref_cx = self.grid.cx_ref.copy()
            self.ref_cy = self.grid.cy_ref.copy()
            log.info("Reference: geometric centres")

        elif method == "flat_frame":
            assert flat_frame is not None
            cx, cy, _, _ = self._centroider.process_frame(flat_frame)
            self.ref_cx = cx
            self.ref_cy = cy
            log.info("Reference: flat frame centroids")

        elif method == "time_average":
            assert avg_frames is not None
            n = len(avg_frames)
            cx_acc = np.zeros(self.grid.n_sa_total, dtype=np.float64)
            cy_acc = np.zeros(self.grid.n_sa_total, dtype=np.float64)
            for f in avg_frames:
                cx, cy, _, _ = self._centroider.process_frame(f)
                cx_acc += cx
                cy_acc += cy
            self.ref_cx = (cx_acc / n).astype(np.float32)
            self.ref_cy = (cy_acc / n).astype(np.float32)
            log.info("Reference: time-average over %d frames", n)
        else:
            raise ValueError(f"Unknown reference method: {method}")

    def run(self, frame_source, progress: bool = True
            ) -> dict:
        """
        Run centroiding over all frames.

        Parameters
        ----------
        frame_source : FrameLoader or NumpyFrameLoader
        progress     : show tqdm progress bar

        Returns
        -------
        dict with keys:
          'cx'    : float32 [N_frames, N_sa]
          'cy'    : float32 [N_frames, N_sa]
          'flux'  : float32 [N_frames, N_sa]
          'valid' : bool    [N_frames, N_sa]
          'dx'    : float32 [N_frames, N_sa]  — cx - ref_cx (slopes in pixels)
          'dy'    : float32 [N_frames, N_sa]
          'timestamps_ms' : float [N_frames]
        """
        if self.ref_cx is None:
            self.set_reference("geometric")

        n_frames = frame_source.n_frames
        n_sa     = self.grid.n_sa_total

        cx_all    = np.empty((n_frames, n_sa), dtype=np.float32)
        cy_all    = np.empty((n_frames, n_sa), dtype=np.float32)
        flux_all  = np.empty((n_frames, n_sa), dtype=np.float32)
        valid_all = np.empty((n_frames, n_sa), dtype=bool)
        timestamps= np.empty(n_frames, dtype=np.float32)

        iter_fn = frame_source.iter_frames()
        if progress:
            try:
                from tqdm import tqdm
                iter_fn = tqdm(frame_source.iter_frames(),
                               total=n_frames, desc="Centroiding", unit="frame")
            except ImportError:
                pass

        for i, (frame, meta) in enumerate(iter_fn):
            cx, cy, flux, valid = self._centroider.process_frame(frame)
            cx_all[i]    = cx
            cy_all[i]    = cy
            flux_all[i]  = flux
            valid_all[i] = valid
            timestamps[i]= meta.timestamp_ms

        dx_all = cx_all - self.ref_cx[np.newaxis, :]
        dy_all = cy_all - self.ref_cy[np.newaxis, :]

        return {
            "cx": cx_all, "cy": cy_all,
            "flux": flux_all, "valid": valid_all,
            "dx": dx_all, "dy": dy_all,
            "ref_cx": self.ref_cx, "ref_cy": self.ref_cy,
            "timestamps_ms": timestamps,
            "grid": self.grid,
        }
