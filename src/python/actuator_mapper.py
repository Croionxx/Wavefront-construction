"""
actuator_mapper.py — Stage 4: DM Actuator Map & Control
=========================================================
Computes per-frame deformable mirror actuator commands from the
reconstructed wavefront (Stage 2 output) and evaluates the residual
wavefront after correction.

Physics
-------
Each actuator k at position (x_k, y_k) on the DM produces a Gaussian
influence function on the mirror surface:

    F_k(x, y) = exp(−((x−x_k)² + (y−y_k)²) / (2σ_IF²))

where σ_IF is derived from the coupling coefficient c:
    c = F_k(pitch)  →  σ_IF = pitch / √(2 ln(1/c))

The mirror surface (in wavefront phase units, including the factor of 2
for double-pass reflection) is:

    W_mirror(x, y) = Σ_k u_k F_k(x, y)

To conjugate the measured aberration we solve:
    u* = arg min ‖F_pupil u + W_pupil‖²  →  u* = −F_pupil⁺ W_pupil

Actuator commands are clipped to max_stroke before applying correction.

Unit conventions
----------------
*  All phase quantities in radians throughout.
*  u_k in radians of wavefront phase correction (already includes the
   factor-of-2 reflection geometry).
*  Physical stroke [µm] = u_k * λ / (4π) * 1e6.

References
----------
* Hardy (1998) Ch. 6 — Influence function matrices and DM control
* Southwell (1980) — Least-squares wavefront fitting
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import svd

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Output data structure
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DmResult:
    """Stage 4 output — actuator commands and corrected wavefront metrics."""

    # Per-frame actuator commands [rad of wavefront phase correction]
    actuator_commands:    np.ndarray   # [N, n_act_y, n_act_x]  float32

    # Physical DM surface stroke  h = u * λ/(4π)
    stroke_um:            np.ndarray   # [N, n_act_y, n_act_x]  µm  float32

    # Corrected wavefront phase maps
    corrected_phase:      np.ndarray   # [N, H, W]  rad  float32

    # Per-frame quality metrics
    uncorrected_strehl:   np.ndarray   # [N]  Maréchal approximation
    corrected_strehl:     np.ndarray   # [N]
    uncorrected_wfe_rms:  np.ndarray   # [N]  rad
    corrected_wfe_rms:    np.ndarray   # [N]  rad

    # Geometry
    pupil_mask:           np.ndarray   # [H, W]  bool
    timestamps_ms:        np.ndarray   # [N]
    n_act_x:              int
    n_act_y:              int
    saturation_frac:      float        # fraction of frames with clipped actuators

    @property
    def n_frames(self) -> int:
        return self.actuator_commands.shape[0]

    @property
    def wfe_improvement_factor(self) -> float:
        return float(self.uncorrected_wfe_rms.mean() /
                     (self.corrected_wfe_rms.mean() + 1e-12))

    def print_summary(self):
        u = self.uncorrected_wfe_rms
        c = self.corrected_wfe_rms
        us = self.uncorrected_strehl
        cs = self.corrected_strehl
        print(
            f"\nStage 4 — DM Correction Summary\n"
            f"  Actuators           : {self.n_act_x} × {self.n_act_y} "
            f"= {self.n_act_x * self.n_act_y}\n"
            f"  Frames              : {self.n_frames}\n"
            f"  Uncorrected WFE RMS : {u.mean()*1e3:.2f} ± {u.std()*1e3:.2f} mrad\n"
            f"  Corrected WFE RMS   : {c.mean()*1e3:.2f} ± {c.std()*1e3:.2f} mrad\n"
            f"  WFE improvement     : {self.wfe_improvement_factor:.1f}×\n"
            f"  Uncorrected Strehl  : {us.mean():.4f}\n"
            f"  Corrected Strehl    : {cs.mean():.4f}\n"
            f"  Max stroke |u|      : {np.abs(self.actuator_commands).max():.2f} rad  "
            f"({np.abs(self.stroke_um).max():.2f} µm)\n"
            f"  Saturation          : {self.saturation_frac*100:.1f}% of frames\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Actuator mapper
# ═══════════════════════════════════════════════════════════════════════════════

class ActuatorMapper:
    """
    Compute DM actuator commands via SVD least-squares inversion of the
    Gaussian influence function matrix.

    Parameters
    ----------
    n_act_x, n_act_y : int
        Number of actuators (= n_sa + 1 in each direction, Fried geometry).
    d_act_norm : float
        Actuator pitch in NORMALISED pupil coordinates.
        = actuator_pitch_m / (aperture_diameter_m / 2)
        = 2 / n_sa_x  (for Fried geometry exactly spanning the pupil).
    coupling : float
        Inter-actuator coupling: F_k evaluated at the adjacent actuator.
        Determines the Gaussian σ: σ = d_act_norm / √(2 ln(1/coupling)).
    max_stroke_rad : float
        Per-actuator wavefront stroke limit [rad].
        = max_stroke_m * 4π / λ  (both surfaces + double-pass).
    svd_threshold : float
        Singular value truncation fraction for the pseudo-inverse.
    output_grid_px : int
        Grid side-length; must match ModalReconstructor.output_grid_px.
    """

    def __init__(self,
                 n_act_x:        int,
                 n_act_y:        int,
                 d_act_norm:     float,
                 coupling:       float = 0.15,
                 max_stroke_rad: float = 30.0,
                 svd_threshold:  float = 1e-3,
                 output_grid_px: int   = 64):

        self.n_act_x        = n_act_x
        self.n_act_y        = n_act_y
        self.n_act          = n_act_x * n_act_y
        self.d_act          = d_act_norm
        self.coupling       = coupling
        self.max_stroke_rad = max_stroke_rad
        self.svd_threshold  = svd_threshold
        self.G              = output_grid_px

        # Gaussian σ from coupling constraint: F(d) = coupling → σ = d/√(2ln(1/c))
        self.sigma_norm = d_act_norm / math.sqrt(2.0 * math.log(1.0 / coupling))

        # Actuator positions in normalised pupil coords [-1, +1]
        # Fried: actuators at lenslet corners, spanning the full pupil diameter
        self.act_x_norm = np.array([-1.0 + i * d_act_norm for i in range(n_act_x)])
        self.act_y_norm = np.array([-1.0 + j * d_act_norm for j in range(n_act_y)])

        # Calibrated after calibrate()
        self.F_full:  Optional[np.ndarray] = None   # [G², n_act]  full grid IF
        self.F_pup:   Optional[np.ndarray] = None   # [n_pupil, n_act]
        self.F_pinv:  Optional[np.ndarray] = None   # [n_act, n_pupil]
        self._pupil_mask: Optional[np.ndarray] = None
        self._pupil_flat_idx: Optional[np.ndarray] = None
        self._singular_values: Optional[np.ndarray] = None

        log.info(
            "ActuatorMapper: %d×%d actuators | "
            "d_act=%.3f | σ_IF=%.4f | max_stroke=%.1f rad",
            n_act_x, n_act_y, d_act_norm, self.sigma_norm, max_stroke_rad
        )

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate(self, pupil_mask: np.ndarray):
        """
        Build the Gaussian influence function matrix and compute its
        SVD pseudo-inverse.

        Parameters
        ----------
        pupil_mask : bool [G, G]
            In-pupil pixel map from WavefrontResult.pupil_mask.
        """
        self._pupil_mask     = pupil_mask
        self._pupil_flat_idx = np.where(pupil_mask.ravel())[0]
        n_pup  = len(self._pupil_flat_idx)
        G      = self.G
        n_act  = self.n_act

        log.info("Building influence matrix [%d pupil pixels × %d actuators]",
                 n_pup, n_act)

        lin = np.linspace(-1.0, 1.0, G)
        Xg, Yg = np.meshgrid(lin, lin)          # [G, G] in normalised coords

        F_full = np.zeros((G * G, n_act), dtype=np.float64)
        inv2s2 = 1.0 / (2.0 * self.sigma_norm ** 2)

        for k in range(n_act):
            iy = k // self.n_act_x
            ix = k  % self.n_act_x
            xa = self.act_x_norm[ix]
            ya = self.act_y_norm[iy]
            r2 = (Xg - xa) ** 2 + (Yg - ya) ** 2
            F_full[:, k] = np.exp(-r2.ravel() * inv2s2)

        self.F_full = F_full.astype(np.float32)
        self.F_pup  = F_full[self._pupil_flat_idx, :]     # [n_pup, n_act]

        # SVD pseudo-inverse of F_pup
        U, sv, Vt = svd(self.F_pup, full_matrices=False)
        self._singular_values = sv
        thr    = self.svd_threshold * sv[0]
        sv_inv = np.where(sv > thr, 1.0 / sv, 0.0)
        n_kept = int((sv > thr).sum())

        self.F_pinv = ((Vt.T * sv_inv) @ U.T).astype(np.float32)  # [n_act, n_pup]

        log.info("SVD: kept %d/%d vectors (threshold %.1e × max_sv=%.3e)",
                 n_kept, n_act, self.svd_threshold, sv[0])
        log.info("IF condition number: %.2e", sv[0] / (sv[n_kept - 1] + 1e-30))

    # ── Per-frame operations ─────────────────────────────────────────────────

    def compute_commands(self,
                         phase_pupil_flat: np.ndarray,
                         ) -> Tuple[np.ndarray, bool]:
        """
        Compute actuator commands for one frame.

        Parameters
        ----------
        phase_pupil_flat : float [n_pupil]  piston-free wavefront in radians

        Returns
        -------
        u        : float [n_act]  commands in radians
        clipped  : bool  True if any actuator hit the stroke limit
        """
        u = -(self.F_pinv @ phase_pupil_flat.astype(np.float32))
        u -= u.mean()                                      # remove piston
        clipped = bool(np.any(np.abs(u) > self.max_stroke_rad))
        u = np.clip(u, -self.max_stroke_rad, self.max_stroke_rad)
        return u, clipped

    def apply_correction(self,
                         phase_2d: np.ndarray,
                         u:        np.ndarray,
                         ) -> np.ndarray:
        """
        Compute corrected wavefront W_corr = W_input + F @ u.

        The DM conjugates the aberration: the mirror command is the
        NEGATIVE of the wavefront, so the corrected wavefront is the
        residual W + (F u) ≈ 0 for a perfect correction.

        Parameters
        ----------
        phase_2d : float [G, G]  input phase
        u        : float [n_act] actuator commands
        """
        mirror_phase = (self.F_full @ u).reshape(self.G, self.G)
        return (phase_2d + mirror_phase).astype(np.float32)

    # ── Batch run ────────────────────────────────────────────────────────────

    def run(self,
            wf_result,
            wavelength_m: float = 633e-9,
            ) -> DmResult:
        """
        Compute actuator commands and corrected wavefront for all frames.

        Parameters
        ----------
        wf_result    : WavefrontResult from Stage 2
        wavelength_m : sensing wavelength [m]

        Returns
        -------
        DmResult
        """
        assert self.F_pinv is not None, "Call calibrate() before run()"

        n_frames = wf_result.n_frames
        G        = self.G
        mask     = self._pupil_mask

        act_cmds     = np.zeros((n_frames, self.n_act_y, self.n_act_x), dtype=np.float32)
        corr_phase   = np.zeros((n_frames, G, G),                        dtype=np.float32)
        uncorr_strehl= np.zeros(n_frames, dtype=np.float32)
        corr_strehl  = np.zeros(n_frames, dtype=np.float32)
        uncorr_wfe   = np.zeros(n_frames, dtype=np.float32)
        corr_wfe     = np.zeros(n_frames, dtype=np.float32)

        n_clipped = 0

        for i in range(n_frames):
            W = wf_result.phase_maps[i]          # [G, G]

            # Piston-free wavefront
            W_pupil = W[mask]
            W0      = W - W_pupil.mean()

            # Uncorrected metrics (piston removed)
            phi0    = W0[mask] - W0[mask].mean()
            s0      = float(np.sqrt(np.mean(phi0 ** 2)))
            uncorr_wfe[i]    = s0
            uncorr_strehl[i] = float(np.exp(-s0 ** 2))

            # Actuator commands
            u, clipped = self.compute_commands(W0[mask])
            n_clipped  += int(clipped)
            act_cmds[i] = u.reshape(self.n_act_y, self.n_act_x)

            # Corrected wavefront
            Wc       = self.apply_correction(W0, u)
            corr_phase[i] = Wc

            # Corrected metrics
            phi1 = Wc[mask] - Wc[mask].mean()
            s1   = float(np.sqrt(np.mean(phi1 ** 2)))
            corr_wfe[i]    = s1
            corr_strehl[i] = float(np.exp(-s1 ** 2))

        # Physical stroke conversion: h[m] = u_rad * λ/(4π)
        lam_over_4pi = wavelength_m / (4.0 * math.pi)
        stroke_um = (act_cmds * lam_over_4pi * 1e6).astype(np.float32)

        sat_frac = float(n_clipped) / max(n_frames, 1)

        log.info(
            "Stage 4 complete: WFE %.2f→%.2f mrad (%.1f×)  "
            "Strehl %.4f→%.4f  saturation %.1f%%",
            uncorr_wfe.mean() * 1e3,
            corr_wfe.mean()   * 1e3,
            uncorr_wfe.mean() / (corr_wfe.mean() + 1e-12),
            uncorr_strehl.mean(),
            corr_strehl.mean(),
            sat_frac * 100,
        )

        return DmResult(
            actuator_commands   = act_cmds,
            stroke_um           = stroke_um,
            corrected_phase     = corr_phase,
            uncorrected_strehl  = uncorr_strehl,
            corrected_strehl    = corr_strehl,
            uncorrected_wfe_rms = uncorr_wfe,
            corrected_wfe_rms   = corr_wfe,
            pupil_mask          = mask,
            timestamps_ms       = wf_result.timestamps_ms,
            n_act_x             = self.n_act_x,
            n_act_y             = self.n_act_y,
            saturation_frac     = sat_frac,
        )

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        np.savez_compressed(
            str(path),
            F_full     = self.F_full,
            F_pinv     = self.F_pinv,
            act_x_norm = self.act_x_norm,
            act_y_norm = self.act_y_norm,
            sigma_norm = np.array([self.sigma_norm]),
            n_act_x    = np.array([self.n_act_x]),
            n_act_y    = np.array([self.n_act_y]),
            d_act      = np.array([self.d_act]),
        )
        log.info("Saved ActuatorMapper → %s", path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "ActuatorMapper":
        data = np.load(str(path))
        obj  = cls(
            n_act_x        = int(data["n_act_x"][0]),
            n_act_y        = int(data["n_act_y"][0]),
            d_act_norm     = float(data["d_act"][0]),
            output_grid_px = kwargs.get("output_grid_px", 64),
        )
        obj.F_full       = data["F_full"]
        obj.F_pinv       = data["F_pinv"]
        obj.act_x_norm   = data["act_x_norm"]
        obj.act_y_norm   = data["act_y_norm"]
        obj.sigma_norm   = float(data["sigma_norm"][0])
        log.info("Loaded ActuatorMapper from %s", path)
        return obj