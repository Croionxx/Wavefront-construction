"""
wavefront_reconstructor.py — Stage 2: Wavefront Reconstruction
===============================================================
Reconstructs the wavefront phase from the SH-WFS slope measurements
produced by Stage 1.

Two reconstruction strategies are implemented, selectable via config:

  1. ``ModalReconstructor``  — Zernike polynomial decomposition.
     Projects the slope vector onto a Zernike interaction matrix and
     returns per-frame Zernike coefficients plus a dense phase map.

  2. ``ZonalReconstructor``  — Fried geometry least-squares.
     Reconstructs wavefront values at actuator-grid nodes directly
     from the slope equations, without modal decomposition.

Both share the same public interface so they can be swapped
transparently.

Physical conventions
--------------------
* All phases are in **radians** (optical path in nm can be recovered
  by multiplying by wavelength/(2*pi)).
* Slopes ``s_x, s_y`` entering this module are already in **radians
  of wavefront tilt** (output of SlopeComputer, Stage 1).
* Zernike polynomials are normalised in the Noll (1976) convention
  over a unit circle (orthonormal: integral over unit disk = pi).
* Piston (j=1) is always excluded — SH-WFS cannot sense it.

Design notes
------------
* Reconstructor matrices are precomputed once (``calibrate()``) and
  stored as ``.npy`` binaries so Stage 2 can be run without regenerating
  them.
* SVD-based pseudo-inverse is used for both methods, with a tunable
  singular-value threshold.
* Modal reconstruction uses an analytic interaction matrix (Zernike
  gradients evaluated at sub-aperture centres) — no calibration frames
  needed.
* Zonal reconstruction uses the sparse Fried geometry matrix A which
  maps wavefront node differences to sub-aperture slope averages.
* A ``WavefrontResult`` dataclass encapsulates all per-frame outputs and
  is the data contract passed to Stage 3.

References
----------
* Noll 1976 — Zernike polynomials and atmospheric turbulence, JOSA 66(3)
* Southwell 1980 — Wavefront estimation from slope measurements, JOSA 70(8)
* Hardy 1998 — Adaptive Optics for Astronomical Telescopes, Ch. 5
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import svd
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr as sparse_lsqr

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Output data structure
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WavefrontResult:
    """
    Per-sequence reconstruction output.  Data contract for Stage 3.

    Attributes
    ----------
    phase_maps  : float32 [N_frames, H_recon, W_recon]
                  Reconstructed wavefront phase in radians.
                  For modal: evaluated on a rectangular output grid.
                  For zonal: values at Fried actuator nodes, reshaped
                  to (n_sa_y+1, n_sa_x+1).
    zernike_coeffs : float32 [N_frames, J] or None
                  Zernike coefficients a_j (j=2..J+1, piston excluded).
                  None for zonal reconstruction.
    residual_rms : float32 [N_frames]
                  RMS of the slope residual (measured - reconstructed)
                  in radians.  Proxy for reconstruction quality.
    timestamps_ms : float32 [N_frames]
    n_modes      : int   Number of Zernike modes (0 if zonal).
    method       : str   'modal_zernike' | 'zonal_fried'
    pupil_mask   : bool  [H_recon, W_recon] — True inside pupil.
    strehl_estimate : float32 [N_frames]
                  Maréchal approximation: S ≈ exp(−σ²_φ).
    """
    phase_maps:       np.ndarray          # [N, H, W]  float32
    zernike_coeffs:   Optional[np.ndarray]  # [N, J]  float32 or None
    residual_rms:     np.ndarray          # [N]  float32
    timestamps_ms:    np.ndarray          # [N]  float32
    n_modes:          int
    method:           str
    pupil_mask:       np.ndarray          # [H, W]  bool
    strehl_estimate:  np.ndarray          # [N]  float32

    @property
    def n_frames(self) -> int:
        return self.phase_maps.shape[0]

    @property
    def phase_rms(self) -> np.ndarray:
        """RMS wavefront error per frame [rad], ignoring outside-pupil pixels."""
        rms = np.zeros(self.n_frames, dtype=np.float32)
        mask = self.pupil_mask
        for i in range(self.n_frames):
            phi = self.phase_maps[i][mask]
            rms[i] = float(np.sqrt(np.mean(phi**2)))
        return rms

    def print_summary(self):
        rms = self.phase_rms
        sr  = self.strehl_estimate
        print(
            f"\nWavefront Reconstruction Summary\n"
            f"  Method         : {self.method}\n"
            f"  Frames         : {self.n_frames}\n"
            f"  Modes / nodes  : {self.n_modes}\n"
            f"  RMS WFE        : {rms.mean()*1e3:.2f} ± {rms.std()*1e3:.2f} mrad\n"
            f"  Strehl (mean)  : {sr.mean():.3f}\n"
            f"  Slope residual : {self.residual_rms.mean()*1e3:.2f} mrad (mean)\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Zernike polynomial library
# ═══════════════════════════════════════════════════════════════════════════════

class ZernikeLib:
    """
    Compute Noll-indexed Zernike polynomials and their gradients over a
    unit disk, evaluated at arbitrary (rho, theta) points.

    Noll ordering (JOSA 1976):
        j=1  piston
        j=2  tip (x-tilt)
        j=3  tilt (y-tilt)
        j=4  defocus
        j=5  astigmatism-45
        j=6  astigmatism-0
        ...

    All polynomials are NORMALISED so that integral over unit disk = pi
    (the Noll normalisation), which means they form an orthonormal set
    under the inner product <f,g> = integral(f*g dA)/pi.

    The gradient formulae dZ_j/dx and dZ_j/dy are derived analytically
    (no finite differences), which avoids numerical differentiation
    artefacts for the interaction matrix.
    """

    @staticmethod
    def noll_to_nm(j: int) -> Tuple[int, int]:
        """
        Convert Noll index j (1-based) to radial order n and azimuthal
        frequency m (signed).

        Algorithm: Noll 1976, Table 1.
        """
        n = int(math.ceil((-3 + math.sqrt(9 + 8*(j-1))) / 2))
        j_n_start = n * (n + 1) // 2 + 1
        delta     = j - j_n_start   # 0-based offset within radial order n
        # m runs over: -n, -(n-2), ..., (n-2), n  — see Noll Table 1
        # Even j → cos term (m>0 or m=0), odd j → sin term (m<0)
        if n % 2 == 0:
            ms = list(range(0, n+1, 2))
        else:
            ms = list(range(1, n+1, 2))
        # The pairs go: ...(+m, -m)... inside the radial order
        # Noll's specific interleaving: see his Table 1
        m_abs = ms[delta // 2]
        if delta % 2 == 0:
            m = m_abs if j % 2 == 0 else -m_abs
        else:
            m = -m_abs if j % 2 == 0 else m_abs
        return n, m

    @staticmethod
    def radial(n: int, abs_m: int, rho: np.ndarray) -> np.ndarray:
        """
        Evaluate the radial polynomial R_n^m(rho) using the recurrence:
            R_n^m = sum_{s=0}^{(n-m)/2} (-1)^s * C(n-s,s) * C(n-2s,(n-m)/2-s)
                    * rho^(n-2s)
        where C is "n choose k".
        """
        R = np.zeros_like(rho)
        half = (n - abs_m) // 2
        for s in range(half + 1):
            coeff = ((-1)**s
                     * math.comb(n - s, s)
                     * math.comb(n - 2*s, half - s))
            R += coeff * rho**(n - 2*s)
        return R

    @staticmethod
    def norm_coeff(n: int, m: int) -> float:
        """Noll normalisation coefficient."""
        eps = 1 if m == 0 else 0
        return math.sqrt((2*(n+1)) / (1 + eps))

    @classmethod
    def evaluate(cls,
                 j: int,
                 rho: np.ndarray,
                 theta: np.ndarray) -> np.ndarray:
        """
        Evaluate Zernike polynomial Z_j at polar coordinates (rho, theta).
        Points outside the unit disk (rho > 1) are set to zero.
        """
        n, m = cls.noll_to_nm(j)
        abs_m = abs(m)
        R = cls.radial(n, abs_m, rho)
        N = cls.norm_coeff(n, m)
        if m > 0:
            Z = N * R * np.cos(abs_m * theta)
        elif m < 0:
            Z = N * R * np.sin(abs_m * theta)
        else:
            Z = N * R
        Z[rho > 1.0] = 0.0
        return Z

    @classmethod
    def gradient(cls,
                 j: int,
                 x: np.ndarray,
                 y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analytic partial derivatives dZ_j/dx and dZ_j/dy at Cartesian
        positions (x, y) on the unit disk.

        Strategy: use the chain rule in polar coordinates:
            dZ/dx = dZ/drho * drho/dx  +  dZ/dtheta * dtheta/dx
            dZ/dy = dZ/drho * drho/dy  +  dZ/dtheta * dtheta/dy

        where drho/dx = x/rho, drho/dy = y/rho,
              dtheta/dx = -y/rho², dtheta/dy = x/rho².

        dZ/drho is computed from the derivative of R_n^m:
            dR/drho = sum_{s} coeff * (n-2s) * rho^(n-2s-1)

        dZ/dtheta = ±m * (sin or cos term) depending on sign(m).
        """
        rho   = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        n, m  = cls.noll_to_nm(j)
        abs_m = abs(m)
        N     = cls.norm_coeff(n, m)
        R     = cls.radial(n, abs_m, rho)
        half  = (n - abs_m) // 2

        # dR/drho
        dR = np.zeros_like(rho)
        for s in range(half + 1):
            power = n - 2*s
            if power > 0:
                coeff = ((-1)**s
                         * math.comb(n - s, s)
                         * math.comb(n - 2*s, half - s))
                dR += coeff * power * rho**(power - 1)

        # Angular part and its derivative
        if m > 0:
            ang  =  np.cos(abs_m * theta)
            dang = -abs_m * np.sin(abs_m * theta)
        elif m < 0:
            ang  =  np.sin(abs_m * theta)
            dang =  abs_m * np.cos(abs_m * theta)
        else:
            ang  = np.ones_like(rho)
            dang = np.zeros_like(rho)

        # Guard against rho=0 singularity
        safe_rho = np.where(rho > 1e-10, rho, 1e-10)

        dZ_drho   = N * dR * ang
        dZ_dtheta = N * R  * dang

        dZ_dx = (dZ_drho * x / safe_rho) + (dZ_dtheta * (-y / safe_rho**2))
        dZ_dy = (dZ_drho * y / safe_rho) + (dZ_dtheta * ( x / safe_rho**2))

        # Zero outside pupil
        outside = rho > 1.0
        dZ_dx[outside] = 0.0
        dZ_dy[outside] = 0.0

        return dZ_dx, dZ_dy


# ═══════════════════════════════════════════════════════════════════════════════
# Modal reconstructor (Zernike)
# ═══════════════════════════════════════════════════════════════════════════════

class ModalReconstructor:
    """
    Reconstruct the wavefront as a sum of Zernike polynomials.

        W(x, y) = sum_{j=2}^{J+1} a_j Z_j(x/R, y/R)

    The interaction matrix D ∈ R^(2*N_sa × J) maps Zernike coefficients
    a ∈ R^J to the slope vector s ∈ R^(2*N_sa):

        s = D a   →   a_hat = D^+ s

    D is constructed analytically from the partial derivatives of each
    Zernike evaluated at the sub-aperture centre positions, normalised
    to the pupil radius.

    After calibration:
        ``self.D``   [2*N_sa, J]   interaction matrix
        ``self.Ddag``[J, 2*N_sa]   pseudo-inverse (reconstructor)

    Parameters
    ----------
    n_modes : int
        Number of Zernike modes to fit (j = 2..n_modes+1, piston excluded).
        Recommended: keep below N_sa_valid to avoid ill-conditioning.
    svd_threshold : float
        Singular values below this fraction of the largest are zeroed
        in the pseudo-inverse.  Tune between 1e-4 (more modes, more noise)
        and 1e-2 (fewer modes, smoother).
    output_grid_px : int
        Side length of the square output phase map (in pixels).
        The pupil is mapped to a unit disk on this grid.
    """

    def __init__(self,
                 n_modes:        int   = 21,
                 svd_threshold:  float = 1e-3,
                 output_grid_px: int   = 64):
        self.n_modes        = n_modes
        self.svd_threshold  = svd_threshold
        self.output_grid_px = output_grid_px

        self.D:    Optional[np.ndarray] = None  # [2*N_sa, J]
        self.Ddag: Optional[np.ndarray] = None  # [J, 2*N_sa]
        self._zernike_map:  Optional[np.ndarray] = None  # [J, H, W]
        self._output_pupil: Optional[np.ndarray] = None  # [H, W] bool

        # Stored after calibrate()
        self._n_sa_valid: int = 0
        self._singular_values: Optional[np.ndarray] = None
        self._condition_number: float = 0.0

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate(self,
                  sa_cx_norm: np.ndarray,
                  sa_cy_norm: np.ndarray,
                  valid_mask: np.ndarray,
                  pupil_radius_px: float = 1.0):
        """
        Build the interaction matrix D from sub-aperture centre positions.

        Parameters
        ----------
        sa_cx_norm, sa_cy_norm : float [N_sa_total]
            Sub-aperture centres in NORMALISED pupil coordinates
            (i.e., divided by pupil_radius so they lie in [-1, 1]).
            Caller computes: (cx_px - pupil_cx_px) / pupil_radius_px.
        valid_mask : bool [N_sa_total]
            True for sub-apertures inside the pupil.
        pupil_radius_px : float
            Pupil radius in pixels (kept for phase-map evaluation).
        """
        x = sa_cx_norm[valid_mask]
        y = sa_cy_norm[valid_mask]
        N = len(x)
        J = self.n_modes

        self._n_sa_valid = N

        log.info("ModalReconstructor: building interaction matrix "
                 "[%d × %d] for %d Zernike modes", 2*N, J, J)

        D = np.zeros((2 * N, J), dtype=np.float64)

        for k in range(J):
            j_noll = k + 2           # j=2..J+1 (skip piston j=1)
            dZx, dZy = ZernikeLib.gradient(j_noll, x, y)
            D[:N, k] = dZx           # x-slope block
            D[N:, k] = dZy           # y-slope block

        self.D = D.astype(np.float32)

        # SVD pseudo-inverse
        U, sv, Vt = svd(D, full_matrices=False)
        self._singular_values = sv
        self._condition_number = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf

        threshold = self.svd_threshold * sv[0]
        sv_inv    = np.where(sv > threshold, 1.0 / sv, 0.0)
        n_kept    = int((sv > threshold).sum())

        self.Ddag = (Vt.T * sv_inv) @ U.T   # [J, 2*N_sa]
        self.Ddag = self.Ddag.astype(np.float32)

        log.info("  Singular values: min=%.3e  max=%.3e  condition=%.2e",
                 sv[-1], sv[0], self._condition_number)
        log.info("  Kept %d/%d singular values (threshold=%.2e * max)",
                 n_kept, J, self.svd_threshold)

        # Pre-evaluate Zernike polynomials on the output grid
        self._build_zernike_map()

    def _build_zernike_map(self):
        """
        Pre-evaluate Z_j on a square output grid (unit-disk normalised).
        Stored as [J, H, W] float32 for fast per-frame summation.
        """
        G = self.output_grid_px
        lin = np.linspace(-1.0, 1.0, G)
        Xg, Yg = np.meshgrid(lin, lin)   # origin at centre

        rho   = np.sqrt(Xg**2 + Yg**2)
        theta = np.arctan2(Yg, Xg)

        self._output_pupil = (rho <= 1.0)

        J = self.n_modes
        Zmap = np.zeros((J, G, G), dtype=np.float32)
        for k in range(J):
            j_noll = k + 2
            Zmap[k] = ZernikeLib.evaluate(j_noll, rho, theta).astype(np.float32)

        self._zernike_map = Zmap   # [J, G, G]
        log.debug("Zernike map shape: %s", Zmap.shape)

    # ── Per-frame reconstruction ─────────────────────────────────────────────

    def reconstruct_frame(self,
                          s_x: np.ndarray,
                          s_y: np.ndarray,
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct wavefront for a single frame.

        Parameters
        ----------
        s_x, s_y : float [N_sa_valid]
            Slopes in radians (output of SlopeComputer, valid SAs only).

        Returns
        -------
        phase_map   : float32 [G, G]  wavefront in radians on output grid
        coeffs      : float32 [J]     Zernike coefficients
        slope_resid : float32 [2*N]   slope residual (for RMS computation)
        """
        assert self.Ddag is not None, "Call calibrate() first"

        # Assemble slope vector [sx | sy]
        s = np.concatenate([s_x, s_y]).astype(np.float32)   # [2*N_sa]

        # Zernike coefficients
        coeffs = self.Ddag @ s    # [J]

        # Phase map from Zernike sum: W = sum_j a_j Z_j
        # _zernike_map [J, G, G], coeffs [J] → phase [G, G]
        phase_map = np.einsum('j,jhw->hw', coeffs, self._zernike_map)

        # Slope residual: what fraction of the slopes does the modal
        # decomposition NOT capture?
        s_reconstructed = self.D @ coeffs
        slope_resid     = s - s_reconstructed

        return phase_map, coeffs, slope_resid

    # ── Batch reconstruction ─────────────────────────────────────────────────

    def reconstruct(self,
                    s_x_all:   np.ndarray,
                    s_y_all:   np.ndarray,
                    valid_all: np.ndarray,
                    timestamps_ms: np.ndarray,
                    ) -> WavefrontResult:
        """
        Reconstruct all frames.

        Parameters
        ----------
        s_x_all   : float32 [N_frames, N_sa_valid]
        s_y_all   : float32 [N_frames, N_sa_valid]
        valid_all : bool    [N_frames, N_sa_valid]
        timestamps_ms : float [N_frames]

        Returns
        -------
        WavefrontResult
        """
        assert self.Ddag is not None, "Call calibrate() first"

        n_frames  = s_x_all.shape[0]
        G         = self.output_grid_px
        J         = self.n_modes

        phase_maps     = np.zeros((n_frames, G, G), dtype=np.float32)
        coeffs_all     = np.zeros((n_frames, J),    dtype=np.float32)
        residual_rms   = np.zeros(n_frames,          dtype=np.float32)
        strehl         = np.zeros(n_frames,          dtype=np.float32)

        for i in range(n_frames):
            sx = s_x_all[i]
            sy = s_y_all[i]

            # Zero out invalid sub-apertures rather than skipping —
            # the reconstruction degrades gracefully with patchy validity.
            v = valid_all[i]
            sx = np.where(v, sx, 0.0)
            sy = np.where(v, sy, 0.0)

            pm, co, sr = self.reconstruct_frame(sx, sy)
            phase_maps[i]   = pm
            coeffs_all[i]   = co
            residual_rms[i] = float(np.sqrt(np.mean(sr**2)))

            # Maréchal Strehl estimate: S ≈ exp(−σ²_φ) over pupil
            phi_pupil = pm[self._output_pupil]
            phi_pupil -= phi_pupil.mean()   # remove piston
            sigma2    = float(np.mean(phi_pupil**2))
            strehl[i] = float(np.exp(-sigma2))

        log.info("Modal reconstruction complete: %d frames, "
                 "mean residual RMS = %.2e rad",
                 n_frames, residual_rms.mean())

        return WavefrontResult(
            phase_maps      = phase_maps,
            zernike_coeffs  = coeffs_all,
            residual_rms    = residual_rms,
            timestamps_ms   = timestamps_ms.astype(np.float32),
            n_modes         = J,
            method          = "modal_zernike",
            pupil_mask      = self._output_pupil,
            strehl_estimate = strehl,
        )

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save calibrated reconstructor matrices to a .npz file."""
        np.savez_compressed(
            str(path),
            D            = self.D,
            Ddag         = self.Ddag,
            zernike_map  = self._zernike_map,
            output_pupil = self._output_pupil,
            singular_values = self._singular_values,
            n_modes      = np.array([self.n_modes]),
            svd_threshold= np.array([self.svd_threshold]),
            output_grid_px = np.array([self.output_grid_px]),
        )
        log.info("Saved modal reconstructor to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ModalReconstructor":
        """Load a previously calibrated reconstructor from .npz."""
        data = np.load(str(path))
        obj  = cls(
            n_modes        = int(data["n_modes"][0]),
            svd_threshold  = float(data["svd_threshold"][0]),
            output_grid_px = int(data["output_grid_px"][0]),
        )
        obj.D               = data["D"]
        obj.Ddag            = data["Ddag"]
        obj._zernike_map    = data["zernike_map"]
        obj._output_pupil   = data["output_pupil"].astype(bool)
        obj._singular_values = data["singular_values"]
        log.info("Loaded modal reconstructor from %s  (%d modes)",
                 path, obj.n_modes)
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# Zonal reconstructor (Fried geometry)
# ═══════════════════════════════════════════════════════════════════════════════

class ZonalReconstructor:
    """
    Reconstruct wavefront values at the (n_sa_x+1)×(n_sa_y+1) Fried
    geometry nodes directly from the slope equations.

    The Fried geometry equations (Southwell 1980) relate slope
    measurements to phase differences:

        sx[j,i] ≈ (phi[j, i+1] − phi[j, i]) / d_sa    (x-slope)
        sy[j,i] ≈ (phi[j+1, i] − phi[j, i]) / d_sa    (y-slope)

    More precisely, in the symmetric Fried formulation the x-slope of
    sub-aperture (i,j) is the average of the x-differences at both
    y-edges:

        sx[j,i] = ( (phi[j,i+1]  − phi[j,i])
                   +(phi[j+1,i+1]− phi[j+1,i]) ) / (2 * d_sa)

    This gives the least-squares system  A Φ = s  solved via the
    sparse SVD pseudo-inverse (LSQR).

    The piston null space is removed by pinning the mean of in-pupil
    nodes to zero (removes the single-dimensional null space of A).

    Parameters
    ----------
    n_sa_x, n_sa_y : int
        Number of sub-apertures in each direction.
    d_sa : float
        Physical sub-aperture size [m].  Used to convert slopes [rad]
        to phase differences [rad·m / m = rad].
    svd_threshold : float
        Singular value truncation fraction for the dense SVD fallback.
        The default LSQR solver ignores this.
    use_lsqr : bool
        If True (default), solve with scipy's LSQR iterative solver
        (fast for large grids).  If False, use dense SVD pseudo-inverse
        (better numerical diagnostics, slower for >20×20 grids).
    """

    def __init__(self,
                 n_sa_x:       int,
                 n_sa_y:       int,
                 d_sa:         float,
                 svd_threshold: float = 1e-3,
                 use_lsqr:     bool  = True):
        self.n_sa_x       = n_sa_x
        self.n_sa_y       = n_sa_y
        self.d_sa         = d_sa
        self.svd_threshold = svd_threshold
        self.use_lsqr     = use_lsqr

        # Node grid: (n_sa_y+1) × (n_sa_x+1)
        self.n_nodes_x    = n_sa_x + 1
        self.n_nodes_y    = n_sa_y + 1
        self.n_nodes      = self.n_nodes_x * self.n_nodes_y

        self.A:    Optional[np.ndarray] = None   # dense [2*N_sa, N_nodes]
        self.Adag: Optional[np.ndarray] = None   # [N_nodes, 2*N_sa]  (dense SVD path)
        self._pupil_node_mask: Optional[np.ndarray] = None  # [n_y+1, n_x+1]

        self._singular_values: Optional[np.ndarray] = None

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate(self, pupil_sa_mask: np.ndarray):
        """
        Build the Fried geometry matrix A.

        Parameters
        ----------
        pupil_sa_mask : bool [n_sa_y, n_sa_x]
            True for sub-apertures inside the pupil.
        """
        Nx  = self.n_sa_x
        Ny  = self.n_sa_y
        NNx = self.n_nodes_x
        NNy = self.n_nodes_y
        Nn  = self.n_nodes

        # Count valid sub-apertures
        n_valid = int(pupil_sa_mask.sum())
        n_meas  = 2 * n_valid

        log.info("ZonalReconstructor: building Fried matrix "
                 "[%d × %d] for %dx%d SA grid  (%d valid SAs)",
                 n_meas, Nn, Nx, Ny, n_valid)

        # Build sparse matrix row by row
        A_sp = lil_matrix((n_meas, Nn), dtype=np.float64)

        row_x = 0   # row index for x-slopes
        row_y = n_valid   # row index for y-slopes (stacked below x-slopes)

        scale = 1.0 / (2.0 * self.d_sa)

        for jj in range(Ny):
            for ii in range(Nx):
                if not pupil_sa_mask[jj, ii]:
                    continue

                # Node indices at the four corners of sub-aperture (ii, jj)
                # Corner layout:
                #   (jj,   ii)  (jj,   ii+1)
                #   (jj+1, ii)  (jj+1, ii+1)
                def nidx(r, c): return r * NNx + c

                n00 = nidx(jj,   ii)
                n01 = nidx(jj,   ii + 1)
                n10 = nidx(jj+1, ii)
                n11 = nidx(jj+1, ii + 1)

                # Symmetric Fried x-slope:
                #   sx = [(phi_01 - phi_00) + (phi_11 - phi_10)] / (2*d)
                A_sp[row_x, n01] +=  scale
                A_sp[row_x, n00] += -scale
                A_sp[row_x, n11] +=  scale
                A_sp[row_x, n10] += -scale

                # Symmetric Fried y-slope:
                #   sy = [(phi_10 - phi_00) + (phi_11 - phi_01)] / (2*d)
                A_sp[row_y, n10] +=  scale
                A_sp[row_y, n00] += -scale
                A_sp[row_y, n11] +=  scale
                A_sp[row_y, n01] += -scale

                row_x += 1
                row_y += 1

        # Convert to CSR for efficient arithmetic
        A_csr = A_sp.tocsr()
        self.A = A_csr.toarray().astype(np.float32)

        # Pupil node mask: a node is "in-pupil" if any adjacent SA is valid
        node_mask = np.zeros((NNy, NNx), dtype=bool)
        for jj in range(Ny):
            for ii in range(Nx):
                if pupil_sa_mask[jj, ii]:
                    node_mask[jj,   ii  ] = True
                    node_mask[jj,   ii+1] = True
                    node_mask[jj+1, ii  ] = True
                    node_mask[jj+1, ii+1] = True
        self._pupil_node_mask = node_mask

        if not self.use_lsqr:
            # Dense SVD pseudo-inverse
            U, sv, Vt = svd(self.A, full_matrices=False)
            self._singular_values = sv
            threshold = self.svd_threshold * sv[0]
            sv_inv    = np.where(sv > threshold, 1.0 / sv, 0.0)
            n_kept    = int((sv > threshold).sum())
            self.Adag = (Vt.T * sv_inv) @ U.T
            self.Adag = self.Adag.astype(np.float32)
            log.info("  Dense SVD: kept %d/%d singular values", n_kept, len(sv))
        else:
            log.info("  Using LSQR iterative solver (use_lsqr=True)")

    # ── Per-frame reconstruction ─────────────────────────────────────────────

    def reconstruct_frame(self,
                          s_x: np.ndarray,
                          s_y: np.ndarray,
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct wavefront node values for a single frame.

        Parameters
        ----------
        s_x, s_y : float [N_sa_valid]  slopes in radians (valid SAs only,
                   ordered row-major to match the order used in calibrate())

        Returns
        -------
        phi_2d   : float32 [n_sa_y+1, n_sa_x+1]  phase at node grid
        slope_resid : float [2*N_sa_valid]
        """
        assert self.A is not None, "Call calibrate() first"

        s = np.concatenate([s_x, s_y]).astype(np.float64)

        if self.use_lsqr:
            # LSQR: faster, iterative, good for large grids
            sol = sparse_lsqr(self.A, s, atol=1e-9, btol=1e-9, iter_lim=1000)
            phi = sol[0].astype(np.float32)
        else:
            phi = (self.Adag @ s).astype(np.float32)

        # Remove piston (set mean of in-pupil nodes to zero)
        mask_flat = self._pupil_node_mask.ravel()
        phi -= phi[mask_flat].mean()

        slope_resid = (s - self.A @ phi.astype(np.float64)).astype(np.float32)

        phi_2d = phi.reshape(self.n_nodes_y, self.n_nodes_x)
        return phi_2d, slope_resid

    # ── Batch reconstruction ─────────────────────────────────────────────────

    def reconstruct(self,
                    s_x_all:   np.ndarray,
                    s_y_all:   np.ndarray,
                    valid_all: np.ndarray,
                    timestamps_ms: np.ndarray,
                    ) -> WavefrontResult:
        """Reconstruct all frames. Same signature as ModalReconstructor."""
        assert self.A is not None, "Call calibrate() first"

        n_frames  = s_x_all.shape[0]
        NNy       = self.n_nodes_y
        NNx       = self.n_nodes_x

        phase_maps   = np.zeros((n_frames, NNy, NNx), dtype=np.float32)
        residual_rms = np.zeros(n_frames, dtype=np.float32)
        strehl       = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            sx = np.where(valid_all[i], s_x_all[i], 0.0)
            sy = np.where(valid_all[i], s_y_all[i], 0.0)

            pm, sr = self.reconstruct_frame(sx, sy)
            phase_maps[i]   = pm
            residual_rms[i] = float(np.sqrt(np.mean(sr**2)))

            phi_pupil = pm[self._pupil_node_mask]
            phi_pupil -= phi_pupil.mean()
            strehl[i] = float(np.exp(-float(np.mean(phi_pupil**2))))

        log.info("Zonal reconstruction complete: %d frames, "
                 "mean residual RMS = %.2e rad",
                 n_frames, residual_rms.mean())

        return WavefrontResult(
            phase_maps      = phase_maps,
            zernike_coeffs  = None,
            residual_rms    = residual_rms,
            timestamps_ms   = timestamps_ms.astype(np.float32),
            n_modes         = self.n_nodes,
            method          = "zonal_fried",
            pupil_mask      = self._pupil_node_mask,
            strehl_estimate = strehl,
        )

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        np.savez_compressed(
            str(path),
            A                = self.A,
            Adag             = self.Adag if self.Adag is not None else np.array([]),
            pupil_node_mask  = self._pupil_node_mask,
            n_sa_x           = np.array([self.n_sa_x]),
            n_sa_y           = np.array([self.n_sa_y]),
            d_sa             = np.array([self.d_sa]),
            use_lsqr         = np.array([int(self.use_lsqr)]),
        )
        log.info("Saved zonal reconstructor to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ZonalReconstructor":
        data = np.load(str(path))
        obj  = cls(
            n_sa_x   = int(data["n_sa_x"][0]),
            n_sa_y   = int(data["n_sa_y"][0]),
            d_sa     = float(data["d_sa"][0]),
            use_lsqr = bool(data["use_lsqr"][0]),
        )
        obj.A                 = data["A"]
        obj._pupil_node_mask  = data["pupil_node_mask"].astype(bool)
        if data["Adag"].size > 0:
            obj.Adag = data["Adag"]
        log.info("Loaded zonal reconstructor from %s", path)
        return obj
