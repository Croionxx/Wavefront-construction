"""
turbulence_characterizer.py — Stage 3: Turbulence Characterization
=====================================================================
Consumes Stage 1 (slopes) and Stage 2 (reconstructed phase / Zernike
coefficients) outputs and estimates the key turbulence parameters:

  * r0   — Fried parameter, via TWO independent methods:
             (a) Zernike coefficient variance fit  (Noll 1976)
             (b) Spatial slope structure function fit (Southwell/Fried)
  * tau0 — coherence time, via TWO independent methods:
             (a) Temporal structure function fit (+ wind speed)
             (b) 1/e autocorrelation crossing
  * v_wind — effective wind speed (from the temporal SF fit + r0)

Physical conventions
---------------------
* Zernike coefficients a_j (from Stage 2 ModalReconstructor) are in
  RADIANS of true phase — no extra conversion needed.
* Stage 1 slopes s_x, s_y are TILT ANGLES in radians, i.e.
      s_x = (wavelength / 2*pi) * d(phase)/dx
  (see run_stage1.py fix notes). To recover a phase-difference proxy
  between adjacent sub-apertures separated by d_sa:
      delta_phi ≈ s_x * (2*pi / wavelength) * d_sa
* All r0 outputs are in the SAME units as the aperture_diameter /
  lenslet_size supplied (meters, per sensor_config.yaml).

References
----------
* Noll 1976 — Zernike polynomials and atmospheric turbulence, JOSA 66(3)
* Fried 1965 — Statistics of a geometric representation of wavefront
  distortion, JOSA 55(11)
* Hardy 1998 — Adaptive Optics for Astronomical Telescopes, Ch. 3/5
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import gamma

log = logging.getLogger(__name__)

# Kolmogorov phase structure function constant: D_phi(r) = 6.88 (r/r0)^(5/3)
KOLMOGOROV_CONST = 6.88

# von Karman / Kolmogorov PSD normalisation constant (matches
# synthetic/generate_turbulence.py's von_karman_psd, kept consistent here)
PSD_CONST = 0.0229


# ═══════════════════════════════════════════════════════════════════════════════
# Output data structure
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TurbulenceResult:
    """Stage 3 output — turbulence parameter estimates."""

    # r0 estimates
    r0_zernike_m:        float                # r0 from Zernike variance fit
    r0_slope_sf_m:        float               # r0 from spatial slope structure fn
    r0_mean_m:            float               # simple mean of the two (headline value)

    # tau0 / wind estimates
    tau0_temporal_sf_ms:  float               # from temporal structure function
    tau0_autocorr_ms:     float               # from 1/e autocorrelation crossing
    v_wind_eff_ms:        float               # effective wind speed (from temporal SF)

    # Diagnostics / supporting arrays
    zernike_variances:    Optional[np.ndarray]   # [J] measured <a_j^2>  [rad^2]
    zernike_theory:       Optional[np.ndarray]   # [J] theoretical K_j (D/r0)^(5/3)
    zernike_modes_fit:    Optional[np.ndarray]   # Noll j indices used in the r0 fit

    slope_sf_r:           np.ndarray          # [n_bins] separations [m]
    slope_sf_phi:         np.ndarray          # [n_bins] D_phi(r) estimate [rad^2]
    slope_sf_fit_line:     np.ndarray          # [n_bins] fitted 6.88(r/r0)^5/3

    temporal_sf_tau:      np.ndarray          # [n_lags] time lags [ms]
    temporal_sf_d:         np.ndarray          # [n_lags] D_phi(tau) [rad^2]
    temporal_sf_fit_line:  np.ndarray          # [n_lags] fitted curve

    autocorr_tau:          np.ndarray          # [n_lags] time lags [ms]
    autocorr_rho:           np.ndarray          # [n_lags] normalised autocorrelation

    signal_used_for_temporal: str              # which channel drove tau0 estimates

    def print_summary(self):
        print(
            f"\nTurbulence Characterization Summary\n"
            f"  r0 (Zernike variance)      : {self.r0_zernike_m*1e2:.3f} cm\n"
            f"  r0 (slope structure fn)    : {self.r0_slope_sf_m*1e2:.3f} cm\n"
            f"  r0 (mean)                  : {self.r0_mean_m*1e2:.3f} cm\n"
            f"  tau0 (temporal SF)         : {self.tau0_temporal_sf_ms:.3f} ms\n"
            f"  tau0 (1/e autocorrelation) : {self.tau0_autocorr_ms:.3f} ms\n"
            f"  v_wind (effective)          : {self.v_wind_eff_ms*1e3:.3f} mm/s\n"
            f"  Temporal signal used        : {self.signal_used_for_temporal}\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Zernike / Noll covariance (analytic, Kolmogorov statistics)
# ═══════════════════════════════════════════════════════════════════════════════

def _noll_to_nm(j: int) -> Tuple[int, int]:
    """Same convention as wavefront_reconstructor.ZernikeLib.noll_to_nm."""
    n = int(math.ceil((-3 + math.sqrt(9 + 8 * (j - 1))) / 2))
    j_n_start = n * (n + 1) // 2 + 1
    delta = j - j_n_start
    if n % 2 == 0:
        ms = list(range(0, n + 1, 2))
    else:
        ms = list(range(1, n + 1, 2))
    m_abs = ms[delta // 2]
    if delta % 2 == 0:
        m = m_abs if j % 2 == 0 else -m_abs
    else:
        m = -m_abs if j % 2 == 0 else m_abs
    return n, m


def noll_diagonal_variance_coefficient(j: int) -> float:
    """
    Theoretical single-mode Zernike variance coefficient K_j such that

        <a_j^2> = K_j * (D / r0)^(5/3)     [rad^2]

    under Kolmogorov statistics (Noll 1976, Eq. 25, specialised to the
    diagonal n=n', m=m' case). Piston (j=1) is undefined/excluded.
    """
    if j <= 1:
        raise ValueError("Piston (j=1) has no defined Kolmogorov variance")

    n, m = _noll_to_nm(j)
    num = gamma((2 * n - 5 / 3) / 2)
    den = (gamma((23 / 3) / 2) ** 2) * gamma((2 * n + 23 / 3) / 2)
    # sign = (-1)^((n+n-2m)/2) = (-1)^(n-m) which is always +1 for diagonal
    # terms since (n - m) is always even by construction of Noll indexing.
    const = PSD_CONST * math.pi ** (8 / 3)
    K = const * (n + 1) * num / den
    return float(K)


# ═══════════════════════════════════════════════════════════════════════════════
# r0 — Method A: Zernike coefficient variance fit
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_r0_zernike_variance(coeffs_all: np.ndarray,
                                  aperture_diameter: float,
                                  j_min: int = 4,
                                  j_max: Optional[int] = None,
                                  ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit r0 from the measured variance of each Zernike mode against the
    theoretical Noll variance coefficients.

    Modes j=2,3 (tip/tilt) are excluded by default (j_min=4): tip/tilt
    variance is frequently corrupted by mechanical vibration, tracking
    drift, or finite-aperture truncation effects unrelated to
    atmospheric statistics, so excluding them gives a more robust fit
    (standard practice, e.g. Hardy 1998 Ch. 5).

    Parameters
    ----------
    coeffs_all : float [N_frames, J]   Zernike coefficients (j=2..J+1)
    aperture_diameter : float [m]
    j_min, j_max : Noll index range used in the fit (inclusive)

    Returns
    -------
    r0_m        : float  Fried parameter estimate [m]
    variances   : float [J]  measured <a_j^2> for ALL available modes
    theory      : float [J]  theoretical K_j (D/r0)^(5/3) for ALL modes
    modes_used  : int   [n_fit]  Noll indices actually used in the fit
    """
    J = coeffs_all.shape[1]
    variances = np.var(coeffs_all, axis=0).astype(np.float64)   # [J], j=2..J+1

    j_max = j_max or (J + 1)
    noll_js = np.arange(2, J + 2)   # j index for each column

    K = np.array([noll_diagonal_variance_coefficient(j) for j in noll_js])

    fit_mask = (noll_js >= j_min) & (noll_js <= j_max)
    if fit_mask.sum() < 2:
        log.warning("Too few modes in fit range [%d, %d] — widening to all modes",
                    j_min, j_max)
        fit_mask = np.ones_like(noll_js, dtype=bool)

    ratios = variances[fit_mask] / K[fit_mask]   # each ≈ (D/r0)^(5/3)
    ratios = ratios[ratios > 0]

    if len(ratios) == 0:
        log.warning("No positive variance ratios — r0 (Zernike) estimate invalid")
        return float("nan"), variances, K * np.nan, noll_js[fit_mask]

    D_over_r0 = np.median(ratios) ** (3 / 5)
    r0 = aperture_diameter / D_over_r0

    theory_full = K * (aperture_diameter / r0) ** (5 / 3)

    log.info("r0 (Zernike variance fit): %.4f m  using modes j=%s",
             r0, list(noll_js[fit_mask]))

    return float(r0), variances, theory_full, noll_js[fit_mask]


# ═══════════════════════════════════════════════════════════════════════════════
# r0 — Method B: Spatial slope structure function
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_r0_slope_structure_function(s_x: np.ndarray,
                                          s_y: np.ndarray,
                                          valid: np.ndarray,
                                          sa_x_idx: np.ndarray,
                                          sa_y_idx: np.ndarray,
                                          d_sa: float,
                                          wavelength: float,
                                          n_bins: int = 12,
                                          ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate r0 from the spatial structure function of the slope field,
    averaged over all frames.

    Slopes are tilt angles [rad] = (wavelength/2pi) * d(phase)/dx, so
    the phase difference between two sub-apertures separated by r is
    approximated as:

        delta_phi(r) ≈ (s(x) - s(x+r)) * (2*pi / wavelength) * d_sa

    and D_phi(r) = <delta_phi(r)^2> is fit against the Kolmogorov form
    6.88 (r/r0)^(5/3).

    Parameters
    ----------
    s_x, s_y   : float [N_frames, N_sa_valid]  (radians, tilt angle)
    valid      : bool  [N_frames, N_sa_valid]
    sa_x_idx, sa_y_idx : int [N_sa_valid]  grid column/row index of each
                 valid sub-aperture (used to compute physical separation)
    d_sa       : float [m]  physical sub-aperture pitch
    wavelength : float [m]
    n_bins     : number of log-spaced separation bins

    Returns
    -------
    r0_m      : float
    r_centers : float [n_bins]   bin-centre separations [m]
    D_phi     : float [n_bins]   measured structure function [rad^2]
    fit_line  : float [n_bins]   fitted 6.88 (r/r0)^(5/3)
    """
    conv = (2.0 * np.pi / wavelength) * d_sa   # slope[rad] -> phase-diff[rad]

    x_m = sa_x_idx.astype(np.float64) * d_sa
    y_m = sa_y_idx.astype(np.float64) * d_sa
    n_sa = len(x_m)

    # Pairwise separations (static across frames)
    dx = x_m[:, None] - x_m[None, :]
    dy = y_m[:, None] - y_m[None, :]
    sep = np.hypot(dx, dy)
    iu, ju = np.triu_indices(n_sa, k=1)
    sep_pairs = sep[iu, ju]

    valid_frac = valid.mean(axis=0)   # [N_sa]  per-SA validity
    pair_ok = (valid_frac[iu] > 0.5) & (valid_frac[ju] > 0.5)

    sep_pairs = sep_pairs[pair_ok]
    iu_ok, ju_ok = iu[pair_ok], ju[pair_ok]

    # Average squared phase-difference over frames for each pair
    sx_diff2 = np.mean((s_x[:, iu_ok] - s_x[:, ju_ok]) ** 2, axis=0)
    sy_diff2 = np.mean((s_y[:, iu_ok] - s_y[:, ju_ok]) ** 2, axis=0)
    D_phi_pairs = 0.5 * (sx_diff2 + sy_diff2) * (conv ** 2)

    # Bin by separation (log-spaced), skip r=0
    r_min = max(d_sa * 0.9, sep_pairs[sep_pairs > 0].min())
    r_max = sep_pairs.max()
    bin_edges = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)

    r_centers, D_phi = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (sep_pairs >= lo) & (sep_pairs < hi)
        if m.sum() < 3:
            continue
        r_centers.append(float(sep_pairs[m].mean()))
        D_phi.append(float(D_phi_pairs[m].mean()))

    r_centers = np.array(r_centers)
    D_phi = np.array(D_phi)

    # Fit log(D_phi) = log(6.88) + 5/3 log(r) - 5/3 log(r0)
    good = D_phi > 0
    log_r = np.log(r_centers[good])
    log_d = np.log(D_phi[good])
    A = np.column_stack([np.ones_like(log_r), log_r])
    coef, *_ = np.linalg.lstsq(A, log_d, rcond=None)
    intercept, slope = coef

    r0 = float(np.exp((np.log(KOLMOGOROV_CONST) - intercept) / (5 / 3)))
    fit_line = KOLMOGOROV_CONST * (r_centers / r0) ** (5 / 3)

    log.info("r0 (slope structure fn): %.4f m  (fitted power-law slope=%.3f, "
             "expected 1.667)", r0, slope)

    return r0, r_centers, D_phi, fit_line


# ═══════════════════════════════════════════════════════════════════════════════
# r0 — Method B (fixed): Spatial phase structure function from Stage 2 maps
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_r0_phase_structure_function(
        phase_maps:        np.ndarray,
        pupil_mask:        np.ndarray,
        aperture_diameter: float,
        n_bins:            int = 12,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate r0 from the spatial structure function of the reconstructed
    wavefront phase maps (Stage 2 output).

    D_phi(r) = <[phi(x) - phi(x+r)]^2>   fitted to   6.88 * (r/r0)^(5/3)

    This is physically correct: the phase difference between two pupil
    points r apart directly obeys the Kolmogorov structure function.
    In contrast, the slope-difference approach computes a second-order
    phase quantity (curvature-like) whose dependence on r is NOT r^(5/3),
    leading to wildly biased r0 estimates.

    Parameters
    ----------
    phase_maps        : float [N_frames, G, G]  reconstructed wavefront [rad]
    pupil_mask        : bool  [G, G]
    aperture_diameter : float [m]  used to convert pixel lag → metres
    n_bins            : number of log-spaced separation bins

    Returns
    -------
    r0_m      : float
    r_centers : float [n_bins]  separations [m]
    D_phi     : float [n_bins]  measured structure function [rad^2]
    fit_line  : float [n_bins]  fitted 6.88 (r/r0)^(5/3)
    """
    N, G, _ = phase_maps.shape
    pixel_m  = aperture_diameter / G      # physical size of one grid pixel [m]
    max_lag  = G // 2

    r_list: List[float] = []
    D_list: List[float] = []

    for lag in range(1, max_lag + 1):
        r_m = lag * pixel_m

        # Horizontal pixel pairs: separation = lag pixels in x
        diff_h = phase_maps[:, :, lag:] - phase_maps[:, :, :-lag]  # [N, G, G-lag]
        mask_h = pupil_mask[:, lag:] & pupil_mask[:, :-lag]        # [G, G-lag]
        n_h = int(mask_h.sum())
        if n_h >= 3:
            r_list.append(r_m)
            D_list.append(float(np.mean(diff_h[:, mask_h] ** 2)))

        # Vertical pixel pairs: separation = lag pixels in y
        diff_v = phase_maps[:, lag:, :] - phase_maps[:, :-lag, :]  # [N, G-lag, G]
        mask_v = pupil_mask[lag:, :] & pupil_mask[:-lag, :]        # [G-lag, G]
        n_v = int(mask_v.sum())
        if n_v >= 3:
            r_list.append(r_m)
            D_list.append(float(np.mean(diff_v[:, mask_v] ** 2)))

    r_arr = np.array(r_list)
    D_arr = np.array(D_list)

    # Average horizontal and vertical values at identical lags
    unique_r = np.unique(r_arr)
    r_avg = np.array([float(r) for r in unique_r])
    D_avg = np.array([float(D_arr[r_arr == r].mean()) for r in unique_r])

    # Log-spaced binning
    r_min, r_max = r_avg.min(), r_avg.max()
    bin_edges   = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    r_centers, D_binned = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (r_avg >= lo) & (r_avg <= hi)
        if m.sum() == 0:
            continue
        r_centers.append(float(r_avg[m].mean()))
        D_binned.append(float(D_avg[m].mean()))

    r_centers = np.array(r_centers)
    D_binned  = np.array(D_binned)

    # Log-log fit: log(D) = log(6.88) + (5/3)*(log(r) - log(r0))
    good = D_binned > 0
    if good.sum() < 2:
        log.warning("Phase SF: too few valid bins — returning NaN for r0")
        return float("nan"), r_centers, D_binned, np.full_like(r_centers, np.nan)

    log_r = np.log(r_centers[good])
    log_d = np.log(D_binned[good])
    A     = np.column_stack([np.ones_like(log_r), log_r])
    coef, *_ = np.linalg.lstsq(A, log_d, rcond=None)
    intercept, slope_fit = coef

    r0       = float(np.exp((np.log(KOLMOGOROV_CONST) - intercept) / (5 / 3)))
    fit_line = KOLMOGOROV_CONST * (r_centers / r0) ** (5 / 3)

    log.info(
        "r0 (phase structure fn): %.4f m  D_phi range %.3f–%.3f rad²  "
        "fitted slope=%.3f (expected 1.667)",
        r0, D_binned.min(), D_binned.max(), slope_fit,
    )
    return r0, r_centers, D_binned, fit_line



# ═══════════════════════════════════════════════════════════════════════════════
# tau0 — Method A: temporal structure function (+ wind speed)
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_tau0_temporal_structure_function(signal: np.ndarray,
                                               dt_ms: float,
                                               r0_m: float,
                                               n_lags: int = 25,
                                               ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the temporal structure function D_phi(tau) = 6.88 (v*tau/r0)^(5/3)
    to a scalar phase-like time series (e.g. a single Zernike mode or a
    mean-slope proxy), extracting effective wind speed and tau0.

    Parameters
    ----------
    signal : float [N_frames]   e.g. tip-mode coefficient [rad]
    dt_ms  : float   nominal frame interval [ms]
    r0_m   : float   Fried parameter estimate [m] (from spatial method)
    n_lags : number of log-spaced lags to evaluate

    Returns
    -------
    tau0_ms   : float
    v_eff_ms  : float  [m/s]
    lags_ms   : float [n_lags]
    D_tau     : float [n_lags]
    fit_line  : float [n_lags]
    """
    n = len(signal)
    max_lag = max(2, n // 4)
    lag_candidates = np.unique(
        np.round(np.logspace(0, np.log10(max_lag), n_lags)).astype(int)
    )
    lag_candidates = lag_candidates[lag_candidates >= 1]

    lags_ms, D_tau = [], []
    for lag in lag_candidates:
        diff = signal[lag:] - signal[:-lag]
        if len(diff) < 5:
            continue
        lags_ms.append(lag * dt_ms)
        D_tau.append(float(np.mean(diff ** 2)))

    lags_ms = np.array(lags_ms)
    D_tau = np.array(D_tau)

    good = D_tau > 0
    log_t = np.log(lags_ms[good])
    log_d = np.log(D_tau[good])
    A = np.column_stack([np.ones_like(log_t), log_t])
    coef, *_ = np.linalg.lstsq(A, log_d, rcond=None)
    intercept, slope = coef

    # D_tau = coefficient * tau^(5/3),  coefficient = 6.88 (v/r0)^(5/3)
    coefficient = float(np.exp(intercept)) / (1e-3) ** (5 / 3)  # tau in seconds now
    # NOTE: lags_ms are in ms; convert fit to SI (seconds) for v_eff
    v_over_r0 = (coefficient / KOLMOGOROV_CONST) ** (3 / 5)
    v_eff = v_over_r0 * r0_m                       # [m/s]
    tau0_s = 0.314 * r0_m / v_eff if v_eff > 0 else float("inf")
    tau0_ms = tau0_s * 1e3

    fit_line = KOLMOGOROV_CONST * (v_over_r0 * lags_ms * 1e-3) ** (5 / 3)

    log.info("tau0 (temporal SF): %.4f ms   v_eff=%.4f mm/s  (fit slope=%.3f)",
             tau0_ms, v_eff * 1e3, slope)

    return tau0_ms, v_eff, lags_ms, D_tau, fit_line


# ═══════════════════════════════════════════════════════════════════════════════
# tau0 — Method B: 1/e autocorrelation crossing
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_tau0_autocorrelation(signal: np.ndarray,
                                   dt_ms: float,
                                   max_lag_frac: float = 0.5,
                                   ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    tau0 = time lag at which the normalised autocorrelation of `signal`
    first drops below 1/e.

    Returns
    -------
    tau0_ms  : float
    lags_ms  : float [n_lags]
    rho      : float [n_lags]   normalised autocorrelation
    """
    n = len(signal)
    s = signal - signal.mean()
    max_lag = max(2, int(n * max_lag_frac))

    var0 = np.mean(s ** 2)
    if var0 <= 0:
        return float("nan"), np.array([]), np.array([])

    lags = np.arange(0, max_lag)
    rho = np.array([
        np.mean(s[:n - lag] * s[lag:]) / var0 for lag in lags
    ])
    lags_ms = lags * dt_ms

    below = np.where(rho < (1.0 / math.e))[0]
    if len(below) == 0:
        tau0_ms = float(lags_ms[-1])   # never crosses within window
        log.warning("Autocorrelation never drops below 1/e within window — "
                     "tau0 (autocorr) is a LOWER BOUND")
    else:
        k = below[0]
        if k == 0:
            tau0_ms = 0.0
        else:
            # linear interpolation between k-1 and k
            r0_, r1_ = rho[k - 1], rho[k]
            t0_, t1_ = lags_ms[k - 1], lags_ms[k]
            frac = (r0_ - 1 / math.e) / (r0_ - r1_ + 1e-12)
            tau0_ms = float(t0_ + frac * (t1_ - t0_))

    log.info("tau0 (1/e autocorrelation): %.4f ms", tau0_ms)
    return tau0_ms, lags_ms, rho


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class TurbulenceCharacterizer:
    """
    Runs all Stage 3 estimators and packages results into a
    TurbulenceResult.
    """

    def __init__(self,
                 aperture_diameter: float,
                 lenslet_size:      float,
                 wavelength:        float,
                 frame_interval_ms: float):
        self.D          = aperture_diameter
        self.d_sa        = lenslet_size
        self.wavelength  = wavelength
        self.dt_ms       = frame_interval_ms

    def run(self,
           s1: dict,
           s2: Optional[dict],
           grid,
           ) -> TurbulenceResult:
        """
        Parameters
        ----------
        s1 : dict   Stage 1 results (s_x, s_y, slope_valid, timestamps_ms)
        s2 : dict | None   Stage 2 results (phase_maps, zernike_coeffs, pupil_mask)
        grid : SubApertureGrid   (from centroiding.build_grid)
        """
        s_x   = s1["s_x"]
        s_y   = s1["s_y"]
        valid = s1["slope_valid"].astype(bool)

        valid_idx = grid.valid_indices
        sa_x_idx  = (valid_idx % grid.n_sa_x)
        sa_y_idx  = (valid_idx // grid.n_sa_x)

        # ── r0, Method B: Phase structure function (preferred) ──────────────
        # Using D_phi(r) = <[phi(x) - phi(x+r)]^2> from Stage 2 phase maps
        # is physically correct and avoids the slope-difference formula which
        # computes a curvature-like quantity (not the phase SF).
        r0_sf = float("nan")
        r_c   = np.array([])
        D_phi = np.array([])
        sf_fit = np.array([])
        spatial_method = "none"

        if (s2 is not None
                and "phase_maps" in s2
                and "pupil_mask" in s2):
            pm    = s2["phase_maps"].astype(np.float32)
            pmask = s2["pupil_mask"].astype(bool)
            r0_sf, r_c, D_phi, sf_fit = estimate_r0_phase_structure_function(
                pm, pmask, self.D)
            spatial_method = "phase SF (Stage 2)"
        else:
            # Fallback: legacy slope-difference approach (biased, kept for
            # completeness when Stage 2 phase maps are not available).
            log.warning(
                "Stage 2 phase maps not available — falling back to slope-"
                "difference structure function (known to be biased).")
            r0_sf, r_c, D_phi, sf_fit = estimate_r0_slope_structure_function(
                s_x, s_y, valid, sa_x_idx, sa_y_idx,
                d_sa=self.d_sa, wavelength=self.wavelength,
            )
            spatial_method = "slope SF (legacy fallback)"

        log.info("Spatial r0 method: %s  r0_sf=%.4f m", spatial_method, r0_sf)

        # ── r0, Method A: Zernike coefficient variance ──────────────────────
        zern_var = zern_theory = zern_modes = None
        r0_zernike = float("nan")

        if s2 is not None and "zernike_coeffs" in s2 and s2["zernike_coeffs"].ndim == 2:
            coeffs = s2["zernike_coeffs"]
            r0_zernike, zern_var, zern_theory, zern_modes = \
                estimate_r0_zernike_variance(coeffs, self.D)

        # ── r0 consensus ────────────────────────────────────────────────────
        r0_mean = (np.nanmean([r0_zernike, r0_sf])
                  if not np.isnan(r0_zernike) else r0_sf)

        # ── Temporal signal for tau0 / v_wind ─────────────────────────────
        # Best signal: phase at a single central pupil pixel.
        # D_pixel(tau) = <[phi(x0,t+tau) - phi(x0,t)]^2>
        #              = 6.88 * (v*tau/r0)^(5/3)   for frozen-flow Kolmogorov
        # This is EXACT — unlike the Zernike tip coefficient whose temporal SF
        # has a different (biased) coefficient.
        signal_used = "unknown"
        temporal_signal = None

        if s2 is not None and "phase_maps" in s2 and "pupil_mask" in s2:
            pm    = s2["phase_maps"].astype(np.float64)
            pmask = s2["pupil_mask"].astype(bool)
            G     = pm.shape[1]
            # Find the pupil pixel closest to the geometric centre
            yx_valid = np.column_stack(np.where(pmask))
            centre   = np.array([G / 2.0, G / 2.0])
            dists    = np.linalg.norm(yx_valid.astype(float) - centre, axis=1)
            best_yx  = yx_valid[np.argmin(dists)]
            temporal_signal = pm[:, best_yx[0], best_yx[1]]
            signal_used = (
                f"central pupil pixel phase (Stage 2, "
                f"y={best_yx[0]}, x={best_yx[1]})"
            )
            log.info("Temporal signal: %s", signal_used)

        elif s2 is not None and "zernike_coeffs" in s2 and s2["zernike_coeffs"].ndim == 2:
            # Secondary fallback: Zernike tip mode
            temporal_signal = s2["zernike_coeffs"][:, 0].astype(np.float64)
            signal_used = "Zernike tip mode j=2 (fallback)"
            log.warning(
                "Phase maps not in s2 — using Zernike tip for temporal SF "
                "(v_wind estimate will be biased).")
        else:
            # Last resort: mean x-slope across valid SAs
            temporal_signal = np.array([
                s_x[i, valid[i]].mean() if valid[i].any() else 0.0
                for i in range(s_x.shape[0])
            ], dtype=np.float64)
            signal_used = "mean x-slope (last-resort fallback)"
            log.warning(
                "Neither phase maps nor Zernike coeffs available — "
                "using mean slope for temporal SF.")

        # ── tau0 / v_wind ─────────────────────────────────────────────────
        # Use the best spatial r0 for the temporal SF (r0 cancels in tau0
        # when derived from coefficient = 6.88*(v/r0)^(5/3), so tau0 is
        # robust even if r0_sf is somewhat biased).
        r0_for_tau = r0_sf if not np.isnan(r0_sf) else \
                     (r0_zernike if not np.isnan(r0_zernike) else 1e-2)

        tau0_sf, v_eff, lag_t, D_tau, temporal_fit = \
            estimate_tau0_temporal_structure_function(
                temporal_signal, self.dt_ms, r0_for_tau)

        tau0_ac, ac_lags, ac_rho = estimate_tau0_autocorrelation(
            temporal_signal, self.dt_ms)

        result = TurbulenceResult(
            r0_zernike_m             = r0_zernike,
            r0_slope_sf_m            = r0_sf,
            r0_mean_m                = r0_mean,
            tau0_temporal_sf_ms      = tau0_sf,
            tau0_autocorr_ms         = tau0_ac,
            v_wind_eff_ms            = v_eff,
            zernike_variances        = zern_var,
            zernike_theory           = zern_theory,
            zernike_modes_fit        = zern_modes,
            slope_sf_r               = r_c,
            slope_sf_phi             = D_phi,
            slope_sf_fit_line        = sf_fit,
            temporal_sf_tau          = lag_t,
            temporal_sf_d            = D_tau,
            temporal_sf_fit_line     = temporal_fit,
            autocorr_tau             = ac_lags,
            autocorr_rho             = ac_rho,
            signal_used_for_temporal = signal_used,
        )
        return result