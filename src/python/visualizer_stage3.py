"""
visualizer_stage3.py — Stage 3 Turbulence Characterization Visualisations
============================================================================
plot_zernike_variance_fit   : measured vs theoretical Zernike variance (log-log)
plot_slope_structure_fn     : spatial structure function + Kolmogorov fit
plot_temporal_structure_fn  : temporal structure function + fit
plot_autocorrelation        : autocorrelation + 1/e crossing
plot_stage3_summary          : 4-panel summary
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

DARK_BG  = "#0d0d1a"
PANEL_BG = "#1a1a2e"


def _save(fig, path, dpi=150):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        log.info("Saved: %s", path)


def _dark(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white")
    for lbl in (ax.xaxis.label, ax.yaxis.label, ax.title):
        lbl.set_color("white")
    ax.grid(True, alpha=0.25, color="#333355")


def plot_zernike_variance_fit(tr, save_path: Optional[str] = None):
    if tr.zernike_variances is None:
        log.info("No Zernike coefficients — skipping variance-fit plot")
        return None, None

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(DARK_BG)
    _dark(ax)

    j = np.arange(2, len(tr.zernike_variances) + 2)
    ax.loglog(j, tr.zernike_variances, "o", color="#00b0ff",
              label="Measured $\\langle a_j^2\\rangle$")
    ax.loglog(j, tr.zernike_theory, "-", color="#ff6d00", lw=1.5,
              label=f"Noll theory (r0={tr.r0_zernike_m*1e2:.2f} cm)")

    if tr.zernike_modes_fit is not None:
        ax.axvspan(tr.zernike_modes_fit.min() - 0.3,
                   tr.zernike_modes_fit.max() + 0.3,
                   color="yellow", alpha=0.08, label="Modes used in fit")

    ax.set_xlabel("Noll index j")
    ax.set_ylabel("Variance [rad$^2$]")
    ax.set_title("Zernike Variance vs Kolmogorov Theory")
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_slope_structure_fn(tr, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(DARK_BG)
    _dark(ax)

    ax.loglog(tr.slope_sf_r, tr.slope_sf_phi, "o", color="#76ff03",
              label="Measured $D_\\phi(r)$")
    ax.loglog(tr.slope_sf_r, tr.slope_sf_fit_line, "-", color="#ff6d00", lw=1.5,
              label=f"Kolmogorov fit (r0={tr.r0_slope_sf_m*1e2:.2f} cm)")

    ax.set_xlabel("Separation r [m]")
    ax.set_ylabel("$D_\\phi(r)$ [rad$^2$]")
    ax.set_title("Spatial Slope Structure Function")
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_temporal_structure_fn(tr, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(DARK_BG)
    _dark(ax)

    ax.loglog(tr.temporal_sf_tau, tr.temporal_sf_d, "o", color="#e040fb",
              label=f"Measured ({tr.signal_used_for_temporal})")
    ax.loglog(tr.temporal_sf_tau, tr.temporal_sf_fit_line, "-",
              color="#ff6d00", lw=1.5,
              label=(f"Fit: v_eff={tr.v_wind_eff_ms*1e3:.2f} mm/s, "
                     f"$\\tau_0$={tr.tau0_temporal_sf_ms:.2f} ms"))

    ax.set_xlabel("Time lag [ms]")
    ax.set_ylabel("$D_\\phi(\\tau)$ [rad$^2$]")
    ax.set_title("Temporal Structure Function")
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_autocorrelation(tr, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(DARK_BG)
    _dark(ax)

    ax.plot(tr.autocorr_tau, tr.autocorr_rho, color="#00b0ff", lw=1.2)
    ax.axhline(1 / np.e, color="orange", lw=0.8, ls="--", label="1/e")
    ax.axvline(tr.tau0_autocorr_ms, color="red", lw=0.8, ls=":",
              label=f"$\\tau_0$={tr.tau0_autocorr_ms:.2f} ms")
    ax.axhline(0, color="white", lw=0.4, ls="--")

    ax.set_xlabel("Time lag [ms]")
    ax.set_ylabel("Normalised autocorrelation")
    ax.set_title(f"Autocorrelation ({tr.signal_used_for_temporal})")
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_stage3_summary(tr, save_path: Optional[str] = None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor(DARK_BG)

    ax_a, ax_b, ax_c, ax_d = axes.flat
    for ax in axes.flat:
        _dark(ax)

    # A: Zernike variance
    if tr.zernike_variances is not None:
        j = np.arange(2, len(tr.zernike_variances) + 2)
        ax_a.loglog(j, tr.zernike_variances, "o", color="#00b0ff", ms=4)
        ax_a.loglog(j, tr.zernike_theory, "-", color="#ff6d00", lw=1.2)
        ax_a.set_title(f"Zernike variance — r0={tr.r0_zernike_m*1e2:.2f} cm")
        ax_a.set_xlabel("Noll j"); ax_a.set_ylabel("Var [rad$^2$]")
    else:
        ax_a.text(0.5, 0.5, "No Zernike coefficients\n(zonal reconstruction)",
                  ha="center", va="center", color="orange",
                  transform=ax_a.transAxes)

    # B: spatial SF
    ax_b.loglog(tr.slope_sf_r, tr.slope_sf_phi, "o", color="#76ff03", ms=4)
    ax_b.loglog(tr.slope_sf_r, tr.slope_sf_fit_line, "-", color="#ff6d00", lw=1.2)
    ax_b.set_title(f"Spatial SF — r0={tr.r0_slope_sf_m*1e2:.2f} cm")
    ax_b.set_xlabel("r [m]"); ax_b.set_ylabel("$D_\\phi$ [rad$^2$]")

    # C: temporal SF
    ax_c.loglog(tr.temporal_sf_tau, tr.temporal_sf_d, "o", color="#e040fb", ms=4)
    ax_c.loglog(tr.temporal_sf_tau, tr.temporal_sf_fit_line, "-",
               color="#ff6d00", lw=1.2)
    ax_c.set_title(f"Temporal SF — $\\tau_0$={tr.tau0_temporal_sf_ms:.2f} ms")
    ax_c.set_xlabel("lag [ms]"); ax_c.set_ylabel("$D_\\phi(\\tau)$ [rad$^2$]")

    # D: autocorrelation
    ax_d.plot(tr.autocorr_tau, tr.autocorr_rho, color="#00b0ff", lw=1.2)
    ax_d.axhline(1 / np.e, color="orange", lw=0.7, ls="--")
    ax_d.axvline(tr.tau0_autocorr_ms, color="red", lw=0.7, ls=":")
    ax_d.set_title(f"Autocorrelation — $\\tau_0$={tr.tau0_autocorr_ms:.2f} ms")
    ax_d.set_xlabel("lag [ms]"); ax_d.set_ylabel("$\\rho(\\tau)$")

    fig.suptitle(
        f"Stage 3 — Turbulence Characterization   "
        f"(r0_mean={tr.r0_mean_m*1e2:.2f} cm, v_wind={tr.v_wind_eff_ms*1e3:.2f} mm/s)",
        color="white", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig