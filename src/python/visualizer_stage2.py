"""
visualizer_stage2.py — Stage 2 Wavefront Reconstruction Visualisations
=======================================================================
All publication-quality plots for Stage 2 output.  Kept separate from
visualizer.py so Stage 1 visualisation is not disturbed.

Functions
---------
plot_phase_map            : single-frame 2D phase map
plot_zernike_spectrum     : bar chart of Zernike coefficients + cumulative power
plot_zernike_temporal     : time series of selected Zernike modes
plot_strehl_timeseries    : Strehl ratio and WFE RMS over time
plot_reconstruction_summary : 4-panel summary for one frame
plot_truth_phase_comparison : measured vs synthetic phase map (if truth available)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import CenteredNorm

log = logging.getLogger(__name__)

# ── Colour palette (matches Stage 1 visualiser) ─────────────────────────────
CMAP_WAVE  = "RdBu_r"
CMAP_SPOT  = "inferno"
DARK_BG    = "#0d0d1a"
PANEL_BG   = "#1a1a2e"


def _save(fig, path: Optional[str | Path], dpi: int = 150):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        log.info("Saved: %s", path)


# ── Individual plot functions ────────────────────────────────────────────────

def plot_phase_map(phase_map:   np.ndarray,
                   pupil_mask:  np.ndarray,
                   title:       str = "Reconstructed Wavefront",
                   wavelength_nm: float = 633.0,
                   save_path:   Optional[str] = None,
                   ) -> Tuple[plt.Figure, plt.Axes]:
    """
    2D false-colour phase map with colour bar in both radians and nm.

    Parameters
    ----------
    phase_map  : float32 [H, W]  wavefront in radians
    pupil_mask : bool   [H, W]   True inside pupil
    """
    display = np.where(pupil_mask, phase_map, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    vmax = float(np.nanpercentile(np.abs(display), 98))
    im   = ax.imshow(display, cmap=CMAP_WAVE, origin="upper",
                     vmin=-vmax, vmax=vmax)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Phase [rad]", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # Second axis in nm
    nm_scale = wavelength_nm / (2 * np.pi)
    cb2 = ax.inset_axes([1.18, 0, 0.04, 1])
    im2 = cb2.imshow(np.linspace(-vmax * nm_scale, vmax * nm_scale, 256
                                  ).reshape(-1, 1),
                     cmap=CMAP_WAVE, aspect="auto")
    cb2.yaxis.set_label_position("right")
    cb2.yaxis.tick_right()
    cb2.set_ylabel("Phase [nm]", color="white", fontsize=8)
    cb2.tick_params(colors="white", labelsize=7)

    ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    ax.set_xlabel("Column", color="white")
    ax.set_ylabel("Row", color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_zernike_spectrum(coeffs:     np.ndarray,
                          title:      str = "Zernike Spectrum",
                          n_modes:    int = 21,
                          save_path:  Optional[str] = None,
                          ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Bar chart of Zernike coefficients (single frame or mean over frames).

    Parameters
    ----------
    coeffs : float [J] or [N_frames, J]
        If 2D, the mean ± std over frames is shown.
    """
    if coeffs.ndim == 2:
        mean_c = coeffs.mean(axis=0)
        std_c  = coeffs.std(axis=0)
    else:
        mean_c = coeffs
        std_c  = None

    J      = min(n_modes, len(mean_c))
    x      = np.arange(2, J + 2)          # Noll index (j=2..J+1)
    power2 = mean_c[:J]**2
    cum    = np.cumsum(power2) / (np.sum(power2) + 1e-30) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(DARK_BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

    colors = ["#00b0ff" if c >= 0 else "#ff6d00" for c in mean_c[:J]]
    bars   = ax1.bar(x, mean_c[:J], color=colors, edgecolor="none", width=0.7)

    if std_c is not None:
        ax1.errorbar(x, mean_c[:J], yerr=std_c[:J], fmt="none",
                     ecolor="white", capsize=3, linewidth=0.8, alpha=0.6)

    ax1.axhline(0, color="white", lw=0.5, ls="--")
    ax1.set_xlabel("Noll index j")
    ax1.set_ylabel("Coefficient [rad]")
    ax1.set_title(f"{title} — Coefficients")

    # Annotate the biggest mode
    dominant = int(np.argmax(np.abs(mean_c[:J])))
    ax1.annotate(f"j={dominant+2}",
                 xy=(dominant + 2, mean_c[dominant]),
                 xytext=(dominant + 2 + 0.5, mean_c[dominant] * 1.2),
                 color="yellow", fontsize=8,
                 arrowprops=dict(arrowstyle="-", color="yellow", lw=0.6))

    # Cumulative power
    ax2.plot(x, cum, color="#76ff03", lw=1.2, marker="o", markersize=4)
    ax2.axhline(90, color="yellow", lw=0.7, ls="--", label="90%")
    ax2.axhline(99, color="orange",  lw=0.7, ls="--", label="99%")
    ax2.set_xlabel("Noll index j")
    ax2.set_ylabel("Cumulative power [%]")
    ax2.set_title(f"{title} — Cumulative Power")
    ax2.set_ylim(0, 101)
    ax2.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")

    fig.tight_layout()
    _save(fig, save_path)
    return fig, (ax1, ax2)


def plot_zernike_temporal(coeffs_all:  np.ndarray,
                          timestamps_ms: np.ndarray,
                          modes:       List[int] = (2, 3, 4, 5, 6),
                          save_path:   Optional[str] = None,
                          ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Time series of selected Zernike modes.

    Parameters
    ----------
    coeffs_all : [N_frames, J]   Zernike coefficients
    modes      : Noll indices to plot (j=2=tip, j=3=tilt, j=4=defocus…)
    """
    palette = ["#00b0ff", "#ff6d00", "#76ff03", "#e040fb", "#ff4081",
               "#18ffff", "#ffea00"]
    n_modes = len(modes)
    t       = timestamps_ms

    fig, axes = plt.subplots(n_modes, 1, figsize=(13, 2.2 * n_modes),
                             sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    mode_names = {2: "Tip (j=2)", 3: "Tilt (j=3)", 4: "Defocus (j=4)",
                  5: "Astig-45 (j=5)", 6: "Astig-0 (j=6)",
                  7: "Coma-x (j=7)", 8: "Coma-y (j=8)"}

    for ax, j_noll, color in zip(axes, modes, palette):
        k = j_noll - 2   # 0-based index into coeffs_all
        if k < 0 or k >= coeffs_all.shape[1]:
            continue
        c = coeffs_all[:, k] * 1e3   # mrad
        ax.plot(t, c, color=color, lw=0.8)
        ax.axhline(0, color="white", lw=0.4, ls="--")
        label = mode_names.get(j_noll, f"j={j_noll}")
        ax.set_ylabel(f"{label}\n[mrad]", color="white", fontsize=8)
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="white")
        ax.text(0.98, 0.85, f"σ={c.std():.2f} mrad",
                transform=ax.transAxes, color="yellow",
                fontsize=7, ha="right")

    axes[-1].set_xlabel("Time [ms]", color="white")
    fig.suptitle("Zernike Mode Time Series", color="white", fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes


def plot_strehl_timeseries(wf_result,
                           save_path: Optional[str] = None,
                           ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Two-panel: (top) Strehl ratio per frame, (bottom) WFE RMS in nm.
    """
    t      = wf_result.timestamps_ms
    strehl = wf_result.strehl_estimate
    wfe    = wf_result.phase_rms   # [rad]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    ax1.plot(t, strehl, color="#76ff03", lw=0.8)
    ax1.set_ylabel("Strehl ratio (Maréchal)", color="white")
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(0.8, color="orange", lw=0.7, ls="--", label="Strehl=0.8 (diffraction limit)")
    ax1.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
    ax1.set_title("Wavefront Quality — Time Series", color="white")

    ax2.plot(t, wfe * 1e3, color="#00b0ff", lw=0.8)
    ax2.set_ylabel("WFE RMS [mrad]", color="white")
    ax2.set_xlabel("Time [ms]", color="white")

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save(fig, save_path)
    return fig, (ax1, ax2)


def plot_reconstruction_summary(phase_map:   np.ndarray,
                                pupil_mask:  np.ndarray,
                                coeffs:      Optional[np.ndarray],
                                wf_result,
                                frame_idx:   int = 0,
                                save_path:   Optional[str] = None,
                                ) -> plt.Figure:
    """
    4-panel Stage 2 summary figure:
      A) Phase map
      B) Zernike bar chart (modal only) or flat placeholder
      C) Strehl time series
      D) Residual RMS time series
    """
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor(DARK_BG)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    def _dark_ax(ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="white")
        for lbl in (ax.xaxis.label, ax.yaxis.label, ax.title):
            lbl.set_color("white")

    # ── A: Phase map ─────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _dark_ax(ax_a)
    display = np.where(pupil_mask, phase_map, np.nan)
    vmax    = float(np.nanpercentile(np.abs(display), 98)) or 1.0
    im_a    = ax_a.imshow(display, cmap=CMAP_WAVE, origin="upper",
                          vmin=-vmax, vmax=vmax)
    plt.colorbar(im_a, ax=ax_a, label="Phase [rad]", fraction=0.046)
    ax_a.set_title(f"Reconstructed Wavefront — Frame {frame_idx}")

    # ── B: Zernike spectrum (modal) or node variance heatmap (zonal) ────────
    ax_b = fig.add_subplot(gs[0, 1])
    _dark_ax(ax_b)

    if coeffs is not None:
        J    = len(coeffs)
        x    = np.arange(2, J + 2)
        cols = ["#00b0ff" if c >= 0 else "#ff6d00" for c in coeffs]
        ax_b.bar(x, coeffs, color=cols, edgecolor="none", width=0.7)
        ax_b.axhline(0, color="white", lw=0.5, ls="--")
        ax_b.set_xlabel("Noll index j")
        ax_b.set_ylabel("Coefficient [rad]")
        ax_b.set_title(f"Zernike Spectrum — Frame {frame_idx}")
    else:
        # Zonal: show node phase as heatmap
        im_b = ax_b.imshow(phase_map, cmap=CMAP_WAVE, origin="upper",
                           norm=CenteredNorm())
        plt.colorbar(im_b, ax=ax_b, label="Phase [rad]", fraction=0.046)
        ax_b.set_title("Zonal Phase (node grid)")

    # ── C: Strehl time series ─────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    _dark_ax(ax_c)
    t = wf_result.timestamps_ms
    ax_c.plot(t, wf_result.strehl_estimate, color="#76ff03", lw=0.8)
    ax_c.axhline(0.8, color="orange", lw=0.6, ls="--", label="0.8")
    ax_c.axvline(t[frame_idx], color="red", lw=0.8, ls=":")
    ax_c.set_ylabel("Strehl ratio")
    ax_c.set_xlabel("Time [ms]")
    ax_c.set_title("Strehl Ratio")
    ax_c.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
    ax_c.grid(True, alpha=0.25)

    # ── D: Slope residual ─────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    _dark_ax(ax_d)
    ax_d.plot(t, wf_result.residual_rms * 1e3, color="#ff6d00", lw=0.8)
    ax_d.axvline(t[frame_idx], color="red", lw=0.8, ls=":")
    ax_d.set_ylabel("Residual RMS [mrad]")
    ax_d.set_xlabel("Time [ms]")
    ax_d.set_title("Slope Residual RMS")
    ax_d.grid(True, alpha=0.25)

    fig.suptitle(f"Stage 2 — Wavefront Reconstruction  ({wf_result.method})",
                 color="white", fontsize=13, y=1.01)
    _save(fig, save_path)
    return fig


def plot_truth_phase_comparison(phase_meas:   np.ndarray,
                                phase_truth:  np.ndarray,
                                pupil_mask:   np.ndarray,
                                frame_idx:    int = 0,
                                save_path:    Optional[str] = None,
                                ) -> plt.Figure:
    """
    Side-by-side: measured phase | truth phase | residual.
    Used when synthetic ground truth is available.

    Parameters
    ----------
    phase_meas, phase_truth : float [H, W]  both in radians
    """
    residual = phase_meas - phase_truth

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)

    titles = ["Measured", "Truth", "Residual (meas − truth)"]
    maps   = [phase_meas, phase_truth, residual]
    labels = ["Phase [rad]", "Phase [rad]", "Residual [rad]"]

    for ax, data, title, label in zip(axes, maps, titles, labels):
        display = np.where(pupil_mask, data, np.nan)
        vmax    = float(np.nanpercentile(np.abs(display), 98)) or 1e-6
        im = ax.imshow(display, cmap=CMAP_WAVE, origin="upper",
                       vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label=label, fraction=0.046)
        ax.set_title(title, color="white")
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="white")

    # Add residual stats to the right-hand panel
    res_pupil = residual[pupil_mask]
    axes[2].text(0.05, 0.05,
                 f"RMS = {res_pupil.std()*1e3:.2f} mrad",
                 transform=axes[2].transAxes,
                 color="yellow", fontsize=9)

    fig.suptitle(f"Phase Reconstruction vs Ground Truth — Frame {frame_idx}",
                 color="white", fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return fig
