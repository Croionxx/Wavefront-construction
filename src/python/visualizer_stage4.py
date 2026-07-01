"""
visualizer_stage4.py — Stage 4 DM Control Visualisations
=========================================================
plot_actuator_map           : 2D actuator command heatmap + bar
plot_influence_functions    : sample IF grid
plot_correction_comparison  : before/after/residual phase side-by-side
plot_strehl_improvement     : time series comparison uncorrected vs corrected
plot_wfe_improvement        : WFE RMS time series
plot_stroke_statistics      : per-actuator stroke box-plots over time
plot_stage4_summary         : 4-panel summary figure
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import CenteredNorm

log = logging.getLogger(__name__)

DARK_BG  = "#0d0d1a"
PANEL_BG = "#1a1a2e"
CMAP_WAVE  = "RdBu_r"
CMAP_PHASE = "RdBu_r"

COLOR_UNCORR = "#ff6d00"
COLOR_CORR   = "#76ff03"
COLOR_ACT    = "#00b0ff"


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


def plot_actuator_map(dm_result,
                      frame_idx: int = 0,
                      save_path: Optional[str] = None,
                      ) -> Tuple[plt.Figure, np.ndarray]:
    """
    2D heatmap of actuator commands for a single frame, plus bar chart
    showing command distribution across all actuators.
    """
    u_2d = dm_result.actuator_commands[frame_idx]          # [n_y, n_x]
    s_2d = dm_result.stroke_um[frame_idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(DARK_BG)

    # Left: actuator map
    ax = axes[0]
    _dark(ax)
    vmax = float(np.abs(u_2d).max()) or 1.0
    im = ax.imshow(u_2d, cmap=CMAP_WAVE, origin="upper",
                   vmin=-vmax, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("Command [rad]", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    ax.set_title(f"Actuator Commands — Frame {frame_idx}", color="white")
    ax.set_xlabel("Actuator col")
    ax.set_ylabel("Actuator row")

    # Annotate each cell
    ny, nx = u_2d.shape
    for r in range(ny):
        for c in range(nx):
            ax.text(c, r, f"{u_2d[r,c]:.1f}", ha="center", va="center",
                    fontsize=5, color="white", alpha=0.8)

    # Right: stroke histogram
    ax2 = axes[1]
    _dark(ax2)
    s_flat = s_2d.ravel()
    bins = np.linspace(-float(np.abs(s_flat).max()) * 1.1 or -1,
                        float(np.abs(s_flat).max()) * 1.1 or 1, 25)
    ax2.hist(s_flat, bins=bins, color=COLOR_ACT, edgecolor="none", alpha=0.8)
    ax2.axvline(0, color="white", lw=0.6, ls="--")
    ax2.set_xlabel("Stroke [µm]", color="white")
    ax2.set_ylabel("Actuator count", color="white")
    ax2.set_title(f"Stroke Distribution — Frame {frame_idx}", color="white")
    ax2.text(0.98, 0.95, f"max |u| = {np.abs(s_flat).max():.2f} µm",
             transform=ax2.transAxes, ha="right", va="top",
             color="yellow", fontsize=9)

    fig.suptitle("DM Actuator Map", color="white", fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes


def plot_correction_comparison(dm_result,
                                wf_result,
                                frame_idx: int = 0,
                                save_path: Optional[str] = None,
                                ) -> plt.Figure:
    """
    Three-panel: uncorrected phase | corrected phase | residual (corr - 0).
    """
    mask   = dm_result.pupil_mask
    W_in   = np.where(mask, wf_result.phase_maps[frame_idx],  np.nan)
    W_corr = np.where(mask, dm_result.corrected_phase[frame_idx], np.nan)
    W_res  = W_corr - np.nanmean(W_corr[mask])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)

    panels = [
        (W_in,   "Uncorrected",
         f"WFE = {dm_result.uncorrected_wfe_rms[frame_idx]*1e3:.1f} mrad  "
         f"Strehl = {dm_result.uncorrected_strehl[frame_idx]:.4f}"),
        (W_corr, "DM-Corrected",
         f"WFE = {dm_result.corrected_wfe_rms[frame_idx]*1e3:.1f} mrad  "
         f"Strehl = {dm_result.corrected_strehl[frame_idx]:.4f}"),
        (W_res,  "Corrected (piston-free residual)", ""),
    ]

    for ax, (data, title, subtitle) in zip(axes, panels):
        _dark(ax)
        vmax = float(np.nanpercentile(np.abs(data), 98)) or 1.0
        im = ax.imshow(data, cmap=CMAP_PHASE, origin="upper",
                       vmin=-vmax, vmax=vmax)
        cb = plt.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("Phase [rad]", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        ax.set_title(f"{title}\n{subtitle}", color="white", fontsize=9)
        ax.tick_params(colors="white")

    fig.suptitle(f"Wavefront Correction — Frame {frame_idx}",
                 color="white", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_strehl_improvement(dm_result,
                             save_path: Optional[str] = None,
                             ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Two-panel: Strehl ratio time series and WFE RMS time series,
    overlaid before/after correction.
    """
    t  = dm_result.timestamps_ms

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    # Top: Strehl
    ax = axes[0]
    _dark(ax)
    ax.plot(t, dm_result.uncorrected_strehl, color=COLOR_UNCORR,
            lw=0.8, label="Uncorrected", alpha=0.9)
    ax.plot(t, dm_result.corrected_strehl,   color=COLOR_CORR,
            lw=0.8, label="DM-Corrected")
    ax.axhline(0.8, color="white", lw=0.5, ls="--", alpha=0.5, label="Strehl=0.8")
    ax.set_ylabel("Strehl ratio (Maréchal)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Wavefront Quality — Before and After DM Correction", color="white")
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
    ax.grid(True, alpha=0.2)

    # Bottom: WFE RMS
    ax2 = axes[1]
    _dark(ax2)
    ax2.plot(t, dm_result.uncorrected_wfe_rms * 1e3, color=COLOR_UNCORR,
             lw=0.8, label="Uncorrected")
    ax2.plot(t, dm_result.corrected_wfe_rms   * 1e3, color=COLOR_CORR,
             lw=0.8, label="DM-Corrected")
    ax2.set_ylabel("WFE RMS [mrad]")
    ax2.set_xlabel("Time [ms]")
    ax2.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
    ax2.grid(True, alpha=0.2)

    # Improvement annotation
    imp = dm_result.wfe_improvement_factor
    ax2.text(0.02, 0.92, f"Mean improvement: {imp:.1f}×",
             transform=ax2.transAxes, color="yellow", fontsize=9)

    for ax in axes:
        ax.tick_params(colors="white")

    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes


def plot_stroke_statistics(dm_result,
                            save_path: Optional[str] = None,
                            ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Per-actuator stroke statistics over the full sequence.
    Left: RMS stroke per actuator (2D heatmap).
    Right: Time-series of max absolute stroke across all actuators.
    """
    n_frames = dm_result.n_frames
    strokes  = dm_result.stroke_um   # [N, n_y, n_x]

    rms_per_act = np.sqrt(np.mean(strokes ** 2, axis=0))  # [n_y, n_x]
    max_abs     = np.abs(strokes).max(axis=(1, 2))         # [N]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(DARK_BG)

    ax = axes[0]
    _dark(ax)
    im = ax.imshow(rms_per_act, cmap="plasma", origin="upper",
                   aspect="auto", interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("RMS stroke [µm]", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    ax.set_title("Per-actuator RMS stroke", color="white")
    ax.set_xlabel("Col")
    ax.set_ylabel("Row")

    ax2 = axes[1]
    _dark(ax2)
    ax2.plot(dm_result.timestamps_ms, max_abs, color="#e040fb", lw=0.8)
    max_stroke_um = dm_result.stroke_um.__class__  # get value from dm_result
    ax2.set_ylabel("Max |stroke| [µm]")
    ax2.set_xlabel("Time [ms]")
    ax2.set_title("Max Actuator Stroke per Frame", color="white")
    ax2.grid(True, alpha=0.2)

    fig.suptitle("DM Stroke Statistics", color="white", fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes


def plot_stage4_summary(dm_result,
                         wf_result,
                         frame_idx: int = 0,
                         save_path: Optional[str] = None,
                         ) -> plt.Figure:
    """
    4-panel Stage 4 summary:
      A) Actuator command map (single frame)
      B) Before/after phase comparison (single frame)
      C) Strehl time series
      D) WFE RMS time series
    """
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor(DARK_BG)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    t = dm_result.timestamps_ms

    # ── A: Actuator command map ───────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _dark(ax_a)
    u_2d = dm_result.actuator_commands[frame_idx]
    vmax = float(np.abs(u_2d).max()) or 1.0
    im_a = ax_a.imshow(u_2d, cmap=CMAP_WAVE, origin="upper",
                       vmin=-vmax, vmax=vmax, aspect="auto")
    cb = plt.colorbar(im_a, ax=ax_a, fraction=0.046)
    cb.set_label("u [rad]", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    ax_a.set_title(f"Actuator Commands — Frame {frame_idx}", color="white")
    ax_a.set_xlabel("Actuator col")
    ax_a.set_ylabel("Actuator row")

    # ── B: Uncorrected vs corrected phase (side-by-side in one axis) ─────
    ax_b = fig.add_subplot(gs[0, 1])
    _dark(ax_b)
    mask   = dm_result.pupil_mask
    W_in   = np.where(mask, wf_result.phase_maps[frame_idx],      np.nan)
    W_corr = np.where(mask, dm_result.corrected_phase[frame_idx],  np.nan)
    G      = mask.shape[0]
    # Stitch them side by side with a NaN gap
    gap    = np.full((G, 4), np.nan)
    mosaic = np.hstack([W_in, gap, W_corr])
    vmax_b = float(np.nanpercentile(np.abs(W_in), 97)) or 1.0
    im_b   = ax_b.imshow(mosaic, cmap=CMAP_PHASE, origin="upper",
                          vmin=-vmax_b, vmax=vmax_b)
    cb2 = plt.colorbar(im_b, ax=ax_b, fraction=0.046)
    cb2.set_label("Phase [rad]", color="white")
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")
    ax_b.set_title(
        f"Uncorrected  |  Corrected  (frame {frame_idx})\n"
        f"WFE: {dm_result.uncorrected_wfe_rms[frame_idx]*1e3:.1f} → "
        f"{dm_result.corrected_wfe_rms[frame_idx]*1e3:.1f} mrad",
        color="white", fontsize=8)

    # ── C: Strehl time series ─────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    _dark(ax_c)
    ax_c.plot(t, dm_result.uncorrected_strehl, color=COLOR_UNCORR,
              lw=0.8, label="Uncorrected")
    ax_c.plot(t, dm_result.corrected_strehl,   color=COLOR_CORR,
              lw=0.8, label="Corrected")
    ax_c.axvline(t[frame_idx], color="red", lw=0.8, ls=":")
    ax_c.set_ylabel("Strehl ratio")
    ax_c.set_xlabel("Time [ms]")
    ax_c.set_title("Strehl Ratio", color="white")
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
    ax_c.grid(True, alpha=0.2)

    # ── D: WFE RMS ───────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    _dark(ax_d)
    ax_d.plot(t, dm_result.uncorrected_wfe_rms * 1e3, color=COLOR_UNCORR,
              lw=0.8, label="Uncorrected")
    ax_d.plot(t, dm_result.corrected_wfe_rms   * 1e3, color=COLOR_CORR,
              lw=0.8, label="Corrected")
    ax_d.axvline(t[frame_idx], color="red", lw=0.8, ls=":")
    ax_d.set_ylabel("WFE RMS [mrad]")
    ax_d.set_xlabel("Time [ms]")
    ax_d.set_title("Wavefront Error RMS", color="white")
    ax_d.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
    ax_d.grid(True, alpha=0.2)

    imp = dm_result.wfe_improvement_factor
    fig.suptitle(
        f"Stage 4 — DM Correction  "
        f"(WFE {imp:.1f}×,  "
        f"Strehl {dm_result.uncorrected_strehl.mean():.3f}→"
        f"{dm_result.corrected_strehl.mean():.3f})",
        color="white", fontsize=13, y=1.01
    )
    _save(fig, save_path)
    return fig