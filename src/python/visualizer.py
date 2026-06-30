"""
visualizer.py — SH-WFS Pipeline Visualisation Utilities
=========================================================
Publication-quality plots for all pipeline stages:

  Stage 1  : raw frame with sub-aperture grid overlay, centroid map
  Stage 1  : slope quiver plot and magnitude heatmap
  Stage 2  : wavefront phase map (placeholder API for Stage 2 output)
  Stats    : temporal evolution, RMS vs frame, structure function

All functions return (fig, axes) tuples and save to disk optionally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import CenteredNorm

log = logging.getLogger(__name__)

# ── Colour palette ───────────────────────────────────────────────────────────
CMAP_WAVE    = "RdBu_r"    # wavefront / slope magnitude
CMAP_SPOT    = "inferno"   # spot intensity
CMAP_SLOPE   = "coolwarm"  # signed slope map
ARROW_COLOR  = "#00e5ff"   # quiver arrows
VALID_COLOR  = "#00e676"   # valid SA outline
INVALID_COLOR= "#ff1744"   # masked SA outline

# ── Helper ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Optional[str | Path], dpi: int = 150):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        log.info("Saved plot: %s", path)


# ── Stage 1 plots ────────────────────────────────────────────────────────────

def plot_raw_frame(frame:       np.ndarray,
                   grid,
                   centroids_x: Optional[np.ndarray] = None,
                   centroids_y: Optional[np.ndarray] = None,
                   valid:       Optional[np.ndarray] = None,
                   title:       str  = "SH-WFS Frame",
                   save_path:   Optional[str] = None,
                   ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Show a raw detector frame with the sub-aperture grid overlaid
    and optional centroid markers.

    Parameters
    ----------
    frame      : float32 [H, W]
    grid       : SubApertureGrid
    centroids_x, centroids_y : float [N_sa_total] — full-frame px coords
    valid      : bool [N_sa_total]
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    im = ax.imshow(frame, cmap=CMAP_SPOT, origin="upper",
                   interpolation="nearest",
                   vmin=np.percentile(frame, 1),
                   vmax=np.percentile(frame, 99))
    plt.colorbar(im, ax=ax, label="Intensity (norm.)", fraction=0.046, pad=0.04)

    # Draw sub-aperture grid
    p = grid.pix_per_sa
    for idx in range(grid.n_sa_total):
        xs = grid.x_start[idx]
        ys = grid.y_start[idx]
        v  = grid.valid[idx] if valid is None else (
            grid.valid[idx] and valid[idx])
        color = VALID_COLOR if v else INVALID_COLOR
        rect  = mpatches.Rectangle(
            (xs - 0.5, ys - 0.5), p, p,
            linewidth=0.6, edgecolor=color, facecolor="none", alpha=0.7
        )
        ax.add_patch(rect)

    # Centroid markers
    if centroids_x is not None and centroids_y is not None:
        mask = grid.valid if valid is None else (grid.valid & valid)
        ax.scatter(
            centroids_x[mask], centroids_y[mask],
            s=12, c="yellow", marker="+", linewidths=0.8,
            zorder=5, label="Centroid"
        )
        ax.legend(loc="upper right", fontsize=7)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Column [px]")
    ax.set_ylabel("Row [px]")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_centroid_map(ref_cx: np.ndarray,
                      ref_cy: np.ndarray,
                      cx:     np.ndarray,
                      cy:     np.ndarray,
                      valid:  np.ndarray,
                      grid,
                      title:  str = "Centroid Displacements",
                      save_path: Optional[str] = None,
                      ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of centroid positions vs reference, coloured by displacement.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Reference positions
    ax.scatter(ref_cx[grid.valid], ref_cy[grid.valid],
               s=30, c="white", edgecolors="grey", linewidths=0.5,
               marker="o", zorder=2, label="Reference", alpha=0.6)

    # Measured centroids (valid only)
    v_idx = np.where(grid.valid & valid)[0]
    disp  = np.hypot(cx[v_idx] - ref_cx[v_idx],
                     cy[v_idx] - ref_cy[v_idx])
    sc = ax.scatter(cx[v_idx], cy[v_idx],
                    s=40, c=disp, cmap=CMAP_WAVE,
                    edgecolors="none", zorder=3, label="Measured")
    plt.colorbar(sc, ax=ax, label="|displacement| [px]", fraction=0.046)

    # Displacement arrows
    for i in v_idx:
        ax.annotate("",
            xy=(cx[i], cy[i]), xytext=(ref_cx[i], ref_cy[i]),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR,
                            lw=0.7, mutation_scale=6),
            zorder=4)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Column [px]")
    ax.set_ylabel("Row [px]")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_slope_quiver(sx_2d:     np.ndarray,
                      sy_2d:     np.ndarray,
                      title:     str = "Wavefront Slopes",
                      scale:     float = 1.0,
                      save_path: Optional[str] = None,
                      ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Two-panel: quiver plot of slope vectors + slope magnitude heatmap.

    Parameters
    ----------
    sx_2d, sy_2d : float [n_sa_y, n_sa_x]  NaN where masked
    scale        : scaling factor for quiver arrow lengths
    """
    ny, nx = sx_2d.shape
    xs = np.arange(nx)
    ys = np.arange(ny)
    X, Y = np.meshgrid(xs, ys)

    mag  = np.hypot(sx_2d, sy_2d)
    vmax = np.nanpercentile(mag, 98)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: quiver ────────────────────────────────────────────────────────
    ax = axes[0]
    bg = ax.imshow(mag, origin="upper", cmap=CMAP_WAVE,
                   vmin=0, vmax=vmax, aspect="auto")
    plt.colorbar(bg, ax=ax, label="|slope| [rad]")

    # Only draw arrows where not NaN
    mask = ~np.isnan(sx_2d)
    q = ax.quiver(
        X[mask], Y[mask],
        sx_2d[mask] * scale, -sy_2d[mask] * scale,  # flip y for image coords
        color=ARROW_COLOR, scale=None, scale_units="xy",
        angles="xy", width=0.004, headwidth=3, headlength=4,
        zorder=5
    )
    ax.quiverkey(q, 0.85, 1.03, float(np.nanstd(mag)),
                 f"{np.nanstd(mag):.2e} rad", labelpos="E", fontproperties={"size": 8})
    ax.set_title(f"{title} — Quiver")
    ax.set_xlabel("SA index x")
    ax.set_ylabel("SA index y")

    # ── Right: x and y slopes separately ────────────────────────────────────
    ax2 = axes[1]
    im2 = ax2.imshow(sx_2d, origin="upper", cmap=CMAP_SLOPE,
                     norm=CenteredNorm(), aspect="auto")
    plt.colorbar(im2, ax=ax2, label="sx [rad]")
    ax2.set_title(f"{title} — x-Slope")
    ax2.set_xlabel("SA index x")
    ax2.set_ylabel("SA index y")

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes


def plot_slope_timeseries(slope_result,
                          n_modes: int = 3,
                          save_path: Optional[str] = None,
                          ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot temporal evolution of slopes:
    - Top: RMS slope magnitude per frame
    - Bottom: tip/tilt (mean sx, sy) time series

    Parameters
    ----------
    slope_result : SlopeResult
    """
    sr = slope_result
    t  = sr.timestamps_ms

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # RMS slope
    axes[0].plot(t, sr.rms_slope * 1e3, color="#00b0ff", lw=0.8)
    axes[0].set_ylabel("RMS slope [mrad]")
    axes[0].set_title("Slope Time Series")
    axes[0].grid(True, alpha=0.3)

    # Mean x-slope (tip)
    axes[1].plot(t, sr.mean_x_slope * 1e3, color="#ff6d00", lw=0.8)
    axes[1].set_ylabel("Mean sx [mrad]  (Tip)")
    axes[1].axhline(0, color="white", lw=0.4, ls="--")
    axes[1].grid(True, alpha=0.3)

    # Mean y-slope (tilt)
    axes[2].plot(t, sr.mean_y_slope * 1e3, color="#76ff03", lw=0.8)
    axes[2].set_ylabel("Mean sy [mrad]  (Tilt)")
    axes[2].set_xlabel("Time [ms]")
    axes[2].axhline(0, color="white", lw=0.4, ls="--")
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes


def plot_centroid_benchmark(results: dict,
                            save_path: Optional[str] = None,
                            ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar chart comparing centroiding algorithms (speed and accuracy).

    Parameters
    ----------
    results : dict
        {algorithm_name: {"time_ms": float, "error_px": float}}
    """
    names  = list(results.keys())
    times  = [results[n]["time_ms"]  for n in names]
    errors = [results[n]["error_px"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = ["#00b0ff", "#ff6d00", "#76ff03"][:len(names)]
    ax1.bar(names, times, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Time per frame [ms]")
    ax1.set_title("Centroiding Speed")

    ax2.bar(names, errors, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("RMS centroid error [px]")
    ax2.set_title("Centroiding Accuracy")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    fig.patch.set_facecolor("#0f0f1a")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, (ax1, ax2)


def plot_stage1_summary(frame:        np.ndarray,
                        grid,
                        cx:           np.ndarray,
                        cy:           np.ndarray,
                        ref_cx:       np.ndarray,
                        ref_cy:       np.ndarray,
                        valid:        np.ndarray,
                        sx_2d:        np.ndarray,
                        sy_2d:        np.ndarray,
                        frame_idx:    int = 0,
                        save_path:    Optional[str] = None,
                        ) -> plt.Figure:
    """
    4-panel Stage 1 summary figure for a single frame.
    Top-left  : raw frame with grid + centroids
    Top-right : centroid displacement map
    Bottom-left : x-slope 2D heatmap
    Bottom-right: y-slope 2D heatmap
    """
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor("#0d0d1a")
    gs  = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel A: raw frame ──────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    vmin = np.percentile(frame, 1)
    vmax = np.percentile(frame, 99.5)
    im_a = ax_a.imshow(frame, cmap="inferno", origin="upper",
                       interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(im_a, ax=ax_a, label="Intensity", fraction=0.046)

    p = grid.pix_per_sa
    for i in range(grid.n_sa_total):
        c = VALID_COLOR if grid.valid[i] else INVALID_COLOR
        ax_a.add_patch(mpatches.Rectangle(
            (grid.x_start[i] - 0.5, grid.y_start[i] - 0.5),
            p, p, linewidth=0.5, edgecolor=c, facecolor="none", alpha=0.6))

    vmask = grid.valid & valid
    ax_a.scatter(cx[vmask], cy[vmask], s=8, c="yellow",
                 marker="+", linewidths=0.6, zorder=5)
    ax_a.set_title(f"Frame {frame_idx} — Raw + Centroids", color="white")
    ax_a.set_facecolor("#0d0d1a")
    ax_a.tick_params(colors="white")

    # ── Panel B: displacement arrows ────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    disp  = np.hypot(cx - ref_cx, cy - ref_cy)
    v_idx = np.where(vmask)[0]
    sc = ax_b.scatter(ref_cx[v_idx], ref_cy[v_idx],
                      c=disp[v_idx], cmap=CMAP_WAVE,
                      s=30, edgecolors="none")
    plt.colorbar(sc, ax=ax_b, label="|Δ| [px]", fraction=0.046)
    for i in v_idx:
        ax_b.annotate("",
            xy=(cx[i], cy[i]), xytext=(ref_cx[i], ref_cy[i]),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR,
                            lw=0.5, mutation_scale=5), zorder=3)
    ax_b.invert_yaxis()
    ax_b.set_aspect("equal")
    ax_b.set_title("Centroid Displacements", color="white")
    ax_b.set_facecolor("#0d0d1a")
    ax_b.tick_params(colors="white")

    # ── Panel C: x-slope ────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    lim  = float(np.nanpercentile(np.abs(sx_2d), 97))
    im_c = ax_c.imshow(sx_2d * 1e3, origin="upper", cmap=CMAP_SLOPE,
                       vmin=-lim*1e3, vmax=lim*1e3, aspect="auto")
    plt.colorbar(im_c, ax=ax_c, label="sx [mrad]", fraction=0.046)
    ax_c.set_title("x-Slope map  (Tip)", color="white")
    ax_c.set_facecolor("#0d0d1a"); ax_c.tick_params(colors="white")

    # ── Panel D: y-slope ────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    lim  = float(np.nanpercentile(np.abs(sy_2d), 97))
    im_d = ax_d.imshow(sy_2d * 1e3, origin="upper", cmap=CMAP_SLOPE,
                       vmin=-lim*1e3, vmax=lim*1e3, aspect="auto")
    plt.colorbar(im_d, ax=ax_d, label="sy [mrad]", fraction=0.046)
    ax_d.set_title("y-Slope map  (Tilt)", color="white")
    ax_d.set_facecolor("#0d0d1a"); ax_d.tick_params(colors="white")

    fig.suptitle("Stage 1 — Centroiding Summary", color="white",
                 fontsize=14, y=1.01)
    _save(fig, save_path)
    return fig
