#!/usr/bin/env python3
"""
run_stage1.py — SH-WFS Pipeline: Stage 1 End-to-End Runner
============================================================
Orchestrates the full Stage 1 pipeline:

  1. Synthetic turbulence data generation (Kolmogorov, Taylor frozen flow)
  2. Frame loading
  3. Centroiding (C backend with Python fallback)
  4. Algorithm benchmark (CoG vs T-CoG vs W-CoG)
  5. Slope computation
  6. All diagnostic visualisations
  7. Results saved to outputs/

Run:
    cd sh_wfs_pipeline
    python run_stage1.py

Optional flags:
    --config     path to sensor_config.yaml
    --no-generate  skip frame generation (use existing outputs/frames/)
    --c-backend  force C centroiding backend
    --n-frames   override number of frames to process

────────────────────────────────────────────────────────────────────────
FIXES vs the original version (see accompanying explanation):

1. `step3_benchmark` previously measured "accuracy" as RMS distance from
   the *geometric* sub-aperture centre. Since turbulence genuinely moves
   spots away from that centre, a more accurate algorithm reported a
   *larger* number — inverting the ranking. It now compares against the
   truth-derived expected centroid position (reference + the spot
   displacement implied by sx_truth/sy_truth via the same optical
   formula used by the synthetic generator), which is the actual
   ground truth.

2. `_plot_truth_comparison` previously converted sx_truth/sy_truth
   (rad/m, i.e. d(phase)/dx) into the same units as the measured slope
   using `sx_truth_2d * lenslet_size`. That is dimensionally wrong: the
   correct optical conversion (matching what render_shwfs_frame() and
   SlopeComputer jointly implement) is `sx_truth_2d * wavelength/(2*pi)`.
   lenslet_size and wavelength/(2*pi) differ by ~3000x for this sensor,
   which is exactly the scale mismatch seen in the truth-comparison plot.
   The centroiding + slope pipeline itself was already correct — only
   this validation/plotting step had the wrong constant.
────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "synthetic"))

from frame_loader   import NumpyFrameLoader
from centroiding    import (build_grid, PythonCentroider, CCentroider,
                            CentroidPipeline)
from slope_computer import SlopeComputer
import visualizer   as vis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_params(config: dict) -> dict:
    """Extract commonly used parameters into a flat dict."""
    cam = config["camera"]
    mla = config["mla"]
    pup = config["pupil"]
    cen = config["centroiding"]
    tel = config["telescope"]
    return dict(
        # Camera
        pixel_size      = cam["pixel_size"],
        frame_interval  = cam["frame_interval"],
        bit_depth       = cam["bit_depth"],
        # MLA
        n_sa_x          = mla["n_sa_x"],
        n_sa_y          = mla["n_sa_y"],
        pix_per_sa      = mla["pixels_per_sa"],
        focal_length    = mla["focal_length"],
        lenslet_size    = mla["lenslet_size"],
        # Telescope / optics
        wavelength      = tel["wavelength"],
        aperture_diameter = tel["aperture_diameter"],
        # Pupil
        pupil_cx        = pup["center_x"],
        pupil_cy        = pup["center_y"],
        pupil_radius    = pup["radius_px"],
        # Centroiding
        cent_method     = cen["algorithm"],
        thresh_sigma    = cen["threshold_sigma"],
        window_sigma    = cen["window_sigma_px"],
        min_flux        = cen["min_flux_threshold"],
        border_px       = cen["border_px"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 Steps
# ═══════════════════════════════════════════════════════════════════════════════

def step1_generate(config: dict) -> dict:
    """Generate synthetic turbulence frames and return truth data."""
    from generate_turbulence import TurbulenceSequenceGenerator
    log.info("━" * 60)
    log.info("STEP 1 — Generating synthetic SH-WFS frames")
    log.info("━" * 60)

    gen    = TurbulenceSequenceGenerator(config)
    result = gen.generate()

    log.info("✓ Generated %d frames  (r0=%.4f m, r0_fit=%.4f m)",
             result["meta"]["n_frames"],
             result["r0"],
             result["sf_info"]["r0_fit"])
    return result


def step2_build_grid(params: dict):
    """Build the sub-aperture grid from MLA/pupil parameters."""
    log.info("━" * 60)
    log.info("STEP 2 — Building sub-aperture grid")
    log.info("━" * 60)

    grid = build_grid(
        n_sa_x        = params["n_sa_x"],
        n_sa_y        = params["n_sa_y"],
        pix_per_sa    = params["pix_per_sa"],
        pupil_cx      = params["pupil_cx"],
        pupil_cy      = params["pupil_cy"],
        pupil_radius  = params["pupil_radius"],
    )
    log.info("✓ Grid: %d×%d SA  |  %d valid (in pupil)  |  %d masked",
             grid.n_sa_x, grid.n_sa_y, grid.n_sa_valid,
             grid.n_sa_total - grid.n_sa_valid)
    return grid


def step3_benchmark(frames: np.ndarray,
                    grid,
                    params: dict,
                    out_dir: Path,
                    truth: dict) -> dict:
    """
    Benchmark all three centroiding algorithms on a subset of frames.
    Returns timing and accuracy measurements.

    Accuracy is measured against the TRUTH-DERIVED expected centroid
    position (geometric reference + the pixel displacement implied by
    the synthetic sx_truth/sy_truth slopes), NOT the bare geometric
    reference — turbulence genuinely displaces spots away from the
    geometric centre, so comparing against the geometric centre alone
    penalises accurate algorithms.
    """
    log.info("━" * 60)
    log.info("STEP 3 — Centroiding algorithm benchmark")
    log.info("━" * 60)

    n_bench = min(20, len(frames))
    bench_frames = frames[:n_bench]

    algorithms = {
        "CoG":          "cog",
        "T-CoG (C)":    "threshold_cog",
        "W-CoG (C)":    "windowed_cog",
    }

    common_kwargs = dict(
        threshold_sigma = params["thresh_sigma"],
        window_sigma_px = params["window_sigma"],
        border_px       = params["border_px"],
        min_flux        = params["min_flux"],
    )

    bench_results = {}
    all_centroids = {}

    # Reference: geometric centres
    ref_cx = grid.cx_ref
    ref_cy = grid.cy_ref

    # ── Truth-derived expected centroid positions for the bench frames ──────
    # Same optical conversion used by generate_turbulence.render_shwfs_frame():
    #   dpix = f * (wavelength / 2*pi) * sx_truth[rad/m] / pixel_size
    have_truth = "sx_truth" in truth and "sy_truth" in truth
    if have_truth:
        conv = (params["focal_length"] * (params["wavelength"] / (2.0 * np.pi))
                / params["pixel_size"])
        n_sa_y, n_sa_x = params["n_sa_y"], params["n_sa_x"]
        sx_truth_bench = np.asarray(truth["sx_truth"])[:n_bench].reshape(n_bench, -1)
        sy_truth_bench = np.asarray(truth["sy_truth"])[:n_bench].reshape(n_bench, -1)
        expected_cx = ref_cx[np.newaxis, :] + conv * sx_truth_bench
        expected_cy = ref_cy[np.newaxis, :] + conv * sy_truth_bench
    else:
        log.warning("No truth data available — falling back to geometric "
                    "reference for the benchmark (accuracy numbers will be "
                    "biased, see docstring).")
        expected_cx = np.broadcast_to(ref_cx, (n_bench, ref_cx.shape[0]))
        expected_cy = np.broadcast_to(ref_cy, (n_bench, ref_cy.shape[0]))

    for name, method in algorithms.items():
        # Try C backend, fall back to Python
        try:
            centroider = CCentroider(grid=grid, method=method, **common_kwargs)
        except Exception:
            centroider = PythonCentroider(grid=grid, method=method, **common_kwargs)

        # Time it
        t0 = time.perf_counter()
        cx_all, cy_all = [], []
        for f in bench_frames:
            cx, cy, flux, valid = centroider.process_frame(f)
            cx_all.append(cx)
            cy_all.append(cy)
        t1 = time.perf_counter()

        cx_arr = np.array(cx_all)   # [n_bench, N_sa]
        cy_arr = np.array(cy_all)

        elapsed_ms  = (t1 - t0) * 1e3
        per_frame   = elapsed_ms / n_bench

        # Accuracy: RMS displacement from the TRUTH-derived expected
        # centroid, for valid sub-apertures only.
        v = grid.valid
        disp = np.hypot(cx_arr[:, v] - expected_cx[:, v],
                        cy_arr[:, v] - expected_cy[:, v])
        rms_disp = float(np.sqrt(np.mean(disp**2)))

        bench_results[name] = {
            "time_ms":  per_frame,
            "error_px": rms_disp,
        }
        all_centroids[name] = (cx_arr, cy_arr)

        log.info("  %-15s: %.2f ms/frame  |  RMS error vs truth = %.3f px",
                 name, per_frame, rms_disp)

    # Plot benchmark
    vis.plot_centroid_benchmark(
        bench_results,
        save_path=str(out_dir / "benchmark_algorithms.png")
    )
    log.info("✓ Benchmark plot saved")

    return bench_results, all_centroids


def step4_centroid(frames: np.ndarray,
                   grid,
                   params: dict,
                   frame_interval_ms: float,
                   use_c: bool = True) -> dict:
    """Run full centroiding pipeline over all frames."""
    log.info("━" * 60)
    log.info("STEP 4 — Full centroiding pipeline")
    log.info("━" * 60)

    loader = NumpyFrameLoader(frames, frame_interval_ms=frame_interval_ms)

    pipeline = CentroidPipeline(
        grid            = grid,
        method          = params["cent_method"],
        threshold_sigma = params["thresh_sigma"],
        window_sigma_px = params["window_sigma"],
        border_px       = params["border_px"],
        min_flux        = params["min_flux"],
        use_c           = use_c,
    )
    pipeline.set_reference("geometric")

    t0 = time.perf_counter()
    result = pipeline.run(loader, progress=True)
    t1 = time.perf_counter()

    n = frames.shape[0]
    log.info("✓ Centroided %d frames in %.2f s  (%.1f ms/frame)",
             n, t1-t0, (t1-t0)/n*1e3)

    valid_frac = result["valid"].mean()
    log.info("  Valid centroid fraction: %.1f%%", valid_frac * 100)

    return result


def step5_slopes(centroid_result: dict, params: dict) -> object:
    """Convert centroid displacements to angular slopes."""
    log.info("━" * 60)
    log.info("STEP 5 — Slope computation")
    log.info("━" * 60)

    grid = centroid_result["grid"]
    sc   = SlopeComputer(
        pixel_size   = params["pixel_size"],
        focal_length = params["focal_length"],
        lenslet_size = params["lenslet_size"],
        valid_mask   = grid.valid,
        clip_sigma   = 4.0,
    )
    slope_result = sc.compute(centroid_result)
    sc.print_summary(slope_result)

    return slope_result, sc


def step6_visualise(frames:          np.ndarray,
                    grid,
                    centroid_result: dict,
                    slope_result,
                    sc:              SlopeComputer,
                    truth:           dict,
                    params:          dict,
                    out_dir:         Path):
    """Generate all diagnostic visualisations."""
    log.info("━" * 60)
    log.info("STEP 6 — Visualisation")
    log.info("━" * 60)

    # Pick a representative frame (middle of sequence)
    n      = frames.shape[0]
    fi     = n // 2
    frame  = frames[fi]

    cx     = centroid_result["cx"][fi]
    cy     = centroid_result["cy"][fi]
    ref_cx = centroid_result["ref_cx"]
    ref_cy = centroid_result["ref_cy"]
    valid  = centroid_result["valid"][fi]

    sr  = slope_result
    # Reshape slopes back to 2D for this frame
    sx_frame = sr.s_x[fi]
    sy_frame = sr.s_y[fi]
    v_frame  = sr.valid[fi]

    sx_2d, sy_2d = sc.slope_map_2d_from_grid(sx_frame, sy_frame, v_frame, grid)

    # ── 1. Raw frame + grid overlay ─────────────────────────────────────────
    vis.plot_raw_frame(
        frame, grid, cx, cy, valid,
        title=f"SH-WFS Frame {fi}  (r0={truth['r0']:.3f} m)",
        save_path=str(out_dir / "frame_with_grid.png")
    )
    log.info("  ✓ frame_with_grid.png")

    # ── 2. Centroid displacement map ─────────────────────────────────────────
    vis.plot_centroid_map(
        ref_cx, ref_cy, cx, cy, valid, grid,
        title=f"Centroid Displacements — Frame {fi}",
        save_path=str(out_dir / "centroid_map.png")
    )
    log.info("  ✓ centroid_map.png")

    # ── 3. Slope quiver ──────────────────────────────────────────────────────
    vis.plot_slope_quiver(
        sx_2d, sy_2d,
        title=f"Wavefront Slopes — Frame {fi}",
        scale=3.0,
        save_path=str(out_dir / "slope_quiver.png")
    )
    log.info("  ✓ slope_quiver.png")

    # ── 4. Temporal timeseries ───────────────────────────────────────────────
    vis.plot_slope_timeseries(
        sr,
        save_path=str(out_dir / "slope_timeseries.png")
    )
    log.info("  ✓ slope_timeseries.png")

    # ── 5. Full 4-panel Stage 1 summary ─────────────────────────────────────
    vis.plot_stage1_summary(
        frame, grid, cx, cy, ref_cx, ref_cy, valid,
        sx_2d, sy_2d,
        frame_idx=fi,
        save_path=str(out_dir / "stage1_summary.png")
    )
    log.info("  ✓ stage1_summary.png")

    # ── 6. Truth comparison if available ────────────────────────────────────
    if "sx_truth" in truth:
        _plot_truth_comparison(truth, sr, sc, grid, fi, out_dir,
                               wavelength=params["wavelength"])

    log.info("✓ All plots saved to %s", out_dir)


def _plot_truth_comparison(truth: dict,
                            slope_result,
                            sc: SlopeComputer,
                            grid,
                            frame_idx: int,
                            out_dir: Path,
                            wavelength: float):
    """Compare measured slopes against synthetic ground truth."""
    import matplotlib.pyplot as plt

    sx_truth_2d = truth["sx_truth"][frame_idx]   # [n_sa_y, n_sa_x]  rad/m
    sy_truth_2d = truth["sy_truth"][frame_idx]

    sr = slope_result
    sx_frame = sr.s_x[frame_idx]
    sy_frame = sr.s_y[frame_idx]
    v_frame  = sr.valid[frame_idx]

    sx_meas_2d, sy_meas_2d = sc.slope_map_2d_from_grid(
        sx_frame, sy_frame, v_frame, grid
    )

    # ── FIX ──────────────────────────────────────────────────────────────
    # sx_truth/sy_truth are d(phase)/dx in [rad/m]. SlopeComputer's measured
    # slope is the physical wavefront TILT ANGLE in radians:
    #     theta = (wavelength / 2*pi) * d(phase)/dx
    # This is the same conversion used by generate_turbulence.py's
    # render_shwfs_frame() to produce the pixel displacement in the first
    # place, so it is the correct, self-consistent factor to invert.
    # The previous version used `sx_truth_2d * sc.lenslet_size`, which is
    # off by ~3 orders of magnitude (lenslet_size=300um vs wavelength/2pi
    # ~0.1um) — that was the entire cause of the huge truth/measured
    # mismatch.
    tilt_conv = wavelength / (2.0 * np.pi)
    sx_truth_rad = sx_truth_2d * tilt_conv
    sy_truth_rad = sy_truth_2d * tilt_conv
    # ─────────────────────────────────────────────────────────────────────

    mask = grid.valid.reshape(grid.n_sa_y, grid.n_sa_x) & ~np.isnan(sx_meas_2d)

    # Scatter: measured vs truth for x-slope
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("#0d0d1a")

    for ax, s_m, s_t, label in zip(
        axes,
        [sx_meas_2d, sy_meas_2d],
        [sx_truth_rad, sy_truth_rad],
        ["sx", "sy"]
    ):
        sm_flat = s_m[mask] * 1e3
        st_flat = s_t[mask] * 1e3
        ax.scatter(st_flat, sm_flat, s=15, alpha=0.7, c="#00b0ff", edgecolors="none")
        if len(st_flat) == 0 or len(sm_flat) == 0:
            ax.text(0.3, 0.5, "No valid centroids for this frame",
                    transform=ax.transAxes, color="orange")
            continue
        lim = max(np.abs(st_flat).max(), np.abs(sm_flat).max()) * 1.1
        if lim == 0: lim = 1.0
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=0.8, label="Ideal")
        ax.set_xlabel(f"Truth {label} [mrad]", color="white")
        ax.set_ylabel(f"Measured {label} [mrad]", color="white")
        ax.set_title(f"{label}: measured vs truth  (frame {frame_idx})", color="white")
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")

        # Residual stats
        res = sm_flat - st_flat
        ax.text(0.05, 0.92, f"RMS residual: {res.std():.2f} mrad",
                transform=ax.transAxes, color="yellow", fontsize=9)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_dir / "truth_comparison.png"),
                dpi=150, bbox_inches="tight")
    log.info("  ✓ truth_comparison.png")
    plt.close(fig)


def step7_save_results(centroid_result: dict,
                       slope_result,
                       out_dir: Path):
    """Save centroid and slope arrays for downstream stages."""
    log.info("━" * 60)
    log.info("STEP 7 — Saving results")
    log.info("━" * 60)

    sr = slope_result
    save_path = out_dir / "stage1_results.npz"

    np.savez_compressed(
        save_path,
        # Centroids
        cx            = centroid_result["cx"],
        cy            = centroid_result["cy"],
        flux          = centroid_result["flux"],
        valid         = centroid_result["valid"].astype(np.uint8),
        ref_cx        = centroid_result["ref_cx"],
        ref_cy        = centroid_result["ref_cy"],
        dx_pix        = centroid_result["dx"],
        dy_pix        = centroid_result["dy"],
        timestamps_ms = centroid_result["timestamps_ms"],
        # Slopes
        s_x           = sr.s_x,
        s_y           = sr.s_y,
        slopes        = sr.slopes,
        slope_valid   = sr.valid.astype(np.uint8),
        pix_to_rad    = np.array([sr.pix_to_rad]),
    )
    log.info("✓ Stage 1 results → %s  (%.1f MB)",
             save_path, save_path.stat().st_size / 1e6)

    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SH-WFS Pipeline — Stage 1")
    parser.add_argument("--config",       default="config/sensor_config.yaml")
    parser.add_argument("--no-generate",  action="store_true",
                        help="Skip frame generation, use existing files")
    parser.add_argument("--c-backend",    action="store_true", default=True)
    parser.add_argument("--n-frames",     type=int, default=None)
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    config   = load_config(args.config)
    if args.n_frames:
        config["synthetic"]["n_frames"] = args.n_frames

    params   = resolve_params(config)
    out_dir  = ROOT / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("╔" + "═" * 58 + "╗")
    log.info("║  SH-WFS Pipeline — Stage 1: Centroiding & Slopes        ║")
    log.info("╚" + "═" * 58 + "╝")
    log.info("Config: %s", args.config)
    log.info("Output: %s", out_dir)

    # ── Step 1: Generate synthetic data ─────────────────────────────────────
    truth = {}
    if not args.no_generate:
        truth = step1_generate(config)
        frames = truth["frames"]
    else:
        # Load pre-existing truth.npz
        truth_path = ROOT / "outputs" / "truth.npz"
        if truth_path.exists():
            log.info("Loading existing truth data from %s", truth_path)
            data   = np.load(truth_path)
            frames = data["frames"]
            truth  = {k: data[k] for k in data.files}
            truth["r0"] = float(data["r0"])
        else:
            log.error("No truth.npz found. Run without --no-generate first.")
            sys.exit(1)

    n_frames = frames.shape[0]
    log.info("Working with %d frames, shape %s", n_frames, frames.shape)

    # Keep frames in raw ADU float32 for centroiding
    # (thresholds in sensor_config.yaml are in ADU units)
    frames_f = frames.astype(np.float32)

    # ── Step 2: Sub-aperture grid ────────────────────────────────────────────
    grid = step2_build_grid(params)

    # ── Step 3: Algorithm benchmark ──────────────────────────────────────────
    bench_results, _ = step3_benchmark(
        frames_f, grid, params, out_dir, truth
    )

    # ── Step 4: Full centroiding ─────────────────────────────────────────────
    frame_interval_ms = config["camera"]["frame_interval"] * 1e3
    centroid_result   = step4_centroid(
        frames_f, grid, params,
        frame_interval_ms=frame_interval_ms,
        use_c=args.c_backend,
    )

    # ── Step 5: Slopes ───────────────────────────────────────────────────────
    slope_result, sc = step5_slopes(centroid_result, params)

    # ── Step 6: Visualisation ────────────────────────────────────────────────
    step6_visualise(
        frames_f, grid,
        centroid_result, slope_result, sc,
        truth, params, out_dir
    )

    # ── Step 7: Save ─────────────────────────────────────────────────────────
    save_path = step7_save_results(centroid_result, slope_result, out_dir)

    # ── Final summary ────────────────────────────────────────────────────────
    sr = slope_result
    print()
    print("╔" + "═" * 58 + "╗")
    print("║              STAGE 1 COMPLETE                            ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Frames processed    : {n_frames:<33d}║")
    print(f"║  Valid sub-apertures : {grid.n_sa_valid}/{str(grid.n_sa_total):<31}║")
    print(f"║  RMS slope (mean)    : {sr.rms_slope.mean()*1e3:.4f} mrad{'':<23s}║")
    print(f"║  pix → rad factor    : {sr.pix_to_rad:.4e}{'':<23s}║")
    print(f"║  r0 (input)          : {truth.get('r0', 'N/A')!s:<33}║")
    print(f"║  Results saved to    : {str(save_path.name):<33}║")
    print("╚" + "═" * 58 + "╝")
    print(f"\nNext: run Stage 2 (wavefront reconstruction) using:")
    print(f"      python run_stage2.py")


if __name__ == "__main__":
    main()
