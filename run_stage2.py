#!/usr/bin/env python3
"""
run_stage2.py — SH-WFS Pipeline: Stage 2 Wavefront Reconstruction
===================================================================
Reads the Stage 1 output (outputs/plots/stage1_results.npz + optionally
outputs/truth.npz) and produces a complete wavefront reconstruction.

Steps
-----
  1. Load Stage 1 results and sensor config
  2. Build sub-aperture grid (needed for coordinate conversion)
  3. Calibrate the reconstructor (build interaction / Fried matrices)
  4. Run reconstruction over all frames (modal and/or zonal)
  5. Validate against synthetic ground truth (if available)
  6. Produce all diagnostic plots
  7. Save Stage 2 results to outputs/plots/stage2_results.npz

Run
---
    python run_stage2.py
    python run_stage2.py --method modal_zernike --n-modes 21
    python run_stage2.py --method zonal_fried
    python run_stage2.py --method both          # run both and compare

Optional flags
--------------
    --config     path to sensor_config.yaml            [default: config/sensor_config.yaml]
    --stage1     path to stage1_results.npz            [default: outputs/plots/stage1_results.npz]
    --method     modal_zernike | zonal_fried | both    [default: from config]
    --n-modes    number of Zernike modes               [default: from config]
    --svd        SVD truncation threshold               [default: from config]
    --save-matrices  save reconstructor matrices to outputs/recon/
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "synthetic"))

from centroiding          import build_grid
from wavefront_reconstructor import (ModalReconstructor, ZonalReconstructor,
                                      WavefrontResult)
import visualizer_stage2 as vis2

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
    cam  = config["camera"]
    mla  = config["mla"]
    pup  = config["pupil"]
    tel  = config["telescope"]
    recon= config["reconstruction"]
    return dict(
        pixel_size      = cam["pixel_size"],
        n_sa_x          = mla["n_sa_x"],
        n_sa_y          = mla["n_sa_y"],
        pix_per_sa      = mla["pixels_per_sa"],
        focal_length    = mla["focal_length"],
        lenslet_size    = mla["lenslet_size"],
        pupil_cx        = pup["center_x"],
        pupil_cy        = pup["center_y"],
        pupil_radius    = pup["radius_px"],
        wavelength      = tel["wavelength"],
        aperture_diameter = tel["aperture_diameter"],
        n_zernike_modes = recon["n_zernike_modes"],
        svd_threshold   = float(recon["svd_threshold"]),
        recon_method    = recon["method"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_stage1(path: Path) -> dict:
    """Load Stage 1 NPZ and return dict of arrays."""
    log.info("Loading Stage 1 results from %s", path)
    data = np.load(str(path))
    out  = {k: data[k] for k in data.files}
    # Restore boolean valid array
    if "slope_valid" in out:
        out["slope_valid"] = out["slope_valid"].astype(bool)
    if "valid" in out:
        out["valid"] = out["valid"].astype(bool)
    log.info("  s_x shape:       %s", out["s_x"].shape)
    log.info("  s_y shape:       %s", out["s_y"].shape)
    log.info("  slope_valid:     %s", out["slope_valid"].shape)
    log.info("  timestamps:      %s", out["timestamps_ms"].shape)
    return out


def load_truth(path: Path) -> dict:
    """Load synthetic truth data if available."""
    if not path.exists():
        log.warning("No truth.npz found at %s — skipping truth comparison", path)
        return {}
    data = np.load(str(path))
    log.info("Loaded truth data from %s", path)
    return {k: data[k] for k in data.files}


# ═══════════════════════════════════════════════════════════════════════════════
# Steps
# ═══════════════════════════════════════════════════════════════════════════════

def step1_load(stage1_path: Path, truth_path: Path) -> tuple:
    log.info("━" * 60)
    log.info("STEP 1 — Loading Stage 1 data")
    log.info("━" * 60)
    s1    = load_stage1(stage1_path)
    truth = load_truth(truth_path)
    return s1, truth


def step2_build_grid(params: dict):
    log.info("━" * 60)
    log.info("STEP 2 — Building sub-aperture grid")
    log.info("━" * 60)
    grid = build_grid(
        n_sa_x       = params["n_sa_x"],
        n_sa_y       = params["n_sa_y"],
        pix_per_sa   = params["pix_per_sa"],
        pupil_cx     = params["pupil_cx"],
        pupil_cy     = params["pupil_cy"],
        pupil_radius = params["pupil_radius"],
    )
    log.info("✓ Grid: %d×%d  |  %d valid SAs", grid.n_sa_x, grid.n_sa_y,
             grid.n_sa_valid)
    return grid


def step3_calibrate_modal(grid, params: dict, n_modes: int,
                           svd_threshold: float,
                           save_dir: Optional[Path] = None) -> ModalReconstructor:
    """
    Build and optionally save the modal reconstructor.

    The sub-aperture centres need to be in normalised pupil coordinates
    ([-1, +1]) for the Zernike gradient formulae to work on the unit disk.
    We achieve this by:
        x_norm[k] = (cx_px[k] - pupil_cx_px) / pupil_radius_px
        y_norm[k] = (cy_px[k] - pupil_cy_px) / pupil_radius_px
    """
    log.info("━" * 60)
    log.info("STEP 3a — Calibrating modal (Zernike) reconstructor "
             "(%d modes, SVD threshold=%.1e)", n_modes, svd_threshold)
    log.info("━" * 60)

    # Normalise sub-aperture centres to unit disk
    sa_cx_norm = (grid.cx_ref - params["pupil_cx"]) / params["pupil_radius"]
    sa_cy_norm = (grid.cy_ref - params["pupil_cy"]) / params["pupil_radius"]

    recon = ModalReconstructor(
        n_modes        = n_modes,
        svd_threshold  = svd_threshold,
        output_grid_px = 64,
    )
    recon.calibrate(
        sa_cx_norm      = sa_cx_norm,
        sa_cy_norm      = sa_cy_norm,
        valid_mask      = grid.valid,
        pupil_radius_px = params["pupil_radius"],
    )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        recon.save(save_dir / "modal_reconstructor.npz")

    log.info("✓ Modal reconstructor calibrated")
    return recon


def step3_calibrate_zonal(grid, params: dict,
                           save_dir: Optional[Path] = None) -> ZonalReconstructor:
    """Build and optionally save the zonal (Fried) reconstructor."""
    log.info("━" * 60)
    log.info("STEP 3b — Calibrating zonal (Fried) reconstructor")
    log.info("━" * 60)

    recon = ZonalReconstructor(
        n_sa_x   = params["n_sa_x"],
        n_sa_y   = params["n_sa_y"],
        d_sa     = params["lenslet_size"],
        use_lsqr = True,
    )

    # Build 2D pupil mask for sub-apertures
    pupil_sa_mask = grid.valid.reshape(params["n_sa_y"], params["n_sa_x"])
    recon.calibrate(pupil_sa_mask)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        recon.save(save_dir / "zonal_reconstructor.npz")

    log.info("✓ Zonal reconstructor calibrated")
    return recon


def step4_reconstruct(recon, s1: dict) -> WavefrontResult:
    """Run reconstruction over all frames."""
    name = recon.__class__.__name__
    log.info("━" * 60)
    log.info("STEP 4 — Running %s on %d frames", name, s1["s_x"].shape[0])
    log.info("━" * 60)

    t0 = time.perf_counter()
    result = recon.reconstruct(
        s_x_all      = s1["s_x"],
        s_y_all      = s1["s_y"],
        valid_all    = s1["slope_valid"],
        timestamps_ms= s1["timestamps_ms"],
    )
    t1 = time.perf_counter()

    n = result.n_frames
    log.info("✓ %s: %.2f s  (%.1f ms/frame)",
             name, t1-t0, (t1-t0)/n*1e3)
    result.print_summary()
    return result


def step5_validate(wf_result: WavefrontResult,
                   truth: dict,
                   params: dict,
                   grid,
                   out_dir: Path):
    """
    If synthetic truth is available, compare reconstructed phase against
    it.  The truth phase screen is in radians (output of generate_turbulence.py)
    but spans the full aperture screen — we interpolate / reshape it to
    match the reconstruction output grid.

    For the modal reconstructor the output is a 64×64 grid normalised to
    the unit pupil disk.  For the zonal reconstructor it is (n_sa_y+1)×
    (n_sa_x+1) nodes.

    The comparison is done on a frame-by-frame scatter: modal sum vs
    the truth phase sub-sampled at the Zernike evaluation grid.
    """
    if "phase_crops" not in truth:
        log.info("No phase_crops in truth — skipping phase comparison.")
        return

    log.info("━" * 60)
    log.info("STEP 5 — Validating against synthetic truth")
    log.info("━" * 60)

    # Use the middle frame for the side-by-side comparison figure
    n       = wf_result.n_frames
    fi      = n // 2
    G       = wf_result.phase_maps.shape[-1]

    # Resize (interpolate) truth phase crop to the output grid size
    from PIL import Image as PILImage

    phase_crop_raw = truth["phase_crops"][fi]           # [Np, Np]  rad
    phase_truth_resized = np.asarray(
        PILImage.fromarray(phase_crop_raw).resize((G, G), PILImage.BILINEAR),
        dtype=np.float32
    )

    # Zero-mean both inside pupil (remove piston)
    pm = wf_result.phase_maps[fi]
    mask = wf_result.pupil_mask

    pm_pupil = pm[mask]
    pt_pupil = phase_truth_resized[mask]
    phase_meas_z  = pm  - pm_pupil.mean()
    phase_truth_z = phase_truth_resized - pt_pupil.mean()

    vis2.plot_truth_phase_comparison(
        phase_meas   = phase_meas_z,
        phase_truth  = phase_truth_z,
        pupil_mask   = mask,
        frame_idx    = fi,
        save_path    = str(out_dir / "stage2_truth_comparison.png"),
    )
    log.info("  ✓ stage2_truth_comparison.png")

    # Residual statistics across all frames
    residuals = []
    for i in range(n):
        pc = truth["phase_crops"][i]
        pc_r = np.asarray(
            PILImage.fromarray(pc).resize((G, G), PILImage.BILINEAR),
            dtype=np.float32
        )
        pm_i = wf_result.phase_maps[i]
        if mask.shape == pm_i.shape:
            r = (pm_i - pc_r)[mask]
            residuals.append(float(np.std(r)))

    if residuals:
        arr = np.array(residuals)
        log.info("  Phase residual (measured − truth):  "
                 "mean=%.3f rad  std=%.3f rad  max=%.3f rad",
                 arr.mean(), arr.std(), arr.max())


def step6_visualise(wf_result: WavefrontResult,
                    out_dir: Path,
                    wavelength_nm: float = 633.0):
    """Produce all Stage 2 diagnostic plots."""
    log.info("━" * 60)
    log.info("STEP 6 — Visualisation")
    log.info("━" * 60)

    fi = wf_result.n_frames // 2   # representative middle frame

    # ── 1. Phase map (middle frame) ─────────────────────────────────────────
    vis2.plot_phase_map(
        wf_result.phase_maps[fi],
        wf_result.pupil_mask,
        title         = f"Reconstructed Wavefront — Frame {fi}  "
                        f"({wf_result.method})",
        wavelength_nm = wavelength_nm,
        save_path     = str(out_dir / "stage2_phase_map.png"),
    )
    log.info("  ✓ stage2_phase_map.png")

    # ── 2. Zernike spectrum (modal only) ────────────────────────────────────
    if wf_result.zernike_coeffs is not None:
        vis2.plot_zernike_spectrum(
            wf_result.zernike_coeffs,
            title     = "Zernike Spectrum (all frames)",
            save_path = str(out_dir / "stage2_zernike_spectrum.png"),
        )
        log.info("  ✓ stage2_zernike_spectrum.png")

        # ── 3. Temporal mode evolution ───────────────────────────────────────
        vis2.plot_zernike_temporal(
            wf_result.zernike_coeffs,
            wf_result.timestamps_ms,
            modes     = [2, 3, 4, 5, 6],
            save_path = str(out_dir / "stage2_zernike_timeseries.png"),
        )
        log.info("  ✓ stage2_zernike_timeseries.png")

    # ── 4. Strehl / WFE time series ─────────────────────────────────────────
    vis2.plot_strehl_timeseries(
        wf_result,
        save_path = str(out_dir / "stage2_strehl_timeseries.png"),
    )
    log.info("  ✓ stage2_strehl_timeseries.png")

    # ── 5. 4-panel summary ───────────────────────────────────────────────────
    coeffs_fi = (wf_result.zernike_coeffs[fi]
                 if wf_result.zernike_coeffs is not None else None)
    vis2.plot_reconstruction_summary(
        wf_result.phase_maps[fi],
        wf_result.pupil_mask,
        coeffs_fi,
        wf_result,
        frame_idx = fi,
        save_path = str(out_dir / "stage2_summary.png"),
    )
    log.info("  ✓ stage2_summary.png")

    log.info("✓ All Stage 2 plots saved to %s", out_dir)


def step7_save(wf_result: WavefrontResult, out_dir: Path) -> Path:
    """Save Stage 2 results for downstream Stage 3 consumption."""
    log.info("━" * 60)
    log.info("STEP 7 — Saving Stage 2 results")
    log.info("━" * 60)

    save_path = out_dir / "stage2_results.npz"
    payload   = dict(
        phase_maps     = wf_result.phase_maps,
        residual_rms   = wf_result.residual_rms,
        timestamps_ms  = wf_result.timestamps_ms,
        strehl         = wf_result.strehl_estimate,
        pupil_mask     = wf_result.pupil_mask.astype(np.uint8),
        method         = np.array([wf_result.method]),
        n_modes        = np.array([wf_result.n_modes]),
    )
    if wf_result.zernike_coeffs is not None:
        payload["zernike_coeffs"] = wf_result.zernike_coeffs

    np.savez_compressed(str(save_path), **payload)
    log.info("✓ Stage 2 results → %s  (%.1f MB)",
             save_path, save_path.stat().st_size / 1e6)
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

# Need Optional for Python 3.9 compat (typing module)


def main():
    parser = argparse.ArgumentParser(description="SH-WFS Pipeline — Stage 2")
    parser.add_argument("--config",   default="config/sensor_config.yaml")
    parser.add_argument("--stage1",   default="outputs/plots/stage1_results.npz")
    parser.add_argument("--method",   default=None,
                        choices=["modal_zernike", "zonal_fried", "both"])
    parser.add_argument("--n-modes",  type=int, default=None)
    parser.add_argument("--svd",      type=float, default=None,
                        help="SVD truncation threshold")
    parser.add_argument("--save-matrices", action="store_true",
                        help="Save calibrated reconstructor matrices")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    config   = load_config(args.config)
    params   = resolve_params(config)
    out_dir  = ROOT / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CLI overrides
    method   = args.method or params["recon_method"]
    n_modes  = args.n_modes or params["n_zernike_modes"]
    svd_thr  = float(args.svd or params["svd_threshold"])
    save_dir = (ROOT / "outputs" / "recon") if args.save_matrices else None

    wavelength_nm = params["wavelength"] * 1e9

    log.info("╔" + "═" * 58 + "╗")
    log.info("║  SH-WFS Pipeline — Stage 2: Wavefront Reconstruction     ║")
    log.info("╚" + "═" * 58 + "╝")
    log.info("Method:  %s", method)
    log.info("Modes:   %d  |  SVD threshold: %.1e", n_modes, svd_thr)

    # ── Steps ────────────────────────────────────────────────────────────────
    s1, truth = step1_load(
        ROOT / args.stage1,
        ROOT / "outputs" / "truth.npz",
    )
    grid = step2_build_grid(params)

    def _run_modal():
        recon  = step3_calibrate_modal(grid, params, n_modes, svd_thr, save_dir)
        result = step4_reconstruct(recon, s1)
        step5_validate(result, truth, params, grid, out_dir)
        step6_visualise(result, out_dir, wavelength_nm)
        step7_save(result, out_dir)
        return result

    def _run_zonal():
        recon  = step3_calibrate_zonal(grid, params, save_dir)
        result = step4_reconstruct(recon, s1)
        step6_visualise(result, out_dir, wavelength_nm)
        step7_save(result, out_dir)
        return result

    if method == "modal_zernike":
        result = _run_modal()
    elif method == "zonal_fried":
        result = _run_zonal()
    elif method == "both":
        log.info("Running BOTH reconstructors (modal first, then zonal)")
        result_m = _run_modal()
        result_z = _run_zonal()
        # Re-save the modal result as the canonical stage2_results.npz
        step7_save(result_m, out_dir)
        result    = result_m   # use modal for the summary printout
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Summary printout ─────────────────────────────────────────────────────
    sr = result
    wfe_nm = sr.phase_rms.mean() * wavelength_nm / (2 * np.pi)
    print()
    print("╔" + "═" * 58 + "╗")
    print("║              STAGE 2 COMPLETE                            ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Method              : {method:<33}║")
    print(f"║  Frames processed    : {sr.n_frames:<33d}║")
    print(f"║  Modes / nodes       : {sr.n_modes:<33d}║")
    print(f"║  Mean WFE RMS        : {sr.phase_rms.mean()*1e3:.2f} mrad   "
          f"({wfe_nm:.1f} nm){'':<8}║")
    print(f"║  Mean Strehl         : {sr.strehl_estimate.mean():.3f}{'':<33}║")
    print(f"║  Mean slope residual : {sr.residual_rms.mean()*1e3:.3f} mrad{'':<26}║")
    print("╚" + "═" * 58 + "╝")
    print(f"\nNext: run Stage 3 (turbulence characterisation) using:")
    print(f"      python run_stage3.py   "
          f"[inputs: stage1_results.npz + stage2_results.npz]")


if __name__ == "__main__":
    main()
