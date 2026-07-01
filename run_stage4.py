#!/usr/bin/env python3
"""
run_stage4.py — SH-WFS Pipeline: Stage 4 DM Actuator Map & Control
=====================================================================
Reads Stage 2 wavefront reconstruction (stage2_results.npz) and computes
the deformable mirror actuator commands that would conjugate each frame's
aberration, then evaluates the residual wavefront after correction.

Steps
-----
  1. Load Stage 2 results and sensor config
  2. Build ActuatorMapper (Gaussian IFs in Fried geometry)
  3. Calibrate (build F matrix, SVD pseudo-inverse)
  4. Run correction over all frames
  5. Produce diagnostic plots
  6. Save Stage 4 results to outputs/plots/stage4_results.npz

Run
---
    python run_stage4.py
    python run_stage4.py --svd 1e-2    # more regularization
    python run_stage4.py --frame 42    # which frame for single-frame plots
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "python"))

from actuator_mapper import ActuatorMapper, DmResult
import visualizer_stage4 as vis4

# Also need WavefrontResult to reconstruct from the npz
# We reconstruct a lightweight proxy instead of importing wavefront_reconstructor
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ── Lightweight proxy for WavefrontResult loaded from disk ───────────────────

@dataclass
class WavefrontResultProxy:
    """Minimal proxy for WavefrontResult loaded from a .npz file."""
    phase_maps:      np.ndarray
    pupil_mask:      np.ndarray
    timestamps_ms:   np.ndarray
    strehl_estimate: np.ndarray
    zernike_coeffs:  Optional[np.ndarray]
    method:          str
    n_modes:         int

    @property
    def n_frames(self):
        return self.phase_maps.shape[0]


def load_wavefront_result(path: Path) -> WavefrontResultProxy:
    log.info("Loading Stage 2 results from %s", path)
    d = np.load(str(path))

    phase_maps    = d["phase_maps"].astype(np.float32)
    pupil_mask    = d["pupil_mask"].astype(bool)
    timestamps_ms = d["timestamps_ms"].astype(np.float32)
    strehl        = d["strehl"].astype(np.float32)
    method        = str(d["method"][0])
    n_modes       = int(d["n_modes"][0])
    coeffs        = d["zernike_coeffs"].astype(np.float32) if "zernike_coeffs" in d else None

    log.info("  Phase maps: %s   method: %s   n_modes: %d",
             phase_maps.shape, method, n_modes)
    return WavefrontResultProxy(
        phase_maps      = phase_maps,
        pupil_mask      = pupil_mask,
        timestamps_ms   = timestamps_ms,
        strehl_estimate = strehl,
        zernike_coeffs  = coeffs,
        method          = method,
        n_modes         = n_modes,
    )


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_mapper_from_config(config: dict,
                              svd_threshold: float,
                              output_grid_px: int = 64,
                              ) -> ActuatorMapper:
    """Construct ActuatorMapper with parameters from sensor_config.yaml."""
    mla = config["mla"]
    tel = config["telescope"]
    dm  = config["dm"]

    n_sa_x   = mla["n_sa_x"]
    n_sa_y   = mla["n_sa_y"]
    n_act_x  = dm["n_actuators_x"]
    n_act_y  = dm["n_actuators_y"]

    # Fried geometry: actuator pitch = lenslet_size; d_act in normalised coords = 2/n_sa_x
    d_act_norm = 2.0 / n_sa_x

    # Max stroke in radians: stroke_m * 4π / λ
    wavelength = tel["wavelength"]
    max_stroke_rad = dm["max_stroke"] * 4.0 * np.pi / wavelength

    return ActuatorMapper(
        n_act_x        = n_act_x,
        n_act_y        = n_act_y,
        d_act_norm     = d_act_norm,
        coupling       = dm["coupling_coefficient"],
        max_stroke_rad = max_stroke_rad,
        svd_threshold  = svd_threshold,
        output_grid_px = output_grid_px,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SH-WFS Pipeline — Stage 4")
    parser.add_argument("--config", default="config/sensor_config.yaml")
    parser.add_argument("--stage2", default="outputs/plots/stage2_results.npz",
                        help="Path to Stage 2 results NPZ")
    parser.add_argument("--svd",   type=float, default=None,
                        help="SVD threshold (overrides config if given)")
    parser.add_argument("--frame", type=int,   default=0,
                        help="Frame index for single-frame diagnostic plots")
    args = parser.parse_args()

    config = load_config(args.config)

    out_dir = ROOT / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("╔" + "═" * 58 + "╗")
    log.info("║  SH-WFS Pipeline — Stage 4: DM Actuator Map & Control  ║")
    log.info("╚" + "═" * 58 + "╝")

    # ── Load Stage 2 wavefront results ──────────────────────────────────────
    s2_path = ROOT / args.stage2
    wf = load_wavefront_result(s2_path)

    # Clamp frame index
    frame_idx = min(args.frame, wf.n_frames - 1)
    if frame_idx != args.frame:
        log.warning("Requested frame %d out of range; using frame %d",
                    args.frame, frame_idx)

    # ── SVD threshold ─────────────────────────────────────────────────────────
    svd_thr = float(args.svd if args.svd is not None else
                    config.get("reconstruction", {}).get("svd_threshold", 1e-3))
    log.info("SVD threshold: %.1e", svd_thr)

    # ── Output grid size — must match Stage 2 ───────────────────────────────
    output_grid_px = int(wf.phase_maps.shape[1])  # square grid from Stage 2

    # ── Build and calibrate ActuatorMapper ──────────────────────────────────
    mapper = build_mapper_from_config(config, svd_thr, output_grid_px)
    mapper.calibrate(wf.pupil_mask)

    # ── Save influence function matrix ────────────────────────────────────
    mapper.save(out_dir / "stage4_actuator_mapper.npz")

    # ── Run DM correction over all frames ────────────────────────────────
    wavelength = config["telescope"]["wavelength"]
    dm_result  = mapper.run(wf, wavelength_m=wavelength)
    dm_result.print_summary()

    # ── Diagnostic plots ─────────────────────────────────────────────────
    log.info("Generating Stage 4 plots...")

    vis4.plot_actuator_map(
        dm_result, frame_idx=frame_idx,
        save_path=str(out_dir / "stage4_actuator_map.png"))

    vis4.plot_correction_comparison(
        dm_result, wf, frame_idx=frame_idx,
        save_path=str(out_dir / "stage4_correction_comparison.png"))

    vis4.plot_strehl_improvement(
        dm_result,
        save_path=str(out_dir / "stage4_strehl_improvement.png"))

    vis4.plot_stroke_statistics(
        dm_result,
        save_path=str(out_dir / "stage4_stroke_statistics.png"))

    vis4.plot_stage4_summary(
        dm_result, wf, frame_idx=frame_idx,
        save_path=str(out_dir / "stage4_summary.png"))

    log.info("✓ All Stage 4 plots saved to %s", out_dir)

    # ── Save Stage 4 results ─────────────────────────────────────────────
    save_path = out_dir / "stage4_results.npz"
    np.savez_compressed(
        str(save_path),
        actuator_commands   = dm_result.actuator_commands,
        stroke_um           = dm_result.stroke_um,
        corrected_phase     = dm_result.corrected_phase,
        uncorrected_strehl  = dm_result.uncorrected_strehl,
        corrected_strehl    = dm_result.corrected_strehl,
        uncorrected_wfe_rms = dm_result.uncorrected_wfe_rms,
        corrected_wfe_rms   = dm_result.corrected_wfe_rms,
        pupil_mask          = dm_result.pupil_mask,
        timestamps_ms       = dm_result.timestamps_ms,
        n_act_x             = np.array([dm_result.n_act_x]),
        n_act_y             = np.array([dm_result.n_act_y]),
        saturation_frac     = np.array([dm_result.saturation_frac]),
        wfe_improvement     = np.array([dm_result.wfe_improvement_factor]),
    )
    log.info("✓ Stage 4 results → %s", save_path)

    # ── Final summary banner ─────────────────────────────────────────────
    u = dm_result.uncorrected_wfe_rms
    c = dm_result.corrected_wfe_rms
    W = 58  # banner inner width
    def row(label, value):
        content = f"  {label}: {value}"
        return "║" + content + " " * (W - len(content)) + "║"

    print()
    print("╔" + "═" * W + "╗")
    print("║              STAGE 4 COMPLETE" + " " * 28 + "║")
    print("╠" + "═" * W + "╣")
    print(row("Actuators          ", f"{dm_result.n_act_x}×{dm_result.n_act_y}"))
    print(row("Uncorrected WFE RMS", f"{u.mean()*1e3:.2f} mrad"))
    print(row("Corrected WFE RMS  ", f"{c.mean()*1e3:.2f} mrad"))
    print(row("WFE improvement    ", f"{dm_result.wfe_improvement_factor:.1f}x"))
    print(row("Corrected Strehl   ", f"{dm_result.corrected_strehl.mean():.4f}"))
    print(row("Actuator saturation", f"{dm_result.saturation_frac*100:.1f}%"))
    print("╚" + "═" * W + "╝")


if __name__ == "__main__":
    main()