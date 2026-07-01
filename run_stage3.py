#!/usr/bin/env python3
"""
run_stage3.py — SH-WFS Pipeline: Stage 3 Turbulence Characterization
========================================================================
Reads Stage 1 (stage1_results.npz) + Stage 2 (stage2_results.npz) and
estimates r0, tau0, and effective wind speed via two independent methods
each.

Run
---
    python run_stage3.py
    python run_stage3.py --j-min 6   # exclude more low-order modes from
                                       # the Zernike r0 fit
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "python"))

from centroiding import build_grid
from turbulence_characterizer import TurbulenceCharacterizer
import visualizer_stage3 as vis3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SH-WFS Pipeline — Stage 3")
    parser.add_argument("--config",  default="config/sensor_config.yaml")
    parser.add_argument("--stage1",  default="outputs/plots/stage1_results.npz")
    parser.add_argument("--stage2",  default="outputs/plots/stage2_results.npz")
    parser.add_argument("--j-min",   type=int, default=4,
                        help="Lowest Noll index used in the Zernike r0 fit")
    args = parser.parse_args()

    config = load_config(args.config)
    cam, mla, tel = config["camera"], config["mla"], config["telescope"]

    out_dir = ROOT / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("╔" + "═" * 58 + "╗")
    log.info("║  SH-WFS Pipeline — Stage 3: Turbulence Characterization ║")
    log.info("╚" + "═" * 58 + "╝")

    # ── Load Stage 1 / Stage 2 ──────────────────────────────────────────────
    s1_path = ROOT / args.stage1
    s2_path = ROOT / args.stage2

    log.info("Loading Stage 1: %s", s1_path)
    d1 = np.load(str(s1_path))
    s1 = {k: d1[k] for k in d1.files}
    s1["slope_valid"] = s1["slope_valid"].astype(bool)

    s2 = None
    if s2_path.exists():
        log.info("Loading Stage 2: %s", s2_path)
        d2 = np.load(str(s2_path))
        s2 = {k: d2[k] for k in d2.files}
    else:
        log.warning("No Stage 2 results found at %s — Zernike-based r0/tau0 "
                    "will be skipped, falling back to slope-only estimators.",
                    s2_path)

    # ── Grid (needed for physical sub-aperture positions) ───────────────────
    grid = build_grid(
        n_sa_x       = mla["n_sa_x"],
        n_sa_y       = mla["n_sa_y"],
        pix_per_sa   = mla["pixels_per_sa"],
        pupil_cx     = config["pupil"]["center_x"],
        pupil_cy     = config["pupil"]["center_y"],
        pupil_radius = config["pupil"]["radius_px"],
    )

    # ── Run characterizer ────────────────────────────────────────────────────
    tc = TurbulenceCharacterizer(
        aperture_diameter = tel["aperture_diameter"],
        lenslet_size       = mla["lenslet_size"],
        wavelength          = tel["wavelength"],
        frame_interval_ms   = cam["frame_interval"] * 1e3,
    )
    tr = tc.run(s1, s2, grid)
    tr.print_summary()

    # ── Plots ────────────────────────────────────────────────────────────────
    log.info("Generating Stage 3 plots...")
    vis3.plot_zernike_variance_fit(tr, str(out_dir / "stage3_zernike_variance.png"))
    vis3.plot_slope_structure_fn(tr, str(out_dir / "stage3_slope_structure_fn.png"))
    vis3.plot_temporal_structure_fn(tr, str(out_dir / "stage3_temporal_structure_fn.png"))
    vis3.plot_autocorrelation(tr, str(out_dir / "stage3_autocorrelation.png"))
    vis3.plot_stage3_summary(tr, str(out_dir / "stage3_summary.png"))
    log.info("✓ All Stage 3 plots saved to %s", out_dir)

    # ── Save results ─────────────────────────────────────────────────────────
    save_path = out_dir / "stage3_results.npz"
    payload = dict(
        r0_zernike_m       = tr.r0_zernike_m,
        r0_slope_sf_m       = tr.r0_slope_sf_m,
        r0_mean_m           = tr.r0_mean_m,
        tau0_temporal_sf_ms = tr.tau0_temporal_sf_ms,
        tau0_autocorr_ms    = tr.tau0_autocorr_ms,
        v_wind_eff_ms        = tr.v_wind_eff_ms,
        slope_sf_r           = tr.slope_sf_r,
        slope_sf_phi         = tr.slope_sf_phi,
        temporal_sf_tau      = tr.temporal_sf_tau,
        temporal_sf_d         = tr.temporal_sf_d,
    )
    if tr.zernike_variances is not None:
        payload["zernike_variances"] = tr.zernike_variances
        payload["zernike_theory"]    = tr.zernike_theory
    np.savez_compressed(str(save_path), **payload)
    log.info("✓ Stage 3 results → %s", save_path)

    # ── Compare against input truth (if config has known r0) ───────────────
    true_r0 = config.get("turbulence", {}).get("r0")
    if true_r0:
        print()
        print("╔" + "═" * 58 + "╗")
        print("║              STAGE 3 COMPLETE                            ║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  r0 input (config)   : {true_r0*1e2:.3f} cm{'':<27}║")
        print(f"║  r0 (Zernike)        : {tr.r0_zernike_m*1e2:.3f} cm{'':<27}║")
        print(f"║  r0 (slope SF)       : {tr.r0_slope_sf_m*1e2:.3f} cm{'':<27}║")
        print(f"║  tau0 (temporal SF)  : {tr.tau0_temporal_sf_ms:.3f} ms{'':<27}║")
        print(f"║  tau0 (autocorr)     : {tr.tau0_autocorr_ms:.3f} ms{'':<27}║")
        print("╚" + "═" * 58 + "╝")

    print(f"\nNext: run Stage 4 (actuator map / DM control) using:")
    print(f"      python run_stage4.py")


if __name__ == "__main__":
    main()