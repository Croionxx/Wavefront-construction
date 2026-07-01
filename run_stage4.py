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