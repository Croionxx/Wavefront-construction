"""
generate_turbulence.py — Kolmogorov Phase Screen + SH-WFS Simulator
====================================================================
Generates a realistic time-series of Shack-Hartmann WFS detector frames
simulating atmospheric turbulence in a laboratory setup.

Physics implemented
-------------------
1. Von Kármán phase screen (FFT method) with proper PSD normalisation
2. Taylor's frozen flow for temporal evolution
3. SH-WFS lenslet integration (mean gradient per sub-aperture)
4. Gaussian PSF spot rendering with photon shot noise + read noise
5. Circular pupil masking

Outputs
-------
  - BMP files: outputs/frames/frame_XXXX.bmp
  - metadata.json: complete simulation parameters + ground-truth slopes
  - truth.npz:   ground-truth phase maps, slopes, r0

Usage
-----
  python synthetic/generate_turbulence.py
  python synthetic/generate_turbulence.py --config config/sensor_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── Allow running as a script from the repo root ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase screen generation
# ═══════════════════════════════════════════════════════════════════════════════

def von_karman_psd(fx: np.ndarray,
                   fy: np.ndarray,
                   r0: float,
                   L0: float = 30.0) -> np.ndarray:
    """
    Von Kármán turbulence PSD:
        Φ(f) = 0.0229 * r0^(-5/3) * (f² + f0²)^(-11/6)
    where f0 = 1/L0.

    Returns PSD in units of [rad² · m²] (phase variance per spatial freq).
    """
    f0  = 1.0 / L0
    f2  = fx**2 + fy**2
    f2[0, 0] = f0**2          # avoid singularity at DC; DC is zero-mean
    psd = 0.0229 * r0**(-5/3) * (f2 + f0**2)**(-11/6)
    psd[0, 0] = 0.0            # zero-mean phase
    return psd


def generate_phase_screen(N:   int,
                           D:   float,
                           r0:  float,
                           L0:  float = 30.0,
                           seed: int  = None) -> np.ndarray:
    """
    Generate a single realisation of a turbulent phase screen using
    the FFT (spectral) method.

    Parameters
    ----------
    N    : grid size (NxN pixels)
    D    : physical size of the screen [m]
    r0   : Fried parameter [m]
    L0   : outer scale [m]
    seed : random seed for reproducibility

    Returns
    -------
    phase : float64 [N, N]  wavefront phase in radians
    """
    rng = np.random.default_rng(seed)
    dx  = D / N                             # grid spacing [m]

    # Spatial frequency grid [cycles/m]
    freq = np.fft.fftfreq(N, d=dx)         # [N]
    fx, fy = np.meshgrid(freq, freq)        # [N, N]

    psd    = von_karman_psd(fx, fy, r0, L0)  # [N, N]

    # Generate complex white noise in Fourier space
    # Proper normalisation: multiply by sqrt(PSD / (dx*dx)) / N
    # so that IFFT gives phase with correct variance
    noise  = (rng.standard_normal((N, N)) +
              1j * rng.standard_normal((N, N)))
    noise /= np.sqrt(2)                     # each component has σ=1/√2

    phase_ft = noise * np.sqrt(psd / (dx * dx))
    phase    = np.real(np.fft.ifft2(phase_ft)) * N  # scale for IFFT

    return phase


def validate_structure_function(phase: np.ndarray,
                                  D: float,
                                  r0_nominal: float) -> dict:
    """
    Compute the empirical phase structure function and fit r0.
    Used to verify the phase screen has the correct statistics.

    Returns dict with measured r0, fit quality.
    """
    N  = phase.shape[0]
    dx = D / N
    max_sep = N // 4

    r_vals  = []
    sf_vals = []

    for lag in range(1, max_sep):
        diff  = phase[:, lag:] - phase[:, :-lag]
        sf    = np.mean(diff**2)
        r_vals.append(lag * dx)
        sf_vals.append(sf)

    r_arr  = np.array(r_vals)
    sf_arr = np.array(sf_vals)

    # Fit: D_phi(r) = 6.88 * (r/r0)^(5/3)
    # log: log(sf) = log(6.88) + 5/3*log(r) - 5/3*log(r0)
    log_r  = np.log(r_arr)
    log_sf = np.log(np.clip(sf_arr, 1e-12, None))
    A      = np.column_stack([np.ones_like(log_r), log_r])
    coef   = np.linalg.lstsq(A, log_sf, rcond=None)[0]
    slope  = coef[1]
    r0_fit = np.exp((np.log(6.88) - coef[0]) / (5/3))

    return {
        "r0_nominal": r0_nominal,
        "r0_fit":     float(r0_fit),
        "sf_slope":   float(slope),
        "r_vals":     r_vals,
        "sf_vals":    sf_vals,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SH-WFS Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_subaperture_slopes(phase:      np.ndarray,
                                N_screen:  int,
                                D_screen:  float,
                                n_sa_x:    int,
                                n_sa_y:    int,
                                pupil_mask: np.ndarray) -> tuple:
    """
    Integrate wavefront gradients over each sub-aperture to get
    the slopes measured by the MLA.

    The phase screen is sampled at the lenslet positions.  For a sub-aperture
    (i,j) spanning pixels [xa:xb, ya:yb] on the screen:
        sx[i,j] = mean(∂φ/∂x) over the tile  [rad/m]
        sy[i,j] = mean(∂φ/∂y) over the tile  [rad/m]

    Parameters
    ----------
    phase       : [N_screen, N_screen] phase in radians
    N_screen    : screen pixels
    D_screen    : screen physical size [m]
    n_sa_x/y    : number of sub-apertures
    pupil_mask  : bool [n_sa_y, n_sa_x]

    Returns
    -------
    sx, sy : float [n_sa_y, n_sa_x]  slopes in [rad/m]
    """
    dx     = D_screen / N_screen
    tiles_x = np.array_split(np.arange(N_screen), n_sa_x)
    tiles_y = np.array_split(np.arange(N_screen), n_sa_y)

    # Compute gradient of entire screen first (central differences)
    gy, gx = np.gradient(phase, dx)   # [N, N] in rad/m

    sx_2d = np.zeros((n_sa_y, n_sa_x))
    sy_2d = np.zeros((n_sa_y, n_sa_x))

    for j, ty in enumerate(tiles_y):
        for i, tx in enumerate(tiles_x):
            if not pupil_mask[j, i]:
                continue
            sx_2d[j, i] = gx[np.ix_(ty, tx)].mean()
            sy_2d[j, i] = gy[np.ix_(ty, tx)].mean()

    return sx_2d, sy_2d


def render_shwfs_frame(sx_2d:       np.ndarray,
                        sy_2d:       np.ndarray,
                        pupil_mask:  np.ndarray,
                        n_sa_x:      int,
                        n_sa_y:      int,
                        pix_per_sa:  int,
                        focal_length: float,
                        pixel_size:  float,
                        spot_sigma:  float,
                        n_photons:   float,
                        read_noise:  float,
                        rng:         np.random.Generator) -> np.ndarray:
    """
    Render a simulated SH-WFS detector frame from slope maps.

    Spot displacement on detector:
        Δx [pix] = f_lens * sx [rad/m] * d_sa [m] / pixel_size [m]
    Wait — actually sx is already in [rad/m], and a lenslet measures
    the average tilt angle of the wavefront across its aperture.
    Tilt angle [rad] = sx * d_sa  (for a sub-aperture of size d_sa).
    Spot displacement [m] = f * tilt_angle = f * sx * d_sa
    In pixels: Δx = f * sx * d_sa / pixel_size

    For simplicity here: Δx_pix = f_lens * sx * d_sa / pixel_size
    where d_sa = total_size / n_sa  (physical lenslet size).

    Parameters
    ----------
    Returns
    -------
    frame : uint16 [H_det, W_det]
    """
    H_det = n_sa_y * pix_per_sa
    W_det = n_sa_x * pix_per_sa

    frame = np.zeros((H_det, W_det), dtype=np.float64)

    # Physical size of one sub-aperture [m]
    # (lenslet_size = pupil_D / n_sa, but here we compute from pixel budget)
    # spot_sigma is given in pixels directly

    # Build PSF kernel (Gaussian)
    # Kernel extends ±3σ each side
    ks   = int(np.ceil(3 * spot_sigma)) * 2 + 1   # kernel size
    half = ks // 2
    kx   = np.arange(ks) - half
    ky   = np.arange(ks) - half
    KX, KY = np.meshgrid(kx, ky)
    psf  = np.exp(-0.5 * (KX**2 + KY**2) / spot_sigma**2)
    psf /= psf.sum()

    for j in range(n_sa_y):
        for i in range(n_sa_x):
            if not pupil_mask[j, i]:
                continue

            # Reference spot centre in detector pixels (0-indexed)
            ref_cx = i * pix_per_sa + (pix_per_sa - 1) / 2.0
            ref_cy = j * pix_per_sa + (pix_per_sa - 1) / 2.0

            # Spot displacement [pixels]
            # Correct SH-WFS formula:
            # tilt angle θ = (λ/2π) * slope_in_rad_per_m  [radians]
            # spot displacement = f * θ / pixel_size        [pixels]
            TWO_PI = 6.283185307179586
            _wavelength = 633e-9   # HeNe default
            dx_pix = focal_length * (_wavelength / TWO_PI) * sx_2d[j, i] / pixel_size
            dy_pix = focal_length * (_wavelength / TWO_PI) * sy_2d[j, i] / pixel_size

            # Displaced spot centre
            cx = ref_cx + dx_pix
            cy = ref_cy + dy_pix

            # Render Gaussian PSF at sub-pixel position
            cx_int = int(round(cx))
            cy_int = int(round(cy))

            sub_cx = cx - cx_int   # sub-pixel offset
            sub_cy = cy - cy_int

            # Shift PSF for sub-pixel accuracy
            kx_sh  = KX - sub_cx
            ky_sh  = KY - sub_cy
            psf_sh = np.exp(-0.5 * (kx_sh**2 + ky_sh**2) / spot_sigma**2)
            psf_sh /= psf_sh.sum()

            # Place PSF on detector
            for dy_k in range(ks):
                for dx_k in range(ks):
                    px = cx_int - half + dx_k
                    py = cy_int - half + dy_k
                    if 0 <= px < W_det and 0 <= py < H_det:
                        frame[py, px] += n_photons * psf_sh[dy_k, dx_k]

    # Add Poisson shot noise
    frame = rng.poisson(np.maximum(frame, 0)).astype(np.float64)

    # Add Gaussian read noise
    frame += rng.normal(0.0, read_noise, frame.shape)

    # Clip to 0
    frame = np.clip(frame, 0.0, None)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Main Sequence Generator
# ═══════════════════════════════════════════════════════════════════════════════

class TurbulenceSequenceGenerator:
    """
    Generates a time-series of simulated SH-WFS frames using
    Kolmogorov turbulence and Taylor's frozen flow model.

    Parameters
    ----------
    See sensor_config.yaml for all parameters.
    """

    def __init__(self, config: dict):
        c   = config
        cam = c["camera"]
        mla = c["mla"]
        tel = c["telescope"]
        tur = c["turbulence"]
        pup = c["pupil"]
        syn = c["synthetic"]

        # Camera
        self.pixel_size     = cam["pixel_size"]          # m
        self.frame_interval = cam["frame_interval"]       # s
        self.read_noise     = cam["read_noise_e"]         # electrons
        self.bit_depth      = cam["bit_depth"]

        # MLA
        self.n_sa_x      = mla["n_sa_x"]
        self.n_sa_y      = mla["n_sa_y"]
        self.pix_per_sa  = mla["pixels_per_sa"]
        self.focal_length= mla["focal_length"]           # m
        self.lenslet_size= mla["lenslet_size"]           # m

        # Detector
        self.W_det = self.n_sa_x * self.pix_per_sa
        self.H_det = self.n_sa_y * self.pix_per_sa

        # Pupil
        self.pupil_cx     = pup["center_x"]
        self.pupil_cy     = pup["center_y"]
        self.pupil_radius = pup["radius_px"]
        self.pupil_D      = tel["aperture_diameter"]     # m

        # Turbulence
        self.r0           = tur["r0"]                    # m
        self.L0           = tur["L0"]                    # m
        self.wind_speed   = tur["wind_speed"]            # m/s
        self.wind_dir_deg = tur["wind_direction_deg"]
        self.n_photons    = tur["photons_per_sa"]

        # Synthetic
        self.n_frames     = syn["n_frames"]
        self.screen_factor= syn["screen_size_factor"]
        self.seed         = syn["random_seed"]
        self.out_dir      = Path(syn["output_dir"])

        # Derived
        self.rng  = np.random.default_rng(self.seed)
        vd        = np.deg2rad(self.wind_dir_deg)
        self.vx   = self.wind_speed * np.cos(vd)        # m/s
        self.vy   = self.wind_speed * np.sin(vd)

        # Spot PSF sigma in pixels
        # Diffraction-limited: σ ≈ λ f / (d_sa * pixel_size * 2π) * 2.35 ≈ ...
        # Approximate: 1.5–2.5 pixels is typical
        wavelength = tel.get("wavelength", 633e-9)
        self.spot_sigma = (wavelength * self.focal_length /
                           (self.lenslet_size * self.pixel_size)) * 0.42
        self.spot_sigma = max(1.0, min(self.spot_sigma, 3.5))  # clamp to [1, 3.5] px

        log.info("Spot sigma: %.2f px", self.spot_sigma)

        # Phase screen parameters
        # Screen covers screen_factor * aperture.  Temporal evolution uses
        # cyclic wrap-around so the screen does NOT need to span the full
        # wind drift.  Cap at MAX_SCREEN_PX to keep memory < 64 MB.
        MAX_SCREEN_PX = 512
        self.D_screen    = self.pupil_D * self.screen_factor
        pixels_per_m     = self.n_sa_x * self.pix_per_sa / self.pupil_D
        N_ideal          = int(2 ** np.ceil(np.log2(
                               max(int(np.ceil(self.D_screen * pixels_per_m)), 64))))
        self.N_screen    = min(N_ideal, MAX_SCREEN_PX)
        # Recompute D_screen to match the capped pixel count exactly
        self.D_screen    = self.N_screen / pixels_per_m

        log.info("Phase screen: %d × %d px  (%.2f m × %.2f m)",
                 self.N_screen, self.N_screen, self.D_screen, self.D_screen)

        # Pupil mask on sub-aperture grid [n_sa_y, n_sa_x]
        self.pupil_mask = self._build_pupil_mask()
        log.info("Valid sub-apertures: %d / %d",
                 self.pupil_mask.sum(), self.n_sa_x * self.n_sa_y)

        # Spot displacement scale for reference:
        # 1 rad/m of slope → f * d_sa / pixel_size pixels of displacement
        self.slope_to_pix = self.focal_length / self.pixel_size
        log.info("Slope→pixel scale: %.2f px / (rad/m)", self.slope_to_pix)

    def _build_pupil_mask(self) -> np.ndarray:
        """Boolean [n_sa_y, n_sa_x] pupil mask on the sub-aperture grid."""
        mask = np.zeros((self.n_sa_y, self.n_sa_x), dtype=bool)
        for j in range(self.n_sa_y):
            for i in range(self.n_sa_x):
                cx = i * self.pix_per_sa + (self.pix_per_sa - 1) / 2.0
                cy = j * self.pix_per_sa + (self.pix_per_sa - 1) / 2.0
                r  = np.hypot(cx - self.pupil_cx, cy - self.pupil_cy)
                mask[j, i] = (r <= self.pupil_radius)
        return mask

    def _screen_to_detector_indices(self, offset_x: float, offset_y: float,
                                     ) -> tuple:
        """
        Map a sub-aperture (i,j) + screen offset to the slice of the
        phase screen corresponding to that lenslet at this timestep.
        """
        # Pixels per metre on the phase screen
        ppm = self.N_screen / self.D_screen
        # Pupil starts at centre of screen minus half pupil
        pupil_start_x = (self.N_screen / 2) - (self.n_sa_x * self.pix_per_sa / 2)
        pupil_start_y = (self.N_screen / 2) - (self.n_sa_y * self.pix_per_sa / 2)
        # Offset from wind in screen pixels
        off_x_pix = int(round(offset_x * ppm))
        off_y_pix = int(round(offset_y * ppm))
        return (pupil_start_x + off_x_pix,
                pupil_start_y + off_y_pix)

    def generate(self) -> dict:
        """
        Main entry point. Generate the full frame sequence.

        Returns
        -------
        dict with keys:
          'frames'    : uint16 [N_frames, H, W]
          'sx_truth'  : float  [N_frames, n_sa_y, n_sa_x]  true x-slopes [rad/m]
          'sy_truth'  : float  [N_frames, n_sa_y, n_sa_x]  true y-slopes [rad/m]
          'phase_maps': float  [N_frames, N_screen, N_screen] phase screens (cropped)
          'r0'        : float  Fried parameter used
          'meta'      : dict   all simulation parameters
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        log.info("Generating phase screen (N=%d, r0=%.3f m)...", self.N_screen, self.r0)
        phase_screen = generate_phase_screen(
            N=self.N_screen, D=self.D_screen,
            r0=self.r0, L0=self.L0,
            seed=self.seed
        )

        # Validate structure function on the generated screen
        sf_info = validate_structure_function(phase_screen, self.D_screen, self.r0)
        log.info("Structure function fit: r0_fit=%.4f m  (nominal=%.4f m)  slope=%.3f",
                 sf_info["r0_fit"], self.r0, sf_info["sf_slope"])

        # Preallocate output arrays
        frames    = np.zeros((self.n_frames, self.H_det, self.W_det), dtype=np.uint16)
        sx_truth  = np.zeros((self.n_frames, self.n_sa_y, self.n_sa_x))
        sy_truth  = np.zeros((self.n_frames, self.n_sa_y, self.n_sa_x))
        phase_crops = []

        # ── ADU saturation level
        max_adu = (2 ** self.bit_depth) - 1

        log.info("Rendering %d frames...", self.n_frames)
        for frame_idx in range(self.n_frames):
            t = frame_idx * self.frame_interval

            # Wind-driven offset into the phase screen
            offset_x = self.vx * t
            offset_y = self.vy * t

            # Map to screen pixel offset
            ppm       = self.N_screen / self.D_screen
            off_x_pix = int(round(offset_x * ppm))
            off_y_pix = int(round(offset_y * ppm))

            # Crop the pupil region from the frozen screen
            # Screen centre
            cx0 = self.N_screen // 2 + off_x_pix
            cy0 = self.N_screen // 2 + off_y_pix
            Np  = self.n_sa_x * self.pix_per_sa   # pupil pixels

            xs  = cx0 - Np // 2
            ys  = cy0 - Np // 2
            xe  = xs + Np
            ye  = ys + Np

            # Bounds check with wrap-around (cyclic)
            xs %= self.N_screen; xe %= self.N_screen
            ys %= self.N_screen; ye %= self.N_screen

            # Extract pupil slice (handle wrap-around)
            if xs < xe and ys < ye:
                phase_crop = phase_screen[ys:ye, xs:xe]
            else:
                # Wrap: tile the screen
                tiled = np.tile(phase_screen, (2, 2))
                xs_t  = xs % self.N_screen
                ys_t  = ys % self.N_screen
                phase_crop = tiled[ys_t:ys_t+Np, xs_t:xs_t+Np]

            # Ensure shape matches
            if phase_crop.shape != (Np, Np):
                log.warning("Phase crop shape mismatch at frame %d: %s → padding",
                            frame_idx, phase_crop.shape)
                pc = np.zeros((Np, Np))
                h  = min(Np, phase_crop.shape[0])
                w  = min(Np, phase_crop.shape[1])
                pc[:h, :w] = phase_crop[:h, :w]
                phase_crop = pc

            phase_crops.append(phase_crop.astype(np.float32))

            # Compute sub-aperture slopes from cropped phase
            sx, sy = compute_subaperture_slopes(
                phase_crop, Np, self.pupil_D,
                self.n_sa_x, self.n_sa_y, self.pupil_mask
            )
            sx_truth[frame_idx] = sx
            sy_truth[frame_idx] = sy

            # Render detector frame
            frame_f64 = render_shwfs_frame(
                sx_2d=sx, sy_2d=sy,
                pupil_mask=self.pupil_mask,
                n_sa_x=self.n_sa_x, n_sa_y=self.n_sa_y,
                pix_per_sa=self.pix_per_sa,
                focal_length=self.focal_length,
                pixel_size=self.pixel_size,
                spot_sigma=self.spot_sigma,
                n_photons=self.n_photons,
                read_noise=self.read_noise,
                rng=self.rng
            )

            # Clip and convert to uint16
            frame_u16 = np.clip(frame_f64, 0, max_adu).astype(np.uint16)
            frames[frame_idx] = frame_u16

            # Save BMP
            bmp_path = self.out_dir / f"frame_{frame_idx:04d}.bmp"
            img = Image.fromarray(frame_u16.astype(np.uint8)
                                  if self.bit_depth == 8
                                  else (frame_u16 >> 4).astype(np.uint8))
            img.save(bmp_path)

            if frame_idx % 20 == 0 or frame_idx == self.n_frames - 1:
                rms_sx = np.sqrt(np.mean(sx[self.pupil_mask]**2))
                log.info("  Frame %3d/%d  RMS sx=%.4e rad/m",
                         frame_idx + 1, self.n_frames, rms_sx)

        # ── Save truth data ──────────────────────────────────────────────────
        truth_path = self.out_dir.parent / "truth.npz"
        np.savez_compressed(
            truth_path,
            frames      = frames,
            sx_truth    = sx_truth.astype(np.float32),
            sy_truth    = sy_truth.astype(np.float32),
            phase_crops = np.array(phase_crops),
            pupil_mask  = self.pupil_mask,
            r0          = self.r0,
            L0          = self.L0,
            timestamps_ms = np.arange(self.n_frames) * self.frame_interval * 1e3,
        )
        log.info("Saved truth data: %s", truth_path)

        # ── Save metadata ────────────────────────────────────────────────────
        meta = {
            "n_frames":        self.n_frames,
            "n_sa_x":          self.n_sa_x,
            "n_sa_y":          self.n_sa_y,
            "pix_per_sa":      self.pix_per_sa,
            "focal_length_m":  self.focal_length,
            "pixel_size_m":    self.pixel_size,
            "lenslet_size_m":  self.lenslet_size,
            "frame_interval_s":self.frame_interval,
            "r0_m":            self.r0,
            "L0_m":            self.L0,
            "wind_speed_ms":   self.wind_speed,
            "wind_dir_deg":    self.wind_dir_deg,
            "r0_fit_m":        sf_info["r0_fit"],
            "sf_slope":        sf_info["sf_slope"],
            "spot_sigma_px":   self.spot_sigma,
            "n_photons_per_sa":self.n_photons,
            "read_noise_e":    self.read_noise,
            "bit_depth":       self.bit_depth,
            "pupil_cx_px":     self.pupil_cx,
            "pupil_cy_px":     self.pupil_cy,
            "pupil_radius_px": self.pupil_radius,
            "H_det":           self.H_det,
            "W_det":           self.W_det,
        }
        meta_path = self.out_dir.parent / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log.info("Saved metadata: %s", meta_path)

        return {
            "frames":     frames,
            "sx_truth":   sx_truth,
            "sy_truth":   sy_truth,
            "phase_crops":np.array(phase_crops),
            "r0":         self.r0,
            "meta":       meta,
            "sf_info":    sf_info,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic SH-WFS frame sequence"
    )
    parser.add_argument("--config", default="config/sensor_config.yaml",
                        help="Path to sensor_config.yaml")
    parser.add_argument("--r0", type=float, default=None,
                        help="Override r0 [m]")
    parser.add_argument("--n_frames", type=int, default=None,
                        help="Override number of frames")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.r0 is not None:
        config["turbulence"]["r0"] = args.r0
    if args.n_frames is not None:
        config["synthetic"]["n_frames"] = args.n_frames

    gen    = TurbulenceSequenceGenerator(config)
    result = gen.generate()

    log.info("=" * 60)
    log.info("Generation complete.")
    log.info("  Frames:    %d", result["meta"]["n_frames"])
    log.info("  r0 input:  %.4f m", result["r0"])
    log.info("  r0 fit:    %.4f m", result["sf_info"]["r0_fit"])
    log.info("  Frames in: %s", gen.out_dir)


if __name__ == "__main__":
    main()
