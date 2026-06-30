# SH-WFS Adaptive Optics Pipeline — Architecture & Design Document

> **Version:** 1.0 | **Author:** Design Document for Wavefront Reconstruction System  
> **Status:** Stage 1 Implemented · Stages 2–4 Designed

---

## Table of Contents

1. [Problem Overview & Physics](#1-problem-overview--physics)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Stage 1 — Frame Ingestion & Centroiding](#3-stage-1--frame-ingestion--centroiding)
4. [Stage 2 — Slope Computation & Wavefront Reconstruction](#4-stage-2--slope-computation--wavefront-reconstruction)
5. [Stage 3 — Turbulence Characterization](#5-stage-3--turbulence-characterization)
6. [Stage 4 — Actuator Map & DM Control](#6-stage-4--actuator-map--dm-control)
7. [Performance Architecture](#7-performance-architecture)
8. [Data Contracts & Interchange Formats](#8-data-contracts--interchange-formats)
9. [Algorithm Selection Rationale](#9-algorithm-selection-rationale)
10. [Validation Strategy](#10-validation-strategy)
11. [Literature & References](#11-literature--references)

---

## 1. Problem Overview & Physics

### 1.1 Atmospheric Turbulence Model

The atmosphere introduces random phase aberrations in a propagating wavefront. The turbulence follows **Kolmogorov statistics** governed by the refractive index structure parameter $C_n^2(h)$. The key statistical descriptor is the **phase structure function**:

$$D_\phi(r) = \langle |\phi(\mathbf{x}) - \phi(\mathbf{x}+\mathbf{r})|^2 \rangle = 6.88 \left(\frac{r}{r_0}\right)^{5/3}$$

where $r_0$ is the **Fried parameter** (atmospheric coherence length). Larger $r_0$ → weaker turbulence. Typical values: $r_0 \in [5\text{ cm}, 30\text{ cm}]$ at $\lambda = 500\text{ nm}$.

The **one-dimensional phase PSD** follows:
$$\Phi_\phi(f) = 0.023\, r_0^{-5/3} \cdot f^{-11/3} \quad [\text{rad}^2 \cdot \text{m}^2]$$

For finite outer scale $L_0$ (von Kármán model):
$$\Phi_\phi(f) = 0.023\, r_0^{-5/3} \left(f^2 + \frac{1}{L_0^2}\right)^{-11/6}$$

### 1.2 Taylor's Frozen Flow Hypothesis

On the timescale of a few milliseconds, turbulence evolves primarily by **advection** of a fixed phase screen with wind velocity $\mathbf{v}$. The temporal structure function is:

$$D_\phi(\tau) = D_\phi(|\mathbf{v}|\,\tau) = 6.88 \left(\frac{v\,\tau}{r_0}\right)^{5/3}$$

The **coherence time** is:
$$\tau_0 = 0.314 \frac{r_0}{|\mathbf{v}|} \approx 0.314 \frac{r_0}{v}$$

For $r_0 = 10\text{ cm}$, $v = 10\text{ m/s}$: $\tau_0 \approx 3.1\text{ ms}$. This sets the AO loop bandwidth requirement.

### 1.3 Shack-Hartmann WFS Principle

An SH-WFS samples the pupil with a **Microlens Array (MLA)**. Each lenslet of diameter $d$ and focal length $f$ focuses light onto a detector. The spot displacement $(\Delta x, \Delta y)$ is proportional to the **average wavefront gradient** over that sub-aperture:

$$\Delta x_{ij} = \frac{f}{\lambda} \cdot \frac{1}{d^2}\iint_{SA_{ij}} \frac{\partial W}{\partial x}\, dx\, dy$$

$$\Delta y_{ij} = \frac{f}{\lambda} \cdot \frac{1}{d^2}\iint_{SA_{ij}} \frac{\partial W}{\partial y}\, dx\, dy$$

In pixel units:
$$s^x_{ij} = \frac{\Delta x_{ij}}{p} = \frac{f}{p} \cdot \langle\nabla_x W\rangle_{ij}, \qquad s^y_{ij} = \frac{\Delta y_{ij}}{p} = \frac{f}{p} \cdot \langle\nabla_y W\rangle_{ij}$$

where $p$ is the detector pixel size.

### 1.4 Fried Geometry

The **Fried geometry** is the standard co-registration of the lenslet grid and actuator grid:

```
Actuators (×) sit at corners of each sub-aperture (□):

  ×---×---×---×
  | □ | □ | □ |
  ×---×---×---×
  | □ | □ | □ |
  ×---×---×---×
  | □ | □ | □ |
  ×---×---×---×
```

For an $N \times N$ lenslet grid → $(N+1) \times (N+1)$ actuator grid. Each actuator influences 4 neighboring sub-apertures. This geometry is critical for the least-squares reconstruction matrix relating slopes to wavefront node values.

---

## 2. Pipeline Architecture

### 2.1 High-Level Data Flow

```
BMP Frame Series
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: FRAME INGESTION & CENTROIDING                         │
│                                                                  │
│  BMP → Float Array → Sub-aperture Grid → Centroid[N_SA]         │
│  Algorithms: CoG | Thresholded-CoG | Windowed-CoG               │
│  Language: C (hot path) + Python (orchestration)                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │  centroids[frame, sa_x, sa_y] = (cx, cy)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: SLOPE COMPUTATION & WAVEFRONT RECONSTRUCTION          │
│                                                                  │
│  (cx,cy) - (ref_cx, ref_cy) → slopes[sx, sy]                   │
│  Reconstruction: Modal (Zernike) | Zonal (Least-squares)        │
│  Output: W(xi, yi) phase map per frame                          │
│  Language: C (matrix ops via LAPACK/GSL) + Python               │
└──────────────────────────┬──────────────────────────────────────┘
                           │  wavefront[frame, x, y] in nm/waves/rad
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: TURBULENCE CHARACTERIZATION                           │
│                                                                  │
│  Time-series of W(xi,yi) → Structure Function D_φ(r, τ)        │
│  Fit → r0, τ0, wind velocity estimate                           │
│  Zernike variance analysis → modal power spectrum               │
│  Language: Python + NumPy/SciPy                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │  r0, τ0, Zernike_variances[]
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: ACTUATOR MAP & DM CONTROL                             │
│                                                                  │
│  -W(xi,yi) → Influence Function Matrix → Actuator strokes       │
│  Inter-actuator coupling compensation                           │
│  Output: A(xi, yi) in actuator stroke units                     │
│  Language: C (real-time path) + Python (calibration)            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Repository Structure

```
sh_wfs_pipeline/
│
├── ARCHITECTURE.md              ← This document
├── README.md
├── requirements.txt
│
├── config/
│   └── sensor_config.yaml       ← All hardware parameters
│
├── src/
│   ├── c/
│   │   ├── centroid.h           ← Stage 1: C centroiding header
│   │   ├── centroid.c           ← Stage 1: C centroiding implementation
│   │   ├── reconstruct.h        ← Stage 2: C reconstruction header (designed)
│   │   ├── reconstruct.c        ← Stage 2: C reconstruction (designed)
│   │   └── Makefile
│   │
│   └── python/
│       ├── __init__.py
│       ├── frame_loader.py      ← Stage 1: BMP/image ingestion
│       ├── centroiding.py       ← Stage 1: Python + C-bridge centroiding
│       ├── slope_computer.py    ← Stage 1/2: Slope computation
│       ├── wavefront_reconstructor.py  ← Stage 2: Zernike/zonal (skeleton)
│       ├── turbulence_characterizer.py ← Stage 3: r0/τ0 estimators (skeleton)
│       ├── actuator_mapper.py          ← Stage 4: DM control (skeleton)
│       └── visualizer.py        ← Visualization utilities
│
├── synthetic/
│   └── generate_turbulence.py   ← Kolmogorov phase screen + SH-WFS sim
│
├── tests/
│   ├── test_centroiding.py
│   └── test_synthetic.py
│
├── notebooks/
│   └── pipeline_demo.ipynb
│
├── outputs/                     ← Generated outputs
│   ├── frames/                  ← Synthetic BMP frames
│   ├── centroids/               ← Centroid NPZ archives
│   └── plots/                   ← Diagnostic plots
│
└── run_stage1.py                ← Main Stage 1 runner
```

---

## 3. Stage 1 — Frame Ingestion & Centroiding

### 3.1 Frame Loading

**Input**: Directory of BMP files (sequential naming `frame_0000.bmp`, etc.)  
**Format**: 8-bit or 16-bit grayscale; pixels represent photon counts + background  
**Output**: `float32` numpy array `[H, W]` normalized to `[0.0, 1.0]`

**Key considerations**:
- 16-bit cameras provide better dynamic range for faint spots
- Bias/dark subtraction if calibration frames are available
- Flat-field correction to normalize lenslet throughput variations

### 3.2 Sub-Aperture Grid Definition

The MLA geometry maps directly to a pixel grid:

```python
# For n_sa × n_sa lenslets, each spanning pix_per_sa × pix_per_sa pixels:
sa_x_start[i] = i * pix_per_sa
sa_y_start[j] = j * pix_per_sa
sa_width = sa_height = pix_per_sa
```

**Pupil mask**: Sub-apertures outside the circular telescope aperture are invalid.  
For aperture diameter $D$ (pixels), sub-aperture $(i,j)$ is valid if:

$$\left(\bar{x}_{ij} - x_c\right)^2 + \left(\bar{y}_{ij} - y_c\right)^2 \leq \left(\frac{D}{2}\right)^2$$

where $(\bar{x}_{ij}, \bar{y}_{ij})$ is the center of sub-aperture $(i,j)$ and $(x_c, y_c)$ is the pupil center.

### 3.3 Centroiding Algorithms

#### Method 1: Center of Gravity (CoG)

The simplest estimator — intensity-weighted centroid:

$$x_c = \frac{\sum_k I_k x_k}{\sum_k I_k}, \qquad y_c = \frac{\sum_k I_k y_k}{\sum_k I_k}$$

**Pros**: Linear, fast, O(N) per sub-aperture  
**Cons**: Biased by background pedestal; noise amplified for faint spots

#### Method 2: Thresholded CoG (T-CoG) ← **Recommended for production**

Apply a threshold $T = \mu_B + k\sigma_B$ before centroiding, where $\mu_B$, $\sigma_B$ are estimated from the sub-aperture border pixels:

$$I_k^* = \max(0, I_k - T), \qquad x_c = \frac{\sum_k I_k^* x_k}{\sum_k I_k^*}$$

**Pros**: Suppresses background bias; robust to read noise  
**Cons**: Can bias toward bright pixels if threshold too aggressive  
**Typical $k$**: 2–4 (2–4σ above background)

#### Method 3: Windowed CoG (W-CoG)

Multiply intensities by a Gaussian window centered on initial estimate:

$$w_k = \exp\left(-\frac{(x_k - x_0)^2 + (y_k - y_0)^2}{2\sigma_w^2}\right)$$
$$x_c = \frac{\sum_k w_k I_k x_k}{\sum_k w_k I_k}$$

Requires one iteration to obtain $x_0$ (use CoG result). Sets $\sigma_w \approx$ spot FWHM.  
**Pros**: Lower noise sensitivity, suppresses cross-talk from adjacent spots  
**Cons**: Slightly more expensive; requires initial estimate

#### Method 4: Matched Filter / Correlation

Cross-correlate sub-aperture with reference PSF template:
$$C(\Delta x, \Delta y) = \sum_{k} I_k \cdot P(x_k - \Delta x, y_k - \Delta y)$$

Peak of $C$ gives sub-pixel displacement. Best for faint sources but expensive (FFT-based: $O(N^2 \log N)$).

### 3.4 C Implementation Strategy

The C centroiding operates on a flat `float*` buffer representing the full detector frame. Key design decisions:

- **Cache locality**: For each sub-aperture, pixel access strides through rows — addressed by processing row-by-row within the sub-aperture window
- **SIMD opportunity**: The accumulation loop in CoG (`sum_Ix`, `sum_Iy`, `sum_I`) can be auto-vectorized with `-O3 -march=native`
- **OpenMP**: Each sub-aperture is independent → embarrassingly parallel via `#pragma omp parallel for`
- **No heap allocation in hot path**: All results written to caller-provided arrays

```c
// Hot path — thresholded CoG for one sub-aperture (pseudocode):
for (int row = y0; row < y0+h; row++) {
    const float* rowptr = image + row * img_width + x0;
    for (int col = 0; col < w; col++) {
        float val = rowptr[col] - threshold;
        if (val < 0) val = 0;
        sum_I  += val;
        sum_Ix += val * (x0 + col);
        sum_Iy += val * (y0 + row);
    }
}
cx = (sum_I > 0) ? sum_Ix / sum_I : x_center;
cy = (sum_I > 0) ? sum_Iy / sum_I : y_center;
```

**Compile flags**: `gcc -O3 -march=native -fopenmp -ffast-math`

### 3.5 Reference Centroids

Three strategies (selectable via config):

| Strategy | Description | When to Use |
|---|---|---|
| `geometric` | Center of each sub-aperture in pixels | Always available; poor for optical aberrations |
| `flat_frame` | Average centroids from a flat wavefront exposure | Best if calibration data exists |
| `time_average` | Mean centroid over all frames in sequence | Good for long sequences; bakes in mean aberration |

The reference defines the zero-slope reference: $s_x = (c_x - c_{x,\text{ref}})\cdot p/f$

---

## 4. Stage 2 — Slope Computation & Wavefront Reconstruction

### 4.1 Slope Computation

From centroids to angular slopes:

$$s^x_{ij} = \frac{(c^x_{ij} - c^x_{\text{ref},ij}) \cdot p}{f} \quad [\text{rad}]$$

The slope vector $\mathbf{s} \in \mathbb{R}^{2N_{SA}}$ is assembled by interleaving $x$ and $y$ slopes for valid sub-apertures.

### 4.2 Modal Reconstruction (Zernike)

Express the wavefront as a sum of Zernike polynomials $Z_j(\rho, \theta)$:

$$W(\rho, \theta) = \sum_{j=2}^{J} a_j Z_j(\rho, \theta)$$

(Tip $j=2$, Tilt $j=3$, ... Spherical, etc. — piston $j=1$ not measured by SH-WFS)

The interaction matrix $\mathbf{D} \in \mathbb{R}^{2N_{SA} \times J}$ maps Zernike coefficients to slopes:

$$D_{ij}^x = \frac{\partial Z_j}{\partial x}\bigg|_{SA_i}, \qquad D_{ij}^y = \frac{\partial Z_j}{\partial y}\bigg|_{SA_i}$$

Reconstruction via least squares:
$$\hat{\mathbf{a}} = \mathbf{D}^+ \mathbf{s} = (\mathbf{D}^T\mathbf{D})^{-1}\mathbf{D}^T\mathbf{s}$$

$\mathbf{D}^+$ is the **reconstructor matrix** — precomputed once during calibration.

**Practical note**: Use SVD to compute $\mathbf{D}^+$, filtering low-energy modes (truncated SVD) to avoid noise amplification.

### 4.3 Zonal Reconstruction (Hudgin/Fried)

In the **Fried geometry**, each wavefront node $\phi_{ij}$ sits at a lenslet corner. The measured slopes relate to phase differences:

$$s^x_{ij} \approx \frac{1}{d}\left[\frac{(\phi_{i,j+1} + \phi_{i+1,j+1}) - (\phi_{i,j} + \phi_{i+1,j})}{2}\right]$$

This linear system $\mathbf{A}\mathbf{\Phi} = \mathbf{s}$ is underdetermined (piston null space). Solve via:

$$\hat{\mathbf{\Phi}} = \mathbf{A}^+ \mathbf{s}$$

where $\mathbf{A}^+$ is the **zone reconstructor** (precomputed, sparse structure allows fast solve).

**Comparison**:

| | Modal (Zernike) | Zonal (Fried) |
|---|---|---|
| Output | Smooth modal coefficients | Per-node phase values |
| Edge handling | Extrapolates naturally | Restricted to measured nodes |
| Computational cost | O(J·N_SA) per frame | O(N_SA) iterative solve |
| Noise filtering | Truncate modes | Tikhonov regularization |
| **Recommendation** | Best for turbulence stats | Best for DM control |

### 4.4 Recommended Hybrid Approach

1. **Zernike modal** reconstruction → Zernike coefficients $a_j$  
   → Used for turbulence characterization (Noll variance analysis)
2. **Reconstruct phase map** from Zernike sum on actuator grid  
   → Used as input to DM control

---

## 5. Stage 3 — Turbulence Characterization

### 5.1 Fried Parameter ($r_0$) Estimation

**Method 1: Zernike Variance**  
The theoretical variance of Zernike mode $j$ for Kolmogorov turbulence is (Noll 1976):

$$\langle a_j^2 \rangle = c_j \left(\frac{D}{r_0}\right)^{5/3}$$

where $c_j$ are tabulated Noll coefficients. Fit observed variances against $D/r_0$:

$$r_0 = D \cdot \left(\frac{c_j}{\sigma_j^2}\right)^{3/5}$$

Use modes $j = 4\ldots20$ (exclude tip/tilt which saturate; exclude piston).

**Method 2: Slope Structure Function**  
Compute the spatial structure function of slope measurements across sub-apertures:

$$D_s(r_{ij}) = \langle |s_{i} - s_j|^2 \rangle \propto r_{ij}^{5/3}$$

Fit power law to extract $r_0$.

### 5.2 Coherence Time ($\tau_0$) Estimation

**Method: Temporal structure function of tip/tilt or selected Zernike mode**:

$$D_\phi(\tau) = \langle |\phi(t+\tau) - \phi(t)|^2 \rangle$$

For Kolmogorov: $D_\phi(\tau) = 6.88(v\tau/r_0)^{5/3}$

Fit the slope in log-log space. The coherence time:
$$\tau_0 = 0.314 r_0 / v_\text{eff}$$

where $v_\text{eff}$ is the effective wind speed from the fit.

**Method 2: Autocorrelation of slope vector**  
$\rho(\tau) = \langle \mathbf{s}(t) \cdot \mathbf{s}(t+\tau) \rangle / \langle |\mathbf{s}|^2 \rangle$

$\tau_0$ is the lag at which $\rho$ drops below $1/e$.

---

## 6. Stage 4 — Actuator Map & DM Control

### 6.1 Influence Function Matrix

Each DM actuator $k$ produces a displacement $F_k(\mathbf{x})$ on the mirror surface (the **influence function**). Typically modeled as a Gaussian:

$$F_k(\mathbf{x}) = \exp\left(-\frac{|\mathbf{x} - \mathbf{x}_k|^2}{2\sigma_k^2}\right)$$

with coupling parameter $\omega = F_k(\mathbf{x}_{k+1})/F_k(\mathbf{x}_k)$ (typically $\omega \approx 0.1$–$0.3$).

The mirror surface for actuator command $\mathbf{u}$:
$$M(\mathbf{x}) = \sum_k u_k F_k(\mathbf{x})$$

### 6.2 Actuator Map Derivation

The AO loop applies the **conjugate** of the reconstructed wavefront:
$$M^\text{target}(\mathbf{x}) = -\hat{W}(\mathbf{x})$$

Solve for actuator commands:
$$\mathbf{F} \mathbf{u} = -\hat{\mathbf{W}}$$

$$\hat{\mathbf{u}} = -\mathbf{F}^+ \hat{\mathbf{W}}$$

where $\mathbf{F}$ is the influence function matrix sampled at wavefront nodes.

### 6.3 Inter-actuator Coupling Compensation

Coupling means commanding one actuator also moves neighbors. If coupling matrix $\mathbf{C}$ is known:

$$\mathbf{M} = \mathbf{C} \mathbf{u}, \qquad \hat{\mathbf{u}} = \mathbf{C}^{-1}(-\hat{\mathbf{W}})$$

In practice: use regularized inversion $\hat{\mathbf{u}} = (\mathbf{C}^T\mathbf{C} + \gamma \mathbf{I})^{-1}\mathbf{C}^T(-\hat{\mathbf{W}})$ with small $\gamma$ for stability.

### 6.4 Real-time Control Loop

```
         ┌─────────────────────────────────────┐
         │                                     │
Turbulence → SH-WFS → Centroid → Reconstruct → DM command
                                    (≤ 1ms in C with LAPACK)
```

**Latency budget** (for 100 Hz AO loop, 10ms total):
- Camera exposure + readout: ~2ms
- Centroiding (C, OpenMP): ~0.1ms
- Matrix multiply (BLAS): ~0.2ms
- DM command computation: ~0.1ms
- DM settling: ~1–2ms
- **Margin**: ~6ms

---

## 7. Performance Architecture

### 7.1 Language Strategy

| Layer | Language | Reason |
|---|---|---|
| Frame I/O, orchestration | Python | Flexibility, rapid iteration |
| Centroiding hot path | C (+ OpenMP) | Cache-efficient, vectorizable |
| Matrix reconstruction | C + LAPACK/CBLAS | Optimized BLAS routines |
| Turbulence statistics | Python + NumPy | Offline, not time-critical |
| DM real-time control | C | Latency-critical |

### 7.2 Memory Layout

Frames are stored as `float32` row-major (C order) arrays. Sub-aperture windows are extracted by pointer arithmetic — no copy. For a 1024×1024 frame: 4 MB per frame in float32.

### 7.3 Precomputed Matrices

All expensive matrices are precomputed during calibration and stored as binary `.npy`/`.bin` files:
- Reconstructor matrix $\mathbf{D}^+$ (loaded once)
- Pupil mask (valid sub-aperture indices)
- Influence function matrix $\mathbf{F}^+$
- Reference centroids $c_\text{ref}$

---

## 8. Data Contracts & Interchange Formats

```yaml
# Per-frame centroid output (Stage 1 → Stage 2):
centroid_data:
  frame_index: int
  timestamp_ms: float
  centroids_x: float32[N_SA]  # pixel coordinates
  centroids_y: float32[N_SA]  # pixel coordinates
  flux: float32[N_SA]         # total flux per sub-aperture
  valid_mask: bool[N_SA]      # pupil mask

# Stage 2 output:
wavefront_data:
  frame_index: int
  phase_map: float32[H, W]     # radians or nm
  zernike_coeffs: float64[J]   # starting from j=2 (tip)

# Stage 3 output:
turbulence_params:
  r0_m: float                  # Fried parameter in meters
  tau0_ms: float               # coherence time in ms
  v_wind_ms: float             # effective wind speed m/s
  zernike_variances: float64[J]

# Stage 4 output:
actuator_map:
  frame_index: int
  stroke_map: float32[N_act_x, N_act_y]  # in actuator stroke units
```

---

## 9. Algorithm Selection Rationale

### Why Thresholded CoG over other methods?

- **Speed**: O(N) per sub-aperture vs O(N log N) for correlation
- **Robustness**: Background suppression is the #1 source of centroid bias
- **Implementation**: Trivially vectorizable; no FFT overhead
- **Literature**: Standard in operational AO systems (SPHERE, GPI, HARMONI)

### Why Modal (Zernike) reconstruction for turbulence characterization?

- Zernike modes are **orthogonal over circular apertures** — covariance is diagonal for Kolmogorov
- Noll (1976) provides exact **analytical covariance**: $\langle a_j a_{j'}\rangle$ for K-statistics
- Direct access to $r_0$ via mode variance fitting without spatial structure function computation
- Modes naturally filter measurement noise (truncate at $J_\text{max}$)

### Why C over CUDA/GPU?

- **Latency** matters more than **throughput**: the AO loop processes 1 frame/iteration
- GPU launch overhead (~100µs) kills latency for small frame sizes
- C + OpenMP achieves <1ms for $64 \times 64$ sub-aperture arrays on modern CPUs
- GPU becomes beneficial for very large AO systems (>1000 sub-apertures) or post-processing

---

## 10. Validation Strategy

### Stage 1 (Centroiding)
- **Synthetic test**: Generate spot at known position with Gaussian PSF + Poisson noise
- **Metric**: Centroid error vs SNR curve (should approach Cramer-Rao bound)
- **Cross-validation**: Compare C and Python implementations pixel-for-pixel

### Stage 2 (Reconstruction)
- **Synthetic test**: Inject known Zernike coefficients ($a_4$=defocus, $a_5$=astigmatism)
- **Metric**: RMS reconstruction error $\|a_\text{true} - a_\text{est}\|_2 / \|a_\text{true}\|_2$
- **Expected**: <5% for J=20 modes, 64 sub-apertures, SNR>50

### Stage 3 (Turbulence)
- **Synthetic test**: Generate phase screen with known $r_0$
- **Metric**: Recovered $r_0$ within 10% of input
- **Check**: Zernike variance slope in log-log plot follows 5/3 power law

### Stage 4 (DM)
- **Closed-loop test**: Apply reconstructed commands, simulate corrected wavefront, measure residual
- **Metric**: Strehl ratio improvement (Maréchal approximation: $S \approx e^{-\sigma_\phi^2}$)

---

## 11. Literature & References

1. **Hardy, J.W.** (1998). *Adaptive Optics for Astronomical Telescopes*. Oxford University Press.
2. **Noll, R.J.** (1976). Zernike polynomials and atmospheric turbulence. *JOSA*, 66(3), 207–211.
3. **Fried, D.L.** (1965). Statistics of a geometric representation of wavefront distortion. *JOSA*, 55(11), 1427–1435.
4. **Roddier, F.** (1999). *Adaptive Optics in Astronomy*. Cambridge University Press.
5. **Southwell, W.** (1980). Wavefront estimation from wavefront slope measurements. *JOSA*, 70(8), 998–1006.
6. **Rousset, G.** (1999). Wave-front sensors. In *Adaptive Optics in Astronomy*, Ch. 5.
7. **Rigaut, F., Gendron, E.** (1992). Laser guide star in adaptive optics. *A&A*, 261, 677–694.

---

*End of Architecture Document*
