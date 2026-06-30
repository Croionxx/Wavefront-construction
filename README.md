# SH-WFS Wavefront Reconstruction Pipeline

Stage 1 implementation (frame ingestion, centroiding, slope computation) for
the Shack-Hartmann Wavefront Sensor turbulence characterization project.
See `ARCHITECTURE.md` for the full design across all 4 stages.

## Quickstart

```bash
# 1. Install Python dependencies
pip install -r requirements.txt --break-system-packages

# 2. Build the C centroiding library
cd src/c
make
cd ../..

# 3. Run the full Stage 1 pipeline (generates synthetic data + processes it)
python run_stage1.py
```

Outputs land in `outputs/plots/` (diagnostic PNGs + `stage1_results.npz`)
and `outputs/truth.npz` (ground-truth synthetic data for validation).

## Using your own SH-WFS BMP data

Replace the synthetic generator step in `run_stage1.py`:

```python
from frame_loader import FrameLoader

loader = FrameLoader(
    directory="path/to/your/bmp_frames/",
    frame_interval_ms=2.0,   # your camera's frame interval
)
frames, metas = loader.load_all()
```

Then update `config/sensor_config.yaml` with your actual MLA geometry,
pupil location/radius, and camera parameters — everything downstream
(`build_grid`, `CentroidPipeline`, `SlopeComputer`) reads from this config
and works unchanged.

## Repository layout

```
ARCHITECTURE.md         Full 4-stage design document (physics, algorithms, data contracts)
config/sensor_config.yaml   All hardware/simulation parameters
run_stage1.py            Main pipeline runner

src/c/                   Fast centroiding (C + OpenMP)
  centroid.h/.c          CoG, Thresholded-CoG, Windowed-CoG implementations
  Makefile               Builds libcentroid.so

src/python/
  frame_loader.py        BMP ingestion + calibration frames
  centroiding.py         Sub-aperture grid + Python/C centroiding backends
  slope_computer.py      Pixel displacement → angular slope conversion
  visualizer.py          All diagnostic plots

synthetic/
  generate_turbulence.py Kolmogorov/von Kármán phase screens, Taylor frozen
                         flow, SH-WFS detector frame rendering with photon
                         shot noise + read noise
```

## Next stage (designed, not yet implemented)

`src/python/wavefront_reconstructor.py` — Zernike modal reconstruction and
zonal (Fried geometry) reconstruction from the slope vectors saved in
`stage1_results.npz`. See ARCHITECTURE.md §4 for the math and interaction
matrix formulation.
