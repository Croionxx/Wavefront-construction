"""
src/python — SH-WFS Pipeline Python Package
"""
from .frame_loader   import FrameLoader, NumpyFrameLoader, CalibrationFrames
from .centroiding    import build_grid, PythonCentroider, CCentroider, CentroidPipeline
from .slope_computer import SlopeComputer
from .visualizer     import (plot_raw_frame, plot_slope_quiver,
                             plot_slope_timeseries, plot_stage1_summary,
                             plot_centroid_benchmark)

__all__ = [
    "FrameLoader", "NumpyFrameLoader", "CalibrationFrames",
    "build_grid", "PythonCentroider", "CCentroider", "CentroidPipeline",
    "SlopeComputer",
    "plot_raw_frame", "plot_slope_quiver", "plot_slope_timeseries",
    "plot_stage1_summary", "plot_centroid_benchmark",
]
