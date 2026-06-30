"""
frame_loader.py — SH-WFS Frame Ingestion
==========================================
Loads BMP (and other) image sequences from disk into normalised float32
NumPy arrays suitable for the centroiding pipeline.

Supports:
  - 8-bit and 16-bit grayscale BMPs (via Pillow)
  - Lazy loading (iterate without loading all frames into RAM)
  - Optional bias/dark/flat calibration frame subtraction
  - Metadata extraction (resolution, bit depth, timestamp from filename)
"""

from __future__ import annotations

import os
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


# ── Metadata ────────────────────────────────────────────────────────────────

@dataclass
class FrameMeta:
    """Metadata for a single detector frame."""
    index: int                      # sequential frame index (0-based)
    path: Path                      # absolute path to the BMP file
    timestamp_ms: float             # time since first frame [ms]; from filename or index*dt
    width: int  = 0
    height: int = 0
    bit_depth: int = 8              # 8 or 16
    dtype: str = "uint8"


@dataclass
class SequenceMeta:
    """Metadata for the complete frame sequence."""
    n_frames: int
    frame_interval_ms: float        # nominal dt between frames
    width: int
    height: int
    bit_depth: int
    source_dir: Path
    frame_paths: List[Path] = field(default_factory=list)


# ── Calibration frames ───────────────────────────────────────────────────────

@dataclass
class CalibrationFrames:
    """Optional calibration images applied during loading."""
    bias:  Optional[np.ndarray] = None   # float32 bias/pedestal frame
    dark:  Optional[np.ndarray] = None   # float32 dark current frame
    flat:  Optional[np.ndarray] = None   # float32 flat-field (normalised to 1.0)

    @classmethod
    def from_files(cls,
                   bias_path: Optional[str] = None,
                   dark_path: Optional[str] = None,
                   flat_path: Optional[str] = None) -> "CalibrationFrames":
        """Load calibration frames from disk."""
        def _load(p):
            if p is None:
                return None
            img = Image.open(p).convert("L")
            arr = np.asarray(img, dtype=np.float32)
            return arr

        bias = _load(bias_path)
        dark = _load(dark_path)
        flat = _load(flat_path)

        if flat is not None:
            # Normalise flat to avoid division by zero
            flat_max = flat.max()
            if flat_max > 0:
                flat = flat / flat_max
            else:
                log.warning("Flat field is all zeros — ignoring")
                flat = None

        return cls(bias=bias, dark=dark, flat=flat)


# ── Core loader ──────────────────────────────────────────────────────────────

class FrameLoader:
    """
    Loads a time-series of BMP frames from a directory.

    Parameters
    ----------
    directory : str | Path
        Directory containing the BMP files.
    pattern : str
        Glob pattern to match frame files.  Default ``"*.bmp"``.
    frame_interval_ms : float
        Nominal time between frames in milliseconds.
    calib : CalibrationFrames | None
        Optional calibration frames.
    normalise : bool
        If True, divide pixel values by max ADU so output is in [0, 1].
    max_frames : int | None
        Load at most this many frames (for testing).
    """

    # Regex to extract frame number from filenames like frame_0042.bmp
    _NUM_RE = re.compile(r"(\d+)")

    def __init__(self,
                 directory: str | Path,
                 pattern: str = "*.bmp",
                 frame_interval_ms: float = 2.0,
                 calib: Optional[CalibrationFrames] = None,
                 normalise: bool = True,
                 max_frames: Optional[int] = None):

        self.directory         = Path(directory)
        self.pattern           = pattern
        self.frame_interval_ms = frame_interval_ms
        self.calib             = calib or CalibrationFrames()
        self.normalise         = normalise
        self.max_frames        = max_frames

        self._paths: List[Path] = []
        self._meta:  Optional[SequenceMeta] = None

        self._scan()

    # ── Scanning ────────────────────────────────────────────────────────────

    def _scan(self):
        """Discover and sort frame files."""
        paths = sorted(
            self.directory.glob(self.pattern),
            key=lambda p: self._extract_index(p)
        )
        if self.max_frames is not None:
            paths = paths[:self.max_frames]

        if not paths:
            raise FileNotFoundError(
                f"No files matching '{self.pattern}' found in {self.directory}"
            )

        self._paths = paths

        # Probe first frame for resolution / bit depth
        probe = self._load_raw(paths[0])
        h, w  = probe.shape
        bd    = 16 if probe.max() > 255 else 8

        self._meta = SequenceMeta(
            n_frames          = len(paths),
            frame_interval_ms = self.frame_interval_ms,
            width             = w,
            height            = h,
            bit_depth         = bd,
            source_dir        = self.directory,
            frame_paths       = paths,
        )
        self._max_adu = (2 ** bd) - 1

        log.info(
            "FrameLoader: found %d frames  [%d × %d px, %d-bit] in %s",
            len(paths), w, h, bd, self.directory
        )

    @staticmethod
    def _extract_index(path: Path) -> int:
        """Extract the leading/trailing integer from a filename for sorting."""
        nums = FrameLoader._NUM_RE.findall(path.stem)
        return int(nums[-1]) if nums else 0

    # ── Raw loading ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_raw(path: Path) -> np.ndarray:
        """
        Load image as uint16 regardless of file bit depth.
        Pillow handles 8/16-bit BMP transparently.
        """
        img = Image.open(path)

        # Convert to grayscale if necessary
        if img.mode not in ("L", "I;16", "I"):
            img = img.convert("L")

        arr = np.asarray(img)

        # Pillow returns 32-bit signed ('I') for 16-bit BMPs
        if arr.dtype == np.int32:
            arr = arr.astype(np.uint16)

        return arr

    # ── Calibration ─────────────────────────────────────────────────────────

    def _apply_calibration(self, arr: np.ndarray) -> np.ndarray:
        """Apply bias, dark, flat corrections in-place on a float32 array."""
        if self.calib.bias is not None:
            arr -= self.calib.bias
        if self.calib.dark is not None:
            arr -= self.calib.dark
        if self.calib.flat is not None:
            # Avoid division by zero
            safe_flat = np.where(self.calib.flat > 0, self.calib.flat, 1.0)
            arr /= safe_flat
        # Clip negatives introduced by calibration
        np.clip(arr, 0.0, None, out=arr)
        return arr

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def meta(self) -> SequenceMeta:
        return self._meta

    @property
    def n_frames(self) -> int:
        return self._meta.n_frames

    def load_frame(self, index: int) -> Tuple[np.ndarray, FrameMeta]:
        """
        Load a single frame by index.

        Returns
        -------
        frame : np.ndarray, shape (H, W), dtype float32
            Calibrated, optionally normalised frame.
        meta : FrameMeta
        """
        path = self._paths[index]
        raw  = self._load_raw(path).astype(np.float32)
        raw  = self._apply_calibration(raw)

        if self.normalise:
            raw /= float(self._max_adu)

        meta = FrameMeta(
            index        = index,
            path         = path,
            timestamp_ms = index * self.frame_interval_ms,
            width        = self._meta.width,
            height       = self._meta.height,
            bit_depth    = self._meta.bit_depth,
        )
        return raw, meta

    def iter_frames(self) -> Iterator[Tuple[np.ndarray, FrameMeta]]:
        """Lazily iterate over all frames in sequence order."""
        for i in range(self.n_frames):
            yield self.load_frame(i)

    def load_all(self) -> Tuple[np.ndarray, List[FrameMeta]]:
        """
        Load entire sequence into memory.

        Returns
        -------
        stack : np.ndarray, shape (N, H, W), dtype float32
        metas : list of FrameMeta
        """
        log.info("Loading all %d frames into memory...", self.n_frames)
        stack = np.empty(
            (self.n_frames, self._meta.height, self._meta.width),
            dtype=np.float32
        )
        metas = []
        for i, (frame, meta) in enumerate(self.iter_frames()):
            stack[i] = frame
            metas.append(meta)
        log.info("Loaded stack shape: %s  (%.1f MB)",
                 stack.shape, stack.nbytes / 1e6)
        return stack, metas

    def compute_reference_frame(self, n_avg: int = 20) -> np.ndarray:
        """
        Compute a time-averaged reference frame from the first n_avg frames.
        Used for time_average reference centroid strategy.
        """
        n = min(n_avg, self.n_frames)
        ref = np.zeros((self._meta.height, self._meta.width), dtype=np.float64)
        for i in range(n):
            frame, _ = self.load_frame(i)
            ref += frame
        return (ref / n).astype(np.float32)

    # ── Convenience ─────────────────────────────────────────────────────────

    def summary(self) -> str:
        m = self._meta
        return (
            f"FrameLoader Summary\n"
            f"  Directory  : {m.source_dir}\n"
            f"  Frames     : {m.n_frames}\n"
            f"  Resolution : {m.width} × {m.height} px\n"
            f"  Bit depth  : {m.bit_depth}\n"
            f"  Frame dt   : {m.frame_interval_ms:.2f} ms\n"
            f"  Total time : {m.n_frames * m.frame_interval_ms:.1f} ms\n"
        )


# ── Synthetic frame source (for testing without real data) ───────────────────

class NumpyFrameLoader:
    """
    Drop-in replacement for FrameLoader that wraps a pre-loaded numpy stack.
    Useful for synthetic data and unit tests — same interface as FrameLoader.
    """

    def __init__(self,
                 stack: np.ndarray,
                 frame_interval_ms: float = 2.0):
        """
        Parameters
        ----------
        stack : np.ndarray, shape (N, H, W), dtype float32
        """
        assert stack.ndim == 3, "Stack must be (N, H, W)"
        self._stack           = stack.astype(np.float32)
        self.frame_interval_ms= frame_interval_ms
        n, h, w               = stack.shape

        self._meta = SequenceMeta(
            n_frames          = n,
            frame_interval_ms = frame_interval_ms,
            width             = w,
            height            = h,
            bit_depth         = 16,
            source_dir        = Path("."),
            frame_paths       = [],
        )

    @property
    def meta(self) -> SequenceMeta:
        return self._meta

    @property
    def n_frames(self) -> int:
        return self._meta.n_frames

    def load_frame(self, index: int) -> Tuple[np.ndarray, FrameMeta]:
        frame = self._stack[index]
        meta  = FrameMeta(
            index        = index,
            path         = Path(f"synthetic_{index:04d}.bmp"),
            timestamp_ms = index * self.frame_interval_ms,
            width        = self._meta.width,
            height       = self._meta.height,
        )
        return frame, meta

    def iter_frames(self) -> Iterator[Tuple[np.ndarray, FrameMeta]]:
        for i in range(self.n_frames):
            yield self.load_frame(i)

    def load_all(self) -> Tuple[np.ndarray, List[FrameMeta]]:
        metas = [self.load_frame(i)[1] for i in range(self.n_frames)]
        return self._stack.copy(), metas
