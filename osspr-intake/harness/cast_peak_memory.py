"""Peak memory of ``_cast_array_like_opencv`` on one 1080p color frame.

The workload is the float32 raster that ``_convert_scale_abs`` and the float32
``addWeighted`` path hand to the saturating cast: shape ``(1080, 1920, 3)``,
values spanning below 0 and above 255 so the integer clip runs. Tracemalloc is
started after the input exists and after one untraced warmup, then one call is
measured. Prints that peak in bytes.
"""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np

from supervision._cv2._common import _cast_array_like_opencv

_HEIGHT = 1080
_WIDTH = 1920
_CHANNELS = 3


def _frame() -> np.ndarray:
    """Build a deterministic 1080p float32 raster with out-of-range samples."""
    values = np.linspace(
        -32.0,
        288.0,
        num=_HEIGHT * _WIDTH * _CHANNELS,
        dtype=np.float32,
    )
    return values.reshape(_HEIGHT, _WIDTH, _CHANNELS)


def peak_bytes(frame: np.ndarray) -> int:
    """Return tracemalloc peak bytes for one cast of ``frame`` to uint8."""
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    cast = _cast_array_like_opencv(frame, np.dtype(np.uint8))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if cast.shape != frame.shape or cast.dtype != np.uint8:
        raise RuntimeError(f"unexpected cast result {cast.shape} {cast.dtype}")
    return int(peak)


def main() -> None:
    """Warm up once, then print the peak of a single measured call."""
    frame = _frame()
    warmup = _cast_array_like_opencv(frame, np.dtype(np.uint8))
    if warmup.shape != frame.shape or warmup.dtype != np.uint8:
        raise RuntimeError(f"warmup cast returned {warmup.shape} {warmup.dtype}")
    del warmup
    print(peak_bytes(frame))


if __name__ == "__main__":
    main()
