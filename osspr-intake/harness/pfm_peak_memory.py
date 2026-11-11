"""Peak memory of ``_read_pfm`` on one full-size stereo disparity map.

The file is a little-endian ``Pf`` of shape 1988×2880 float32 (about 22.9 MiB),
the scale of a full-resolution Middlebury 2014 pair, with ``+inf`` unknown
pixels. Rows are stored bottom-first, which is the PFM convention ``_read_pfm``
flips. The file is written before tracing. Prints tracemalloc peak bytes of one
``_read_pfm`` call after an untraced warmup read.
"""

from __future__ import annotations

import gc
import tempfile
import tracemalloc
from pathlib import Path

import numpy as np

from supervision.depth.readers import _read_pfm

_HEIGHT = 1988
_WIDTH = 2880


def _disparity() -> np.ndarray:
    """Build a smooth disparity field with a few unknown pixels."""
    row = np.arange(_WIDTH, dtype=np.float32)
    column = np.arange(_HEIGHT, dtype=np.float32)[:, None]
    disparity = np.float32(40.0) + row * np.float32(0.05) + column * np.float32(0.01)
    disparity = np.ascontiguousarray(disparity, dtype=np.float32)
    disparity[0, :16] = np.inf
    return disparity


def _write_pfm(path: Path, disparity: np.ndarray) -> None:
    """Write ``disparity`` as a little-endian grayscale PFM, bottom row first."""
    header = f"Pf\n{_WIDTH} {_HEIGHT}\n-1.0\n".encode("ascii")
    payload = np.ascontiguousarray(disparity[::-1], dtype="<f4").tobytes()
    path.write_bytes(header + payload)


def peak_bytes(path: Path) -> int:
    """Return tracemalloc peak bytes for one ``_read_pfm`` call."""
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    loaded = _read_pfm(path)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if loaded.shape != (_HEIGHT, _WIDTH) or loaded.dtype != np.float32:
        raise RuntimeError(f"unexpected PFM result {loaded.shape} {loaded.dtype}")
    return int(peak)


def main() -> None:
    """Write a temp PFM, warm up once, then print one measured peak."""
    disparity = _disparity()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "disp0.pfm"
        _write_pfm(path, disparity)
        del disparity
        warmup = _read_pfm(path)
        if warmup.shape != (_HEIGHT, _WIDTH):
            raise RuntimeError(f"warmup read returned shape {warmup.shape}")
        del warmup
        print(peak_bytes(path))


if __name__ == "__main__":
    main()
