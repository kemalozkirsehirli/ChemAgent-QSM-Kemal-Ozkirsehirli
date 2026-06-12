from __future__ import annotations

from pathlib import Path

import numpy as np


def load_trajectory(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        key = "coordinates" if "coordinates" in data else data.files[0]
        arr = data[key]
    elif suffix == ".xyz":
        arr = load_multiframe_xyz(path)
    else:
        raise ValueError(f"Unsupported trajectory format: {path.suffix}")
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("Trajectory coordinates must have shape (frames, atoms, 3).")
    return arr


def load_multiframe_xyz(path: Path) -> np.ndarray:
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        n = int(lines[i].strip())
        i += 2
        coords = []
        for _ in range(n):
            parts = lines[i].split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1
        frames.append(coords)
    return np.asarray(frames, dtype=float)
