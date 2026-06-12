#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_trajectory(frames: int, atoms: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    side = int(np.ceil(atoms ** (1 / 3)))
    grid = np.array(np.meshgrid(np.arange(side), np.arange(side), np.arange(side))).reshape(3, -1).T[:atoms]
    coords0 = grid * 1.6 + rng.normal(scale=0.08, size=(atoms, 3))
    coords = np.zeros((frames, atoms, 3), dtype=float)
    coords[0] = coords0
    velocities = rng.normal(scale=0.04, size=(atoms, 3))
    mobile = rng.choice(atoms, size=max(1, atoms // 5), replace=False)
    for t in range(1, frames):
        noise = rng.normal(scale=0.025, size=(atoms, 3))
        noise[mobile] += rng.normal(scale=0.055, size=(len(mobile), 3))
        velocities = 0.92 * velocities + noise
        coords[t] = coords[t - 1] + velocities
    return coords


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("examples/synthetic_glass.npz"))
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    coords = make_trajectory(args.frames, args.atoms, args.seed)
    np.savez(args.out, coordinates=coords)
    print(args.out)


if __name__ == "__main__":
    main()
