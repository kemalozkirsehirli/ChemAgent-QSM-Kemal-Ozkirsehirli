from __future__ import annotations

import numpy as np


def gaussian_ir_spectrum(frequencies_cm, intensities=None, x_min=400.0, x_max=4000.0, points=1800, width=20.0):
    freqs = np.asarray(frequencies_cm, dtype=float)
    if intensities is None:
        intensities = np.ones_like(freqs)
    intensities = np.asarray(intensities, dtype=float)
    x = np.linspace(float(x_min), float(x_max), int(points))
    y = np.zeros_like(x)
    width = float(width)
    for freq, inten in zip(freqs, intensities):
        if freq <= 0:
            continue
        y += inten * np.exp(-0.5 * ((x - freq) / width) ** 2)
    if y.max() > 0:
        y = y / y.max()
    return x, y
