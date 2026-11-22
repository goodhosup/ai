"""머신러닝 섹션 그림

1. ml_fit_degrees : 다항 회귀 차수 1·4·15 — 과소적합 / 적당 / 과적합
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import style as s


def ml_fit_degrees() -> None:
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 1, 20))
    f = lambda t: np.sin(2 * np.pi * t)
    y = f(x) + rng.normal(0, 0.25, x.size)
    grid = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, deg, title in zip(axes, (1, 4, 15), ("Degree 1: underfit", "Degree 4: good fit", "Degree 15: overfit")):
        coef = np.polyfit(x, y, deg)
        ax.plot(grid, f(grid), color=s.MUTED, lw=1.2, ls="--", label="true function")
        ax.plot(grid, np.polyval(coef, grid), color=s.DL, lw=2, label="model")
        ax.scatter(x, y, s=18, color=s.UL, zorder=3, label="training data")
        ax.set_ylim(-1.8, 1.8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(s.GRID)
    axes[0].legend(loc="lower left", fontsize=7.5, frameon=False)
    s.save(fig, "ml_fit_degrees")


if __name__ == "__main__":
    s.setup()
    ml_fit_degrees()
