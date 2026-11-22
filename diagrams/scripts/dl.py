"""딥러닝 기초 섹션 그림

1. dl_activations : 시그모이드·tanh·ReLU·GELU·SiLU 함수와 도함수
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

import style as s


def gelu(x):
    erf = np.vectorize(math.erf)
    return 0.5 * x * (1 + erf(x / math.sqrt(2)))


def dl_activations() -> None:
    x = np.linspace(-5, 5, 801)
    sig = 1 / (1 + np.exp(-x))
    funcs = {
        "sigmoid": (sig, s.SPECIAL),
        "tanh": (np.tanh(x), s.SYNC),
        "ReLU": (np.maximum(0, x), s.DL),
        "GELU": (gelu(x), s.UL),
        "SiLU (Swish)": (x * sig, s.CTRL),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, (y, c) in funcs.items():
        axes[0].plot(x, y, color=c, lw=2, label=name)
        axes[1].plot(x, np.gradient(y, x), color=c, lw=2, label=name)
    axes[0].set_title("Activation f(x)", fontsize=10)
    axes[1].set_title("Derivative f'(x)", fontsize=10)
    axes[0].set_ylim(-1.5, 3.5)
    axes[1].set_ylim(-0.2, 1.25)
    for ax in axes:
        ax.axhline(0, color=s.GRID, lw=0.8); ax.axvline(0, color=s.GRID, lw=0.8)
        ax.grid(color=s.EMPTY)
        for sp in ax.spines.values():
            sp.set_color(s.GRID)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    s.save(fig, "dl_activations")


if __name__ == "__main__":
    s.setup()
    dl_activations()
