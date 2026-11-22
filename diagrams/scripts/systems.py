"""시스템과 하드웨어 섹션 그림

1. sys_roofline : H100 루프라인과 LLM 연산(디코딩/프리필)의 위치
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import style as s

PEAK = 989.0    # TFLOP/s, H100 SXM BF16 dense
BW = 3.35       # TB/s, HBM3


def sys_roofline() -> None:
    ai = np.logspace(-0.5, 4, 400)
    perf = np.minimum(PEAK, ai * BW)
    ridge = PEAK / BW
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.loglog(ai, perf, color=s.DL, lw=2.5)
    ax.axvline(ridge, color=s.GRID, lw=1, ls="--")
    ax.text(ridge * 1.08, 3, f"ridge ≈ {ridge:.0f} FLOP/byte", fontsize=8, color="#777777")
    ax.text(1.2, 60, "memory-bound\n(bandwidth limit)", fontsize=9, color=s.SPECIAL)
    ax.text(900, 1250, "compute-bound", fontsize=9, color=s.UL)
    points = {
        "decode, batch 1": (1.0, s.SPECIAL),
        "decode, batch 32": (31.8, s.CTRL),
        "prefill, 4096 tokens": (2048.0, s.UL),
    }
    for name, (x, c) in points.items():
        y = min(PEAK, x * BW)
        ax.scatter([x], [y], color=c, s=50, zorder=3)
        ax.annotate(name, (x, y), xytext=(8, -14), textcoords="offset points", fontsize=8, color=c)
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)", fontsize=9)
    ax.set_ylabel("Attainable TFLOP/s", fontsize=9)
    ax.set_title("Roofline: H100 SXM (BF16 989 TFLOP/s, HBM 3.35 TB/s)", fontsize=10)
    ax.set_ylim(1, 3000)
    ax.grid(color=s.EMPTY, which="both")
    for sp in ax.spines.values():
        sp.set_color(s.GRID)
    s.save(fig, "sys_roofline")


if __name__ == "__main__":
    s.setup()
    sys_roofline()
