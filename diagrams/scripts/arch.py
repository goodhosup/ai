"""신경망 구조 섹션 그림

1. arch_attention_mask : 양방향(인코더) 어텐션 vs 인과(디코더) 어텐션 가중치 행렬 (값은 예시)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import style as s

TOKENS = ["The", "cat", "sat", "on", "the", "mat"]


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def arch_attention_mask() -> None:
    rng = np.random.default_rng(7)
    n = len(TOKENS)
    scores = rng.normal(0, 1.2, (n, n)) + 1.5 * np.eye(n)   # 자기 자신에 조금 더 주목하는 예시
    full = softmax(scores)
    causal = softmax(np.where(np.tril(np.ones((n, n))) > 0, scores, -np.inf))

    cmap = LinearSegmentedColormap.from_list("w", [s.SURFACE, s.DL])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, w, title in ((axes[0], full, "Bidirectional (encoder)"),
                         (axes[1], causal, "Causal (decoder)")):
        ax.imshow(w, cmap=cmap, vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                if title.startswith("Causal") and j > i:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=s.EMPTY,
                                               edgecolor=s.GRID, hatch="///", lw=0.5))
                    continue
                ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if w[i, j] > 0.5 else s.INK)
        ax.set_xticks(range(n), TOKENS)
        ax.set_yticks(range(n), TOKENS)
        ax.set_xlabel("Key (attended token)")
        ax.set_ylabel("Query (current token)")
        ax.set_title(title, fontsize=10, color=s.INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.5, -0.02, "Each row sums to 1 (softmax). Hatched = masked future tokens. Values are illustrative.",
             ha="center", fontsize=8, color=s.MUTED)
    s.save(fig, "arch_attention_mask")


if __name__ == "__main__":
    s.setup()
    arch_attention_mask()
