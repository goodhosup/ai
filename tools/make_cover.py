"""위키독스 책 표지 (1200 × 1600 PNG)를 만든다.

    python tools/make_cover.py            # → docs/assets/cover.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "cover.png"

KO = next((f for f in ("NanumSquare", "NanumBarunGothic", "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei")
           if any(f in x.name for x in font_manager.fontManager.ttflist)), "DejaVu Sans")

BG_TOP = np.array([0x31, 0x1b, 0x92]) / 255     # deep purple
BG_BOT = np.array([0x0d, 0x0a, 0x24]) / 255     # 거의 검정에 가까운 남보라
AMBER = "#ffc107"
VIOLET = "#b39ddb"
WHITE = "#ffffff"
SOFT = "#e3dcf7"


def main() -> None:
    W, H = 12, 16
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    # 배경 그라데이션
    grad = np.linspace(0, 1, 512)[:, None, None]
    img = BG_BOT * grad + BG_TOP * (1 - grad)
    ax.imshow(np.repeat(img, 2, axis=1), extent=[0, W, 0, H], aspect="auto", zorder=0)

    # 신경망: 입력 → 은닉 → 은닉 → 출력
    layers = [5, 7, 7, 3]
    xs = np.linspace(2.3, W - 2.3, len(layers))
    cy, span = 8.4, 4.6
    nodes = [[(x, cy + (i - (n - 1) / 2) * span / 6) for i in range(n)] for x, n in zip(xs, layers)]
    for a, b in zip(nodes[:-1], nodes[1:]):
        for (x0, y0) in a:
            for (x1, y1) in b:
                w = rng.random()
                ax.plot([x0, x1], [y0, y1], color=AMBER if w > 0.86 else VIOLET,
                        lw=2.2 if w > 0.86 else 0.9, alpha=0.9 if w > 0.86 else 0.28, zorder=2)
    for li, layer in enumerate(nodes):
        for (x, y) in layer:
            hot = li == len(nodes) - 1 and y == layer[1][1]
            ax.add_patch(Circle((x, y), 0.42, color=AMBER if hot else "#7e57c2", alpha=0.25, zorder=3))
            ax.add_patch(Circle((x, y), 0.27, facecolor=AMBER if hot else WHITE,
                                edgecolor=AMBER, lw=2, zorder=4))

    # 어텐션 행렬 띠 (인과 마스크: 아래 삼각형만 값)
    n = 24
    x0, x1 = 0.9, W - 0.9
    w = (x1 - x0) / n
    for i in range(3):
        for j in range(n):
            v = rng.random() if j <= (n - 3 + i) else 0.0
            ax.add_patch(Rectangle((x0 + j * w + 0.03, 4.55 - i * 0.36), w - 0.06, 0.3,
                                   color=AMBER if v > 0.8 else "#9575cd", alpha=0.15 + 0.8 * v, zorder=2))

    # 제목
    ax.text(W / 2, 13.75, "AI Handbook", ha="center", va="center", color=WHITE, fontsize=76,
            fontweight="bold", family="DejaVu Sans", zorder=5)
    ax.text(W / 2, 12.6, "Machine Learning  ·  Deep Learning  ·  LLM", ha="center", va="center",
            color=AMBER, fontsize=34, family="DejaVu Sans", zorder=5)
    ax.text(W / 2, 2.95, "AI 핸드북", ha="center", va="center", color=WHITE, fontsize=70,
            fontweight="bold", family=KO, zorder=5)
    ax.text(W / 2, 1.95, "기초부터 전문가까지 · 한글 인공지능 해설", ha="center", va="center",
            color=SOFT, fontsize=30, family=KO, zorder=5)

    # 주제 칩
    chips = ["Math", "ML", "Transformer", "LLM", "Agents", "AI Chips"]
    widths = [0.55 + 0.17 * len(c) for c in chips]
    gap = 0.3
    total = sum(widths) + gap * (len(chips) - 1)
    x = W / 2 - total / 2
    for c, cw in zip(chips, widths):
        ax.add_patch(FancyBboxPatch((x, 0.72), cw, 0.62, boxstyle="round,pad=0,rounding_size=0.31",
                                    facecolor="none", edgecolor=AMBER, lw=2.5, zorder=5))
        ax.text(x + cw / 2, 1.03, c, ha="center", va="center", color=WHITE, fontsize=21,
                family="DejaVu Sans", zorder=6)
        x += cw + gap

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100)
    plt.close(fig)
    print(f"→ {OUT.relative_to(ROOT)} ({KO})")


if __name__ == "__main__":
    main()
