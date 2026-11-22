# 옵티마이저 (SGD·Adam)

!!! spec "핵심 논문 · 연도"
    Robbins·Monro (1951, SGD) · Polyak (1964, 모멘텀) · Nesterov (1983) · Duchi 외, "AdaGrad" (2011) · Hinton, RMSProp (2012, 강의 노트) · Kingma·Ba, "Adam" (2014) · Loshchilov·Hutter, "Decoupled Weight Decay Regularization" (2017, AdamW) · Loshchilov·Hutter, "SGDR" (2016, 코사인 스케줄) · Jordan 외, "Muon" (2024)

!!! basic "한눈에 보기"
    옵티마이저는 "기울기를 보고 파라미터를 **얼마나, 어느 방향으로** 움직일지" 정하는 규칙입니다.

    1. **SGD** — 기울기 반대 방향으로 정해진 보폭만큼. 가장 단순합니다.
    2. **모멘텀** — 언덕을 구르는 공처럼 **관성**을 붙입니다. 같은 방향으로 계속 가면 빨라지고, 좌우로 흔들리는 방향은 상쇄됩니다.
    3. **Adam** — 파라미터마다 **보폭을 따로** 조절합니다. 기울기가 늘 큰 파라미터는 조심스럽게, 작은 파라미터는 과감하게. 대부분의 딥러닝, 특히 트랜스포머의 기본값입니다.

    그리고 보폭(학습률)을 학습 내내 고정하지 않고, 처음에는 천천히 키우고(**워밍업**) 끝으로 갈수록 줄입니다(**감쇠**).

## 업데이트 규칙 비교 { .l2 }

기울기 \(\mathbf{g}_t\), 학습률 \(\eta\):

| 옵티마이저 | 업데이트 | 상태 메모리 (파라미터당) |
|---|---|---|
| SGD | \(\theta \leftarrow \theta - \eta\,\mathbf{g}\) | 0 |
| 모멘텀 | \(\mathbf{v} \leftarrow \beta\mathbf{v} + \mathbf{g}\); \(\theta \leftarrow \theta - \eta\mathbf{v}\) | 1개 |
| RMSProp | \(\mathbf{s} \leftarrow \beta\mathbf{s} + (1-\beta)\mathbf{g}^2\); \(\theta \leftarrow \theta - \eta\,\mathbf{g}/\sqrt{\mathbf{s}+\epsilon}\) | 1개 |
| Adam | 아래 식 | 2개 (m, v) |

**Adam**은 모멘텀(1차 모멘트 m)과 RMSProp(2차 모멘트 v)을 합칩니다.

\[
\mathbf{m} \leftarrow \beta_1\mathbf{m} + (1-\beta_1)\mathbf{g},\quad
\mathbf{v} \leftarrow \beta_2\mathbf{v} + (1-\beta_2)\mathbf{g}^2
\]

\[
\hat{\mathbf{m}} = \frac{\mathbf{m}}{1-\beta_1^t},\quad \hat{\mathbf{v}} = \frac{\mathbf{v}}{1-\beta_2^t},\quad
\theta \leftarrow \theta - \eta\,\frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\epsilon}
\]

\(\hat{}\)는 0에서 시작한 이동 평균의 초기 편향을 보정합니다. 기본값은 \(\beta_1 = 0.9\), \(\beta_2 = 0.999\) (LLM 학습은 0.95를 많이 씀), \(\epsilon = 10^{-8}\).

## AdamW: 가중치 감쇠를 분리 { .l2 }

원래 Adam에 L2 벌점을 손실에 더하면, 벌점의 기울기도 \(\sqrt{\hat{\mathbf{v}}}\)로 나뉘어 파라미터마다 감쇠 강도가 달라지는 문제가 있습니다. **AdamW**는 감쇠를 업데이트에서 따로 적용합니다.

\[
\theta \leftarrow \theta - \eta\left(\frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\epsilon} + \lambda\theta\right)
\]

트랜스포머 학습의 사실상 표준이며, λ는 0.01–0.1, 편향과 LayerNorm 파라미터에는 감쇠를 걸지 않는 것이 관례입니다.

## 학습률 스케줄 { .l2 }

| 단계 | 하는 일 | 이유 |
|---|---|---|
| 워밍업 | 처음 수백–수천 스텝 동안 0에서 최대값까지 선형 증가 | 초기에 Adam의 v 추정이 부정확하고 기울기가 커서 발산하기 쉬움 |
| 감쇠 | 코사인 곡선으로 최대값의 약 10%까지 감소 | 끝으로 갈수록 작은 보폭으로 골짜기 바닥에 정착 |
| WSD | 워밍업 → 일정 유지(Stable) → 짧게 감쇠(Decay) | 학습 길이를 미리 정하지 않아도 되고 중간 체크포인트 재사용 쉬움 |

## 무엇을 고를까 { .l2 }

| 상황 | 추천 |
|---|---|
| 트랜스포머, LLM 사전 학습·미세 조정 | AdamW + 워밍업 + 코사인/WSD |
| CNN 이미지 분류 (충분히 튜닝 가능) | SGD + 모멘텀 0.9 (일반화가 약간 좋다는 보고) |
| 빠른 실험, 잘 모르겠을 때 | AdamW, lr 3e-4 |
| 메모리가 빠듯할 때 | 8비트 Adam, Adafactor, LoRA로 학습 파라미터 축소 |

## 코드로 보기 { .l2 }

같은 작은 회귀 문제를 SGD, 모멘텀, AdamW로 풀어 손실 감소 속도를 비교하고, 워밍업 + 코사인 스케줄을 만듭니다.

```python
import math
import torch
import torch.nn as nn

def train(make_opt, steps=300):
    torch.manual_seed(0)
    X = torch.randn(512, 20); w_true = torch.randn(20, 1)
    y = X @ w_true + 0.1 * torch.randn(512, 1)
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = make_opt(model.parameters())
    for _ in range(steps):
        loss = ((model(X) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

print(f"SGD       : {train(lambda p: torch.optim.SGD(p, lr=0.01)):.4f}")
print(f"Momentum  : {train(lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9)):.4f}")
print(f"AdamW     : {train(lambda p: torch.optim.AdamW(p, lr=3e-3, weight_decay=0.01)):.4f}")

# 워밍업 100스텝 + 코사인 감쇠 (최대 lr의 10%까지)
def lr_lambda(step, warmup=100, total=1000, floor=0.1):
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / (total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
print([round(lr_lambda(s), 2) for s in (0, 50, 99, 300, 550, 999)])
```

같은 300스텝에서 SGD는 손실 약 0.05에 머물고, 모멘텀과 AdamW는 약 0.007로 7배 이상 낮습니다. 이 작은 문제에서는 잘 맞춘 모멘텀이 AdamW와 비슷하지만, 층마다 기울기 크기가 크게 다른 트랜스포머에서는 Adam 계열이 훨씬 안정적입니다. 스케줄 배율은 0.01 → 0.51 → 1.0 → 0.89 → 0.55 → 0.1로 변합니다. `torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)`에 넣어 씁니다.

??? expert "전문가 노트 — Adam의 메모리, β₂와 손실 급등, 학습률과 배치, 새 옵티마이저"
    **메모리.** Adam은 파라미터마다 m, v를 FP32로 두므로 8바이트/파라미터, 혼합 정밀도의 FP32 마스터 가중치까지 더하면 옵티마이저 관련만 12바이트입니다. 70B 모델이면 840 GB라 ZeRO처럼 여러 GPU에 나눠 저장합니다 → [분산 학습](../systems/distributed.md).

    **β₂와 손실 급등.** β₂ = 0.999는 v가 약 1,000스텝의 평균이라, 드물게 큰 기울기가 오면 v가 늦게 따라가 업데이트가 과하게 커질 수 있습니다. 대형 LLM이 β₂ = 0.95를 쓰는 이유이며, ε을 키우거나 기울기 클리핑(노름 1.0)을 함께 씁니다.

    **학습률과 배치.** SGD는 배치를 k배 키우면 학습률도 k배(선형 규칙), Adam은 대략 \(\sqrt{k}\)배가 경험적 출발점입니다. 모델 폭을 바꿀 때 최적 학습률이 바뀌는 문제를 μP(maximal update parametrization)가 해결해, 작은 모델에서 찾은 하이퍼파라미터를 큰 모델로 옮길 수 있게 합니다.

    **새 옵티마이저.** Lion(2023)은 부호(sign)만 써서 메모리를 절반으로 줄이고, Shampoo·SOAP은 행렬 단위 전처리로 수렴을 빠르게 하며, Muon(2024)은 2차원 가중치의 업데이트를 직교화(뉴턴-슐츠 반복)해 같은 토큰 수에서 AdamW보다 낮은 손실을 보고했습니다. 2026년 11월 기준으로도 대규모 사전 학습의 기본은 AdamW이고, Muon 계열이 일부 대형 모델에 채택되는 추세입니다.

## 관련 페이지

- [미분과 경사하강법](../math/calculus.md)
- [최적화의 기초](../math/optimization.md) — 조건수와 모멘텀
- [초기화와 학습 안정성](training-stability.md) — 기울기 클리핑, 손실 급등
