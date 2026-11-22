# 정규화 기법 (BatchNorm·LayerNorm·Dropout)

!!! spec "핵심 논문 · 연도"
    Ioffe·Szegedy, "Batch Normalization" (2015) · Ba·Kiros·Hinton, "Layer Normalization" (2016) · Srivastava 외, "Dropout" (2014) · Wu·He, "Group Normalization" (2018) · Zhang·Sennrich, "Root Mean Square Layer Normalization" (2019) · Santurkar 외, "How Does Batch Normalization Help Optimization?" (2018)

!!! basic "한눈에 보기"
    이 페이지에는 비슷한 이름의 두 종류가 나옵니다.

    1. **정규화 층 (normalization)** — 층 사이를 흐르는 값들의 크기를 **평균 0, 분산 1 근처로 맞춰 주는** 장치입니다. 시험 점수를 표준 점수로 바꾸는 것과 같습니다. 값이 너무 커지거나 작아지지 않으니 학습이 빠르고 안정적입니다. BatchNorm, LayerNorm, RMSNorm이 있습니다.
    2. **드롭아웃 (dropout)** — 학습 중 뉴런 일부를 **무작위로 꺼 버리는** 기법입니다. 팀원 몇 명이 갑자기 결근해도 일이 돌아가도록 훈련시키는 것과 같아서, 특정 뉴런에 지나치게 의존하는 과적합을 막습니다.

    CNN은 주로 BatchNorm, 트랜스포머는 LayerNorm(최근에는 RMSNorm)을 씁니다.

## 정규화 층: 어느 방향으로 평균을 내나 { .l2 }

입력 텐서 (B, T, d) 또는 (B, C, H, W)에서 **어떤 축으로 평균·분산을 계산하느냐**가 차이입니다.

| 방식 | 평균·분산을 계산하는 범위 | 배치 크기 의존 | 주로 쓰는 곳 |
|---|---|---|---|
| BatchNorm | 같은 채널을 **배치 전체**에 걸쳐 | 있음 (작은 배치에 약함) | CNN |
| LayerNorm | **샘플(토큰) 하나**의 모든 특징에 걸쳐 | 없음 | 트랜스포머, RNN |
| RMSNorm | LayerNorm에서 평균 빼기를 생략, RMS로만 나눔 | 없음 | LLaMA 계열 등 최근 LLM |
| GroupNorm | 샘플 하나의 채널 그룹 안에서 | 없음 | 작은 배치 CNN, 확산 모델 U-Net |

공통 식은 다음과 같습니다(γ, β는 학습되는 크기·이동 파라미터).

\[
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}},\qquad y = \gamma\hat{x} + \beta
\qquad\quad
\text{RMSNorm: } y = \gamma\,\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}
\]

## 왜 트랜스포머는 LayerNorm인가 { .l2 }

| 이유 | 설명 |
|---|---|
| 가변 길이 | 문장마다 길이가 달라 패딩이 섞이면 배치 통계가 왜곡됨 |
| 작은 배치 | 큰 모델은 GPU당 배치가 작아 배치 통계가 불안정 |
| 학습·추론 동작 일치 | BatchNorm은 추론 때 이동 평균 통계를 써서 학습과 다르게 동작. LayerNorm은 항상 같음 |
| 생성 시 토큰 하나씩 | 자기회귀 생성에서는 배치 통계라는 개념이 맞지 않음 |

## 드롭아웃 { .l2 }

학습 중 각 뉴런을 확률 p로 0으로 만들고, 남은 값은 \(1/(1-p)\)배 키워 기댓값을 유지합니다(inverted dropout). 추론 때는 끕니다. `model.train()`과 `model.eval()`을 전환해야 하는 대표적인 이유입니다.

| 상황 | 드롭아웃 |
|---|---|
| 작은 데이터의 MLP·CNN | p = 0.2–0.5, 효과 큼 |
| 원조 트랜스포머 (2017) | p = 0.1 |
| 대형 LLM 사전 학습 | 보통 0 (데이터를 1회만 보므로 과적합이 적고, 오히려 학습 효율이 떨어짐) |
| LLM 미세 조정 | 0–0.1 (LoRA 드롭아웃 등) |

## 코드로 보기 { .l2 }

LayerNorm·RMSNorm을 직접 구현해 PyTorch와 맞춰 보고, 드롭아웃이 train/eval 모드에서 다르게 동작하는 것을 확인합니다.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.randn(2, 4, 8) * 5 + 3              # (B, T, d), 평균 3·표준편차 5 근처

def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdim=True); var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps)

def rms_norm(x, eps=1e-6):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)

ln = nn.LayerNorm(8, elementwise_affine=False)
print(torch.allclose(layer_norm(x), ln(x), atol=1e-5))
print(torch.allclose(rms_norm(x), nn.RMSNorm(8, elementwise_affine=False, eps=1e-6)(x), atol=1e-5))
print("LayerNorm 후 토큰별 평균·표준편차:", layer_norm(x)[0, 0].mean().item().__round__(4),
      layer_norm(x)[0, 0].std(unbiased=False).item().__round__(4))

drop = nn.Dropout(p=0.5)
v = torch.ones(10)
drop.train(); print("train:", drop(v))         # 절반 정도 0, 나머지는 2.0
drop.eval();  print("eval :", drop(v))         # 그대로 1.0
```

??? expert "전문가 노트 — BatchNorm의 진짜 효과, Pre-LN과 RMSNorm, 정규화 없는 학습"
    **BatchNorm은 왜 효과가 있나.** 원 논문은 층 입력 분포가 학습 중 계속 바뀌는 "내부 공변량 이동"을 줄인다고 설명했지만, Santurkar 외(2018)는 분포 이동을 일부러 넣어도 BatchNorm이 여전히 도움이 됨을 보이고, 주된 효과가 손실 지형을 매끄럽게 만들어(기울기의 립시츠 상수 감소) 큰 학습률을 허용하는 데 있다고 주장했습니다.

    **BatchNorm의 함정.** 추론 시에는 학습 중 모은 이동 평균 통계를 쓰므로, 학습과 추론의 데이터 분포가 다르면 성능이 떨어집니다. 분산 학습에서는 GPU마다 통계가 달라 SyncBatchNorm이 필요하고, 미세 조정 시 통계를 고정할지(freeze) 결정해야 합니다.

    **RMSNorm.** 평균 빼기를 생략해도 성능이 거의 같다는 관찰에서 나왔습니다. 연산과 메모리 접근이 줄어 메모리 대역폭이 병목인 GPU에서 이득이 있고, 대부분의 최신 오픈 LLM이 채택했습니다. Pre-LN과 결합하면 잔차 경로의 값 크기가 층을 지나며 점점 커지는 경향이 있어, 일부 모델은 블록 출력에도 정규화를 추가합니다(샌드위치 정규화, QK-norm).

    **정규화 없는 학습.** NFNet(2021)은 적응형 기울기 클리핑과 신중한 초기화로 BatchNorm 없이 최고 수준의 이미지 분류를 달성했고, 트랜스포머에서도 정규화 층을 tanh 형태의 단순 원소별 함수로 대체하는 연구(Dynamic Tanh, 2025)가 나왔습니다. 정규화의 본질은 "값의 크기를 일정 범위로 유지하는 것"이라는 관점입니다.

## 관련 페이지

- [초기화와 학습 안정성](training-stability.md)
- [트랜스포머](../arch/transformer.md) — Pre-LN 구조
- [과적합과 정규화](../ml/overfitting.md) — 정규화(regularization) 전반
- [CNN](../arch/cnn.md) — BatchNorm을 쓰는 ResNet
