# 확산 모델

!!! spec "핵심 논문 · 연도"
    Sohl-Dickstein 외, "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (2015) · Song·Ermon, "Score-based generative modeling" (2019) · Ho·Jain·Abbeel, "DDPM" (2020) · Song 외, "DDIM" (2020), "Score SDE" (2021) · Ho·Salimans, "Classifier-Free Guidance" (2021) · Rombach 외, "Latent Diffusion" (2022) · Peebles·Xie, "DiT" (2023) · Lipman 외, "Flow Matching" (2022) · Song 외, "Consistency Models" (2023)

!!! basic "한눈에 보기"
    잉크 한 방울을 물에 떨어뜨리면 점점 퍼져서 결국 균일한 탁한 물이 됩니다. 이것이 **확산**입니다. 그 과정을 **거꾸로 돌리는 법**을 배울 수 있다면, 탁한 물에서 잉크 방울 모양을 되살릴 수 있을 것입니다.

    **확산 모델**은 이미지에 이렇게 합니다.

    1. **망가뜨리기(정방향)**: 진짜 사진에 잡음을 조금씩, 수백–수천 번 더해 완전한 잡음으로 만듭니다. 이건 학습이 필요 없는 정해진 과정입니다.
    2. **되살리기(역방향)**: 신경망이 "이 잡음 섞인 사진에서 **잡음이 어떤 모양인지**"를 맞히도록 배웁니다.
    3. **생성**: 완전한 잡음에서 출발해 신경망이 예측한 잡음을 조금씩 빼 나가면, 세상에 없던 새 사진이 나타납니다.

    Stable Diffusion, Midjourney, DALL·E 3, 영상 생성 모델(Sora 등) 대부분이 이 원리입니다. 글로 조건을 주면("해변의 강아지") 그 내용에 맞게 잡음을 걷어 냅니다.

## 정방향 과정 { .l2 }

원본 \(\mathbf{x}_0\)에 시점 t까지 잡음을 더한 \(\mathbf{x}_t\)는 한 번에 쓸 수 있습니다.

\[
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(0, I)
\]

\(\bar{\alpha}_t\)는 1에서 시작해 0 근처로 줄어드는 **잡음 스케줄**입니다. t가 작으면 원본이 거의 그대로, t = T면 거의 순수한 잡음입니다.

## 학습: 잡음 맞히기 { .l2 }

1. 학습 이미지 \(\mathbf{x}_0\), 무작위 시점 t, 무작위 잡음 \(\boldsymbol{\epsilon}\)을 고릅니다.
2. 위 식으로 \(\mathbf{x}_t\)를 만듭니다.
3. 신경망 \(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\)가 넣었던 잡음 \(\boldsymbol{\epsilon}\)을 맞히도록 MSE를 줄입니다.

\[
\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, t, \boldsymbol{\epsilon}}\big\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\big\|^2
\]

GAN과 달리 **회귀 손실 하나**라 학습이 안정적입니다.

## 생성: 조금씩 잡음 빼기 { .l2 }

순수 잡음 \(\mathbf{x}_T\)에서 시작해 t = T, …, 1에 대해 예측한 잡음을 일부 빼고(DDPM은 약간의 새 잡음을 다시 더함) \(\mathbf{x}_{t-1}\)을 만듭니다. 원 논문(DDPM)은 1,000단계가 필요했지만 지금은 DDIM·DPM-Solver 같은 샘플러로 20–50단계, 증류 기법으로 1–4단계까지 줄였습니다.

## 조건부 생성과 가이던스 { .l2 }

텍스트 조건 c를 신경망에 넣어(교차 어텐션) \(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c)\)를 학습합니다. **분류기 없는 가이던스(CFG)**는 조건 있는 예측과 없는 예측의 차이를 키워 조건을 더 강하게 따르게 합니다.

\[
\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w\,\big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)\big)
\]

w(보통 3–8)가 크면 프롬프트에 충실하지만 다양성이 줄고 색이 과포화됩니다.

## 구조의 변화 { .l2 }

| 요소 | 초기 (2020–2022) | 최근 (2023–) |
|---|---|---|
| 잡음 예측 신경망 | U-Net (CNN + 어텐션) | DiT, MMDiT (트랜스포머) |
| 작업 공간 | 픽셀 | VAE 잠재 공간 (Latent Diffusion) |
| 학습 목표 | 잡음 ε 예측 | 속도 v 예측, 플로 매칭 (직선 경로) |
| 텍스트 인코더 | CLIP | CLIP + T5 등 대형 언어 모델 |
| 대상 | 이미지 | 이미지, 영상, 오디오, 3D, 단백질 구조 |

## 코드로 보기 { .l2 }

2차원 점 데이터(원 위의 점)로 확산 모델의 학습과 생성을 처음부터 구현합니다.

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)
def data(n):                                       # 반지름 1인 원 위의 점
    a = torch.rand(n) * 2 * math.pi
    return torch.stack([a.cos(), a.sin()], 1)

T = 100
beta = torch.linspace(1e-4, 0.05, T)
alpha = 1 - beta; abar = torch.cumprod(alpha, 0)

net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
for step in range(4000):
    x0 = data(256); t = torch.randint(0, T, (256,)); eps = torch.randn_like(x0)
    xt = abar[t, None].sqrt() * x0 + (1 - abar[t, None]).sqrt() * eps
    pred = net(torch.cat([xt, t[:, None] / T], 1))           # 시점도 입력
    loss = ((pred - eps) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# 생성 (DDPM 샘플링)
with torch.no_grad():
    x = torch.randn(2000, 2)
    for t in reversed(range(T)):
        e = net(torch.cat([x, torch.full((2000, 1), t / T)], 1))
        x = (x - beta[t] / (1 - abar[t]).sqrt() * e) / alpha[t].sqrt()
        if t > 0:
            x = x + beta[t].sqrt() * torch.randn_like(x)
r = x.norm(dim=1)
print(f"생성된 점의 반지름: 평균 {r.mean():.3f}, 표준편차 {r.std():.3f}  (목표 1.0, 0)")
```

잡음 2,000개에서 출발해 100단계를 거치면 대부분의 점이 반지름 1 근처의 원 위로 모입니다.

??? expert "전문가 노트 — 점수 함수 관점, 플로 매칭, 빠른 샘플링, 이산 확산"
    **점수 함수.** 잡음 예측은 잡음 섞인 데이터 분포의 로그 밀도 기울기(점수) \(\nabla_{\mathbf{x}}\log p_t(\mathbf{x}) = -\boldsymbol{\epsilon}/\sqrt{1-\bar{\alpha}_t}\)를 추정하는 것과 같습니다(Song 외). 정방향 과정을 확률 미분 방정식(SDE)으로 보면, 역방향 SDE와 같은 분포를 주는 결정적 상미분 방정식(확률 흐름 ODE)이 존재하며, DDIM은 그 ODE의 이산화로 해석됩니다. ODE이므로 고차 수치해법(DPM-Solver 등)으로 단계를 크게 줄일 수 있습니다.

    **플로 매칭과 정류 흐름.** 잡음과 데이터를 직선으로 잇는 경로 \(\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}\)의 속도장 \(\boldsymbol{\epsilon} - \mathbf{x}_0\)을 회귀로 배우는 방식입니다. 경로가 곧을수록 적은 단계로 정확히 적분할 수 있어 Stable Diffusion 3, FLUX 등 최근 모델이 채택했습니다.

    **빠른 샘플링.** 일관성 모델(consistency model)은 ODE 경로의 어느 점에서든 한 번에 끝점으로 가도록 학습하고, 점진적 증류·적대적 증류는 다단계 교사 모델을 1–4단계 학생으로 압축합니다. 온디바이스 이미지 생성이 가능해진 핵심 기술입니다.

    **이산 확산과 언어.** 토큰을 [MASK]로 점점 바꾸는 이산 확산 언어 모델(LLaDA, Mercury 등)이 연구·상용화되고 있습니다. 여러 토큰을 병렬로 생성할 수 있어 자기회귀 LLM보다 빠를 수 있지만, 2026년 11월 기준 품질의 주류는 여전히 자기회귀 모델입니다.

## 관련 페이지

- [오토인코더와 VAE](vae.md) — 잠재 확산의 압축기
- [GAN](gan.md) — 이전 세대의 이미지 생성
- [트랜스포머](transformer.md) — DiT의 기반
- [멀티모달](../apps/multimodal.md) — 이미지·영상 생성 서비스
