# GAN

!!! spec "핵심 논문 · 연도"
    Goodfellow 외, "Generative Adversarial Nets" (2014) · Radford 외, "DCGAN" (2015) · Arjovsky 외, "Wasserstein GAN" (2017) · Karras 외, "Progressive GAN" (2017), "StyleGAN" (2018), "StyleGAN2" (2019) · Isola 외, "pix2pix" (2016) · Zhu 외, "CycleGAN" (2017) · Brock 외, "BigGAN" (2018)

!!! basic "한눈에 보기"
    **GAN(생성적 적대 신경망)**은 **위조지폐범과 경찰의 경쟁**입니다.

    - **생성자(위조범)**: 무작위 잡음에서 가짜 이미지를 만듭니다.
    - **판별자(경찰)**: 진짜 이미지와 가짜 이미지를 구별합니다.

    판별자가 가짜를 잘 잡아내면 생성자는 더 진짜 같은 가짜를 만들도록 배우고, 생성자가 좋아지면 판별자는 더 예리해지도록 배웁니다. 이 경쟁이 계속되면 결국 생성자가 진짜와 구별할 수 없는 이미지를 만들게 됩니다.

    2014–2021년 이미지 생성의 주인공이었고, 실존하지 않는 사람의 얼굴 사진(StyleGAN)으로 유명해졌습니다. 지금은 이미지 생성의 주류가 확산 모델로 넘어갔지만, 빠른 생성이 필요한 곳(초해상도, 실시간 변환)과 확산 모델의 보조 손실로 여전히 쓰입니다.

## 미니맥스 게임 { .l2 }

생성자 G는 잡음 \(\mathbf{z}\)를 이미지 \(G(\mathbf{z})\)로, 판별자 D는 이미지를 "진짜일 확률"로 바꿉니다.

\[
\min_G\max_D\; \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z}\sim p(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]
\]

| 단계 | 하는 일 |
|---|---|
| 판별자 업데이트 | 진짜는 1, 가짜는 0으로 분류하도록 (이진 교차 엔트로피) |
| 생성자 업데이트 | 판별자가 가짜를 1(진짜)로 착각하도록 |

실제로는 학습 초기에 \(\log(1 - D(G(\mathbf{z})))\)의 기울기가 너무 작아서, 생성자는 대신 \(-\log D(G(\mathbf{z}))\)를 최소화합니다(non-saturating 손실). 이론적으로 최적의 판별자에 대해 원래 목적 함수는 데이터 분포와 생성 분포 사이의 젠슨-섀넌 발산을 최소화하는 것과 같습니다.

## 학습이 어려운 이유 { .l2 }

| 문제 | 증상 | 대응 |
|---|---|---|
| 모드 붕괴 | 생성자가 몇 가지 그럴듯한 이미지만 반복 생성 | 미니배치 판별, WGAN, 다양성 손실 |
| 불균형 | 판별자가 너무 강해 생성자 기울기가 사라짐 | 학습 횟수·학습률 조절, 스펙트럼 정규화 |
| 진동·비수렴 | 손실이 줄지 않고 출렁임 | TTUR(서로 다른 학습률), 기울기 벌점 |
| 평가 어려움 | 손실 값이 품질을 말해 주지 않음 | FID, 사람 평가 |

## 주요 발전 { .l2 }

| 모델 | 기여 |
|---|---|
| DCGAN (2015) | 합성곱 구조 가이드라인 (BatchNorm, LeakyReLU, 풀링 대신 스트라이드) |
| WGAN (2017) | 바서슈타인 거리 사용 → 학습 안정, 손실이 품질과 상관 |
| pix2pix / CycleGAN | 이미지-이미지 변환 (스케치 → 사진, 말 → 얼룩말) |
| StyleGAN 1–3 (2018–2021) | 스타일 주입으로 세밀한 제어, 사실적인 얼굴 |
| GAN 기반 초해상도·보코더 | ESRGAN, HiFi-GAN (음성 합성의 파형 생성) |

## 코드로 보기 { .l2 }

1차원 정규 분포 \(\mathcal{N}(4, 1.25^2)\)를 흉내 내는 아주 작은 GAN입니다. 생성자는 처음에 평균 0 근처를 내다가 점점 진짜 분포로 옮겨 갑니다.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
G = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))           # 잡음 8차원 → 숫자 1개
D = nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))    # 숫자 → 진짜일 로짓
optG = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()

for step in range(3000):
    real = 4 + 1.25 * torch.randn(128, 1)
    fake = G(torch.randn(128, 8))
    # 판별자: 진짜 → 1, 가짜 → 0
    lossD = bce(D(real), torch.ones(128, 1)) + bce(D(fake.detach()), torch.zeros(128, 1))
    optD.zero_grad(); lossD.backward(); optD.step()
    # 생성자: 판별자가 가짜를 진짜(1)로 믿게
    lossG = bce(D(fake), torch.ones(128, 1))
    optG.zero_grad(); lossG.backward(); optG.step()

with torch.no_grad():
    s = G(torch.randn(10000, 8))
print(f"생성 분포: 평균 {s.mean():.2f}, 표준편차 {s.std():.2f}  (목표 4.00, 1.25)")
```

`fake.detach()`는 판별자를 학습할 때 생성자로 기울기가 흘러가지 않게 막습니다.

??? expert "전문가 노트 — WGAN과 립시츠 제약, FID, GAN은 왜 밀려났나"
    **바서슈타인 거리.** 두 분포의 지지 집합이 겹치지 않으면 젠슨-섀넌 발산은 상수(log 2)가 되어 기울기를 주지 못합니다. WGAN은 "한 분포를 다른 분포로 옮기는 최소 비용"인 바서슈타인-1 거리를 쓰며, 칸토로비치-루빈슈타인 쌍대성에 따라 판별자(비평가)를 1-립시츠 함수로 제한하면 이 거리를 추정할 수 있습니다. 가중치 클리핑(원 논문), 기울기 벌점(WGAN-GP), 스펙트럼 정규화로 립시츠 제약을 겁니다.

    **FID.** 사전 학습된 Inception 네트워크의 특징 공간에서 진짜·가짜 이미지 특징을 각각 가우시안으로 근사하고 두 가우시안 사이의 프레셰 거리를 잽니다: \(\|\mu_r - \mu_g\|^2 + \mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})\). 낮을수록 좋고, 샘플 수(보통 5만)에 따라 값이 달라 같은 설정끼리만 비교합니다. 확산 모델 평가에도 그대로 쓰입니다.

    **GAN이 밀려난 이유.** 확산 모델은 하나의 안정적인 회귀 손실로 학습되고, 모드 붕괴 없이 데이터 분포 전체를 덮으며, 규모를 키울수록 잘 개선되고, 텍스트 조건 생성에 자연스럽게 맞습니다. 반면 GAN은 생성이 한 번의 순전파로 끝난다는 속도 이점이 있어, 확산 모델을 몇 단계 만에 생성하도록 압축하는 증류 기법(적대적 확산 증류 등)에 판별자 손실이 다시 쓰이고 있습니다.

    **딥페이크.** 사실적인 얼굴·음성 합성은 사기·허위 정보에 악용될 수 있어, 탐지 기술과 콘텐츠 출처 표시(C2PA, 워터마크)가 함께 발전하고 있습니다 → [저작권과 개인정보](../society/copyright-privacy.md).

## 관련 페이지

- [확산 모델](diffusion.md) — 이미지 생성의 현재 주류
- [오토인코더와 VAE](vae.md)
- [CNN](cnn.md) — 생성자·판별자의 기본 구조
