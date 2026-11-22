# 정보이론 (엔트로피·교차 엔트로피)

!!! spec "핵심 논문 · 연도"
    Shannon, "A Mathematical Theory of Communication" (1948) · Kullback·Leibler, "On Information and Sufficiency" (1951) · Cover·Thomas, *Elements of Information Theory* (교과서) · Delétang 외, "Language Modeling Is Compression" (2023)

!!! basic "한눈에 보기"
    정보이론은 **"놀라움"을 숫자로 재는 법**입니다. 매일 뜨는 해가 내일도 뜬다는 소식은 정보가 거의 없고, 한여름에 눈이 왔다는 소식은 정보가 많습니다. 드문 일일수록 정보량이 큽니다.

    - **엔트로피**: 평균적으로 얼마나 놀라운가 = 결과를 맞히기 얼마나 어려운가.
    - **교차 엔트로피**: 실제 세상이 분포 p를 따르는데 내가 q라고 믿을 때 겪는 평균 놀라움. **LLM 학습의 손실 함수**가 바로 이것입니다.
    - **perplexity**: 교차 엔트로피를 "평균 몇 개 후보 중에서 헷갈리고 있나"로 바꾼 숫자. 작을수록 좋습니다.

    언어 모델이 다음 토큰을 잘 맞힌다는 것은, 그 텍스트를 **적은 비트로 압축**할 수 있다는 것과 같은 말입니다.

## 정보량과 엔트로피 { .l2 }

확률 p인 사건의 정보량(놀라움)은

\[
I(x) = -\log_2 p(x) \quad [\text{비트}]
\]

입니다. 동전 앞면(p = 1/2)은 1비트, 주사위 6(p = 1/6)은 약 2.58비트입니다. 분포 전체의 평균 정보량이 **엔트로피**입니다.

\[
H(p) = -\sum_x p(x)\log p(x)
\]

| 분포 | 엔트로피 |
|---|---|
| 항상 같은 결과 (p = 1) | 0 비트 (놀랄 일이 없음) |
| 공정한 동전 | 1 비트 |
| 공정한 주사위 | log₂6 ≈ 2.58 비트 |
| 후보 V개 균등 | log₂V 비트 (최대) |

## 교차 엔트로피와 KL 발산 { .l2 }

실제 분포 p를 모델 분포 q로 근사할 때:

\[
H(p, q) = -\sum_x p(x)\log q(x)
\qquad
D_{\mathrm{KL}}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p, q) - H(p)
\]

- \(D_{\mathrm{KL}} \ge 0\)이고, p = q일 때만 0입니다. 두 분포의 "거리"처럼 쓰지만 대칭이 아닙니다.
- \(H(p)\)는 데이터가 정하는 상수이므로 **교차 엔트로피를 줄이는 것 = KL을 줄이는 것 = 최대우도**입니다.

분류에서 정답이 하나(원-핫)이면 p는 정답 칸만 1이므로 교차 엔트로피는 \(-\log q(\text{정답})\), 즉 [확률과 통계](probability.md)의 NLL과 같아집니다.

## 언어 모델의 손실과 perplexity { .l2 }

LLM은 토큰마다 정답 토큰에 붙인 확률의 음의 로그를 평균합니다.

\[
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log q_\theta(x_t \mid x_{<t})
\qquad
\mathrm{PPL} = e^{\mathcal{L}}
\]

| 손실 (nats/토큰) | perplexity | 해석 |
|---|---|---|
| ln 50,000 ≈ 10.8 | 50,000 | 어휘 5만 개 중 무작위로 찍는 수준 (학습 전) |
| 3.0 | 약 20 | 평균 20개 후보 사이에서 헷갈림 |
| 2.0 | 약 7.4 | 잘 학습된 중형 모델의 일반 텍스트 수준 |

손실은 토크나이저가 다르면 직접 비교할 수 없습니다. 토큰이 길면 토큰당 손실이 커지기 때문입니다. 그래서 모델 간 비교에는 **바이트당 비트(bits per byte)**를 쓰기도 합니다.

## 코드로 보기 { .l2 }

```python
import numpy as np
import torch
import torch.nn.functional as F

def H(p):                            # 엔트로피 (비트)
    p = np.asarray(p, float); p = p[p > 0]
    return -(p * np.log2(p)).sum()

print(round(H([0.5, 0.5]), 3), round(H([1/6] * 6), 3), round(H([0.9, 0.1]), 3))

# 교차 엔트로피 = KL + 엔트로피
p = np.array([0.7, 0.2, 0.1]); q = np.array([0.5, 0.3, 0.2])
ce = -(p * np.log(q)).sum(); kl = (p * np.log(p / q)).sum(); h = -(p * np.log(p)).sum()
print(round(ce, 4), round(kl + h, 4))

# PyTorch 손실 → perplexity
logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 3.0]])
target = torch.tensor([0, 2])
loss = F.cross_entropy(logits, target)   # 자연로그, 평균
print(round(loss.item(), 4), round(loss.exp().item(), 3))
```

??? expert "전문가 노트 — 압축과 예측, KL의 방향, 상호 정보량"
    **예측 = 압축.** 산술 부호화를 쓰면 확률 q로 예측하는 모델로 기호 x를 약 \(-\log_2 q(x)\)비트에 부호화할 수 있습니다. 따라서 언어 모델의 교차 엔트로피(비트/토큰)는 그 모델을 압축기로 썼을 때의 압축률과 같습니다. Delétang 외(2023)는 LLM이 텍스트뿐 아니라 이미지·오디오 바이트도 gzip보다 잘 압축함을 보였습니다. "지능은 압축"이라는 주장의 근거가 이 등가성입니다.

    **KL의 방향.** \(D_{\mathrm{KL}}(p\|q)\)를 줄이면(전방 KL, 최대우도) q는 p가 확률을 가진 모든 곳을 덮으려 해서 **넓게 퍼집니다**(mode-covering). \(D_{\mathrm{KL}}(q\|p)\)를 줄이면(역방향 KL) q는 p의 한 봉우리에 **집중합니다**(mode-seeking). RLHF에서 정책이 기준 모델에서 너무 멀어지지 않도록 거는 KL 벌점은 \(D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}})\) 방향입니다 → [선호 학습 (RLHF·DPO)](../llm/alignment-training.md).

    **상호 정보량.** \(I(X;Y) = H(X) - H(X\mid Y)\)는 Y를 알면 X에 대한 불확실성이 얼마나 주는지입니다. 대조 학습(CLIP 등)의 InfoNCE 손실은 이미지와 텍스트 사이 상호 정보량의 하한을 최대화하는 것으로 해석됩니다 → [멀티모달](../apps/multimodal.md).

    **라벨 스무딩.** 정답 원-핫 대신 정답에 1 − ε, 나머지에 ε을 나눠 준 분포와의 교차 엔트로피를 쓰면 과신을 줄이고 일반화가 좋아지는 경우가 있습니다(원조 트랜스포머는 ε = 0.1 사용).

## 관련 페이지

- [확률과 통계](probability.md) — 최대우도와 NLL
- [사전 학습과 스케일링 법칙](../llm/pretraining.md) — 손실이 규모에 따라 줄어드는 법칙
- [토큰화](../llm/tokenization.md) — 토크나이저가 손실 비교에 주는 영향
