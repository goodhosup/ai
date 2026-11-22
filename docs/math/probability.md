# 확률과 통계

!!! spec "핵심 논문 · 연도"
    Bayes (1763, 베이즈 정리) · Fisher, "On the Mathematical Foundations of Theoretical Statistics" (1922, 최대우도) · Bishop, *Pattern Recognition and Machine Learning* (2006) · Murphy, *Probabilistic Machine Learning* (2022)

!!! basic "한눈에 보기"
    AI 모델은 "정답은 이것이다"라고 단정하지 않고 **"이것일 확률이 70%, 저것일 확률이 20%…"**라고 답합니다. LLM도 다음에 올 토큰 수만 개 각각에 확률을 붙이고, 그중 하나를 뽑아 글을 씁니다.

    - **확률 분포**: 가능한 결과마다 확률을 붙인 표. 합은 1입니다.
    - **최대우도 학습**: "실제로 관찰된 데이터에 가장 높은 확률을 주는" 파라미터를 찾는 것. 딥러닝 학습 대부분이 이것입니다.
    - **베이즈 정리**: 새 증거를 보고 믿음을 고치는 규칙. "양성 판정이 나왔을 때 실제로 병에 걸렸을 확률"을 계산할 때 씁니다.

## 기본 개념 { .l2 }

| 개념 | 식 | 뜻 |
|---|---|---|
| 조건부 확률 | \(p(A \mid B) = \dfrac{p(A, B)}{p(B)}\) | B가 일어났을 때 A의 확률 |
| 독립 | \(p(A, B) = p(A)\,p(B)\) | 하나가 다른 하나에 영향을 주지 않음 |
| 연쇄 법칙 | \(p(x_1,\dots,x_T) = \prod_t p(x_t \mid x_{<t})\) | LLM의 기본 식 (앞 토큰이 주어졌을 때 다음 토큰) |
| 기댓값 | \(\mathbb{E}[X] = \sum_x x\,p(x)\) | 평균적으로 기대되는 값 |
| 분산 | \(\mathrm{Var}[X] = \mathbb{E}[(X-\mathbb{E}X)^2]\) | 흩어진 정도 |

연쇄 법칙 줄을 눈여겨보세요. 문장 전체의 확률을 "다음 토큰 확률"의 곱으로 쪼갤 수 있기 때문에, LLM은 다음 토큰만 맞히도록 학습해도 문장 전체의 분포를 배우게 됩니다.

## 자주 나오는 분포 { .l2 }

| 분포 | 쓰임 |
|---|---|
| 베르누이 | 예/아니오 분류 (스팸인가?) — 시그모이드 출력 |
| 범주형 (categorical) | 여러 클래스 중 하나 (다음 토큰) — softmax 출력 |
| 정규 (가우시안) \(\mathcal{N}(\mu, \sigma^2)\) | 가중치 초기화, 잡음, VAE·확산 모델 |
| 균등 | 무작위 초기화, 샘플링 |

## softmax: 점수를 확률로 { .l2 }

모델의 마지막 층은 각 후보에 아무 실수 점수(**로짓**, logit)를 냅니다. softmax가 이것을 합이 1인 확률로 바꿉니다.

\[
p_i = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}
\]

\(\tau\)는 **온도(temperature)**입니다. τ < 1이면 확률이 1등에게 몰려 답이 결정적이 되고, τ > 1이면 고르게 퍼져 다양한 답이 나옵니다. LLM API의 `temperature` 설정이 바로 이것입니다.

## 최대우도 추정 (MLE) { .l2 }

데이터 \(x_1, \dots, x_N\)이 서로 독립이라면, 파라미터 θ인 모델이 이 데이터를 만들 확률(우도)은 곱 \(\prod_i p_\theta(x_i)\)입니다. 곱은 다루기 불편하고 숫자가 너무 작아지므로 로그를 씌워 합으로 바꾸고, 부호를 뒤집어 "최소화할 손실"로 만듭니다.

\[
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(x_i)
\]

이것이 **음의 로그 우도(NLL)**이며, 분류 문제에서는 [교차 엔트로피](information.md)와 같은 식입니다. 회귀에서 가우시안 잡음을 가정하면 NLL은 평균 제곱 오차(MSE)가 됩니다.

## 베이즈 정리 { .l2 }

\[
p(\theta \mid D) = \frac{p(D \mid \theta)\,p(\theta)}{p(D)}
\qquad
\text{사후} \propto \text{우도} \times \text{사전}
\]

예: 유병률 1%인 병의 검사가 민감도 99%, 위양성률 5%라면 양성일 때 실제 환자일 확률은 \(\frac{0.99\times0.01}{0.99\times0.01 + 0.05\times0.99} \approx 16.7\%\)에 불과합니다. 드문 사건에서는 사전 확률이 결과를 크게 좌우합니다. 머신러닝에서 가중치 감쇠(L2 정규화)는 "가중치가 0 근처일 것"이라는 가우시안 사전을 둔 것과 같습니다 → [과적합과 정규화](../ml/overfitting.md).

## 코드로 보기 { .l2 }

```python
import numpy as np

def softmax(z, tau=1.0):
    z = np.asarray(z, dtype=float) / tau
    z -= z.max()                       # 오버플로 방지 (결과는 같음)
    e = np.exp(z)
    return e / e.sum()

logits = [2.0, 1.0, 0.1]
for tau in (0.5, 1.0, 2.0):
    print(tau, softmax(logits, tau).round(3))

# 샘플링: 확률에 따라 다음 토큰 뽑기
rng = np.random.default_rng(0)
p = softmax(logits)
draws = rng.choice(3, size=10000, p=p)
print(np.bincount(draws) / 10000)      # p와 비슷

# NLL: 정답이 0번일 때
print(round(-np.log(p[0]), 3))
```

온도가 0.5일 때 첫 후보 확률은 약 0.86, 2.0일 때 약 0.50으로 바뀝니다.

??? expert "전문가 노트 — log-sum-exp, MAP, 보정(calibration)"
    **log-sum-exp 트릭.** \(\log\sum_j e^{z_j}\)를 그대로 계산하면 \(z\)가 100만 넘어도 \(e^{100}\)에서 FP32가 넘칩니다. \(m = \max_j z_j\)를 빼서 \(m + \log\sum_j e^{z_j - m}\)로 계산하면 안전합니다. PyTorch의 `F.cross_entropy`는 softmax와 log를 합쳐 이 방식으로 계산하므로, softmax를 먼저 하고 log를 따로 씌우는 것보다 정확합니다.

    **MLE와 MAP.** 사전 분포 \(p(\theta)\)까지 곱해 최대화하면 MAP 추정입니다. 가우시안 사전 \(\mathcal{N}(0, \sigma^2 I)\)는 손실에 \(\frac{1}{2\sigma^2}\|\theta\|^2\)를 더하는 것(L2), 라플라스 사전은 \(\|\theta\|_1\)(L1)에 해당합니다. 완전한 베이즈 방법은 θ의 분포 전체를 유지하지만, 큰 신경망에서는 비용 때문에 거의 쓰지 않습니다.

    **보정.** 확률 0.8이라고 답한 예측이 실제로 80% 맞으면 "잘 보정됐다"고 합니다. 사전 학습만 한 LLM은 비교적 잘 보정되어 있지만, RLHF 같은 선호 학습 뒤에는 과신하는 경향이 보고됩니다. 보정 정도는 ECE(expected calibration error)로 재고, 온도 스케일링으로 사후 보정할 수 있습니다.

    **i.i.d. 가정.** 대부분의 이론은 학습·테스트 데이터가 같은 분포에서 독립적으로 뽑혔다고 가정합니다. 실제 서비스에서는 시간이 지나며 입력 분포가 바뀌는 **분포 이동**이 흔하고, 이것이 실험실 성능과 현장 성능이 다른 주된 이유입니다.

## 관련 페이지

- [정보이론 (엔트로피·교차 엔트로피)](information.md) — NLL과 교차 엔트로피의 관계
- [선형 회귀와 로지스틱 회귀](../ml/linear-models.md) — MLE의 첫 사례
- [오토인코더와 VAE](../arch/vae.md), [확산 모델](../arch/diffusion.md) — 확률 모델로서의 생성
- [프롬프트 설계](../llm/prompting.md) — 온도와 샘플링 설정
