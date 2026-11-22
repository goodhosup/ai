# 선형 회귀와 로지스틱 회귀

!!! spec "핵심 논문 · 연도"
    Legendre (1805)·Gauss (1809), 최소제곱법 · Galton (1886, "회귀" 용어) · Cox, "The Regression Analysis of Binary Sequences" (1958, 로지스틱 회귀) · Tibshirani, "Lasso" (1996) · Hoerl·Kennard, "Ridge Regression" (1970)

!!! basic "한눈에 보기"
    - **선형 회귀**는 점들 사이를 가장 잘 지나는 **직선**을 긋는 일입니다. "평수가 10평 늘면 집값이 얼마 오르나?"
    - **로지스틱 회귀**는 직선의 출력을 0과 1 사이로 눌러 **확률**로 바꿉니다. "이 메일이 스팸일 확률은?"

    둘 다 "입력마다 가중치를 곱해 더한다"는 같은 뼈대를 갖습니다. 사실 **신경망의 뉴런 하나가 로지스틱 회귀**입니다. 이 페이지를 이해하면 딥러닝의 절반은 이해한 셈입니다.

## 선형 회귀 { .l2 }

입력 특징 \(\mathbf{x} = (x_1, \dots, x_d)\)에 대해

\[
\hat{y} = \mathbf{w}^\top\mathbf{x} + b = w_1x_1 + \dots + w_dx_d + b
\]

가중치 \(\mathbf{w}\)와 절편 b는 **평균 제곱 오차(MSE)**를 최소화하도록 정합니다.

\[
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
\]

MSE는 볼록 함수라 해가 하나이고, 닫힌 식(정규 방정식) \(\mathbf{w} = (X^\top X)^{-1}X^\top\mathbf{y}\)으로 바로 구할 수도 있습니다. 데이터가 크면 경사하강법을 씁니다.

| 가중치 해석 | 뜻 |
|---|---|
| \(w_j > 0\) | 특징 j가 커지면 예측값이 커짐 |
| \(\lvert w_j\rvert\)가 큼 | 영향이 큼 (단, 특징의 단위를 맞췄을 때만 비교 가능) |
| b | 모든 특징이 0일 때의 예측값 |

## 로지스틱 회귀 { .l2 }

선형 결합 \(z = \mathbf{w}^\top\mathbf{x} + b\)를 **시그모이드**에 넣어 확률로 만듭니다.

\[
p(y=1\mid\mathbf{x}) = \sigma(z) = \frac{1}{1+e^{-z}}
\]

손실은 **이진 교차 엔트로피**입니다.

\[
\mathcal{L} = -\frac{1}{N}\sum_i \big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]
\]

결정 경계 \(z = 0\)은 직선(고차원에서는 평면)입니다. 클래스가 K개면 시그모이드 대신 **softmax**를 쓰고(다항 로지스틱 회귀), 이것은 신경망 분류기의 마지막 층과 똑같습니다.

| 항목 | 선형 회귀 | 로지스틱 회귀 |
|---|---|---|
| 출력 | 실수 | 0–1 확률 |
| 출력 함수 | 없음 (항등) | 시그모이드 / softmax |
| 손실 | MSE | 교차 엔트로피 |
| 확률적 해석 | 가우시안 잡음의 최대우도 | 베르누이 분포의 최대우도 |
| 경계 | — | 직선·평면 |

## 특징 공학과 비선형 { .l2 }

선형 모델도 특징을 바꾸면 곡선을 표현할 수 있습니다. x 대신 \((x, x^2, x^3)\)을 넣으면 3차 곡선 회귀가 됩니다. 문제는 **어떤 특징을 만들지 사람이 정해야** 한다는 점이고, 딥러닝은 이 특징 만들기를 여러 층에서 자동으로 합니다 → [퍼셉트론과 MLP](../dl/mlp.md).

## 코드로 보기 { .l2 }

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 선형 회귀: y = 3x + 2 + 잡음
rng = np.random.default_rng(0)
x = rng.uniform(0, 10, size=(100, 1))
y = 3 * x[:, 0] + 2 + rng.normal(0, 1, 100)
lr = LinearRegression().fit(x, y)
print(f"w={lr.coef_[0]:.2f}, b={lr.intercept_:.2f}")    # 3과 2 근처

# 로지스틱 회귀: 유방암 진단 (특징 30개)
X, t = load_breast_cancer(return_X_y=True)
Xtr, Xte, ttr, tte = train_test_split(X, t, test_size=0.25, random_state=0)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ttr)
print(f"테스트 정확도: {clf.score(Xte, tte):.3f}")
print("첫 환자 악성/양성 확률:", clf.predict_proba(Xte[:1]).round(3))
```

`StandardScaler`로 특징을 평균 0·분산 1로 맞추는 것이 중요합니다. 단위가 제각각이면(밀리미터와 제곱밀리미터) 경사하강법이 느려지고 가중치 비교도 무의미해집니다.

??? expert "전문가 노트 — 정규 방정식의 함정, 릿지·라소, 로짓의 의미"
    **정규 방정식과 수치 안정성.** \((X^\top X)^{-1}\)은 특징끼리 강하게 상관되면(다중공선성) 거의 특이해져 가중치가 크게 흔들립니다. 실제 구현은 역행렬 대신 QR 분해나 SVD를 씁니다. 계산량이 \(O(Nd^2 + d^3)\)이라 d가 크면 경사하강법이 낫습니다.

    **릿지와 라소.** 손실에 \(\lambda\|\mathbf{w}\|_2^2\)를 더하면 릿지(해: \((X^\top X + \lambda I)^{-1}X^\top\mathbf{y}\), 항상 역행렬 존재), \(\lambda\|\mathbf{w}\|_1\)을 더하면 라소입니다. L1은 다이아몬드 모양 제약 영역의 꼭짓점에서 해가 나오기 쉬워 가중치 상당수가 **정확히 0**이 되므로 특징 선택 효과가 있습니다.

    **로짓과 오즈.** 로지스틱 회귀에서 \(z = \log\frac{p}{1-p}\)(로그 오즈)입니다. 가중치 \(w_j\)는 "특징 j가 1 늘면 오즈가 \(e^{w_j}\)배"라는 뜻이라 의학·사회과학에서 해석 도구로 많이 씁니다. 신경망에서 softmax 직전 값을 "로짓"이라고 부르는 것도 여기서 왔습니다.

    **완전 분리.** 데이터가 직선으로 완벽하게 나뉘면 로지스틱 회귀의 가중치는 무한대로 커지려 합니다(손실이 계속 줄어듦). 정규화가 이를 막습니다. scikit-learn의 `LogisticRegression`은 기본으로 L2 정규화(C = 1.0)가 켜져 있습니다.

## 관련 페이지

- [확률과 통계](../math/probability.md) — 최대우도와 손실의 관계
- [과적합과 정규화](overfitting.md) — 릿지·라소와 다항 회귀의 과적합
- [퍼셉트론과 MLP](../dl/mlp.md) — 로지스틱 회귀를 여러 층으로
- [평가 지표와 검증](evaluation.md) — 정확도 외의 지표
