# SVM과 커널

!!! spec "핵심 논문 · 연도"
    Vapnik·Chervonenkis (1963, 최적 분리 초평면) · Boser·Guyon·Vapnik, "A Training Algorithm for Optimal Margin Classifiers" (1992, 커널 트릭) · Cortes·Vapnik, "Support-Vector Networks" (1995, 소프트 마진) · Schölkopf·Smola, *Learning with Kernels* (2002) · Rahimi·Recht, "Random Features for Large-Scale Kernel Machines" (2007) · Jacot 외, "Neural Tangent Kernel" (2018)

!!! basic "한눈에 보기"
    빨간 공과 파란 공을 막대기 하나로 가르는 방법은 여러 가지입니다. **SVM(서포트 벡터 머신)**은 그중 양쪽 공에서 **가장 멀리 떨어진** 막대기를 고릅니다. 길 한가운데에 선을 긋고, 길 폭(마진)이 최대가 되게 하는 것입니다. 길 폭이 넓을수록 새 공이 와도 헷갈릴 가능성이 줄어듭니다.

    공이 직선으로 안 갈리면? 종이를 **구부려** 높은 차원으로 들어 올린 뒤 평평한 판으로 자릅니다. 이것을 실제로 들어 올리지 않고 계산만으로 해내는 묘기가 **커널 트릭**입니다.

    2000년대에는 SVM이 손글씨·문서 분류의 최강자였고, 딥러닝이 그 자리를 이어받았습니다.

## 최대 마진 분류기 { .l2 }

라벨 \(y_i \in \{-1, +1\}\), 경계 \(\mathbf{w}^\top\mathbf{x} + b = 0\)일 때, 경계와 가장 가까운 점 사이 거리(마진)는 \(1/\|\mathbf{w}\|\)입니다. 마진 최대화는 다음과 같습니다.

\[
\min_{\mathbf{w}, b}\; \frac{1}{2}\|\mathbf{w}\|^2
\quad\text{s.t.}\quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1
\]

경계를 실제로 결정하는 것은 마진 위에 놓인 소수의 점뿐이며, 이들을 **서포트 벡터**라고 합니다. 나머지 점은 지워도 경계가 바뀌지 않습니다.

## 소프트 마진과 힌지 손실 { .l2 }

실제 데이터는 완벽히 갈리지 않으므로 일부 위반을 허용합니다.

\[
\min_{\mathbf{w}, b}\; \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \max\big(0,\; 1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)\big)
\]

뒤쪽 항이 **힌지 손실**입니다. C가 크면 위반을 엄하게 벌해 경계가 복잡해지고(과적합 위험), 작으면 마진을 넓게 잡습니다(과소적합 위험).

| 손실 | 식 (\(m = y\,f(\mathbf{x})\)) | 모델 |
|---|---|---|
| 힌지 | \(\max(0, 1 - m)\) | SVM |
| 로지스틱 | \(\log(1 + e^{-m})\) | 로지스틱 회귀 |
| 0-1 | \(m < 0\)이면 1 | 실제 오분류율 (최적화 불가) |

## 커널 트릭 { .l2 }

SVM의 해는 데이터 점들의 **내적** \(\mathbf{x}_i^\top\mathbf{x}_j\)으로만 표현됩니다. 그래서 내적을 커널 함수 \(k(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top\phi(\mathbf{x}_j)\)로 바꾸면, 고차원 특징 \(\phi\)를 직접 계산하지 않고도 그 공간에서 선형 분리를 할 수 있습니다.

| 커널 | 식 | 특징 |
|---|---|---|
| 선형 | \(\mathbf{x}^\top\mathbf{z}\) | 고차원 희소 데이터(텍스트)에 적합 |
| 다항 | \((\mathbf{x}^\top\mathbf{z} + c)^p\) | p차 상호작용 |
| RBF (가우시안) | \(\exp(-\gamma\|\mathbf{x}-\mathbf{z}\|^2)\) | 무한 차원 특징. 가장 많이 씀 |

RBF의 γ가 크면 각 점의 영향 범위가 좁아져 경계가 구불구불해집니다.

## 코드로 보기 { .l2 }

원 안과 밖으로 나뉜 데이터(직선으로는 불가능)를 선형 SVM과 RBF SVM으로 비교합니다.

```python
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=600, factor=0.4, noise=0.1, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

for kernel in ("linear", "rbf"):
    clf = SVC(kernel=kernel, C=1.0).fit(Xtr, ytr)
    print(f"{kernel:6s} 정확도={clf.score(Xte, yte):.3f}  서포트 벡터 수={clf.n_support_.sum()}")
```

선형 커널은 약 55%(찍기 수준)이고 서포트 벡터가 400개 넘게 필요하지만, RBF 커널은 서포트 벡터 40개로 100% 정확도를 냅니다.

## SVM과 딥러닝 { .l2 }

| 항목 | 커널 SVM | 딥러닝 |
|---|---|---|
| 특징 | 커널이 고정 (사람이 선택) | 데이터로 학습 |
| 최적화 | 볼록 (전역 최적 보장) | 비볼록 |
| 데이터 규모 | 수만 개까지 (커널 행렬 N²) | 수십억 개 |
| 지금의 쓰임 | 작은 데이터, 이론 도구 | 대부분의 지각·언어 과제 |

??? expert "전문가 노트 — 쌍대 문제, 표현 정리, 신경망 탄젠트 커널"
    **쌍대 문제.** 라그랑주 승수 \(\alpha_i\)를 도입하면 \(\max_\alpha \sum_i\alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_iy_j k(\mathbf{x}_i, \mathbf{x}_j)\), \(0 \le \alpha_i \le C\), \(\sum\alpha_iy_i = 0\)이 됩니다. 서포트 벡터는 \(\alpha_i > 0\)인 점이며, 예측은 \(f(\mathbf{x}) = \sum_i\alpha_iy_ik(\mathbf{x}_i, \mathbf{x}) + b\)입니다. SMO 알고리즘이 두 개씩 승수를 풀어 효율적으로 최적화합니다. 커널 행렬이 \(N\times N\)이라 수십만 샘플을 넘으면 메모리·시간이 감당하기 어렵고, 랜덤 특징(Rahimi·Recht)이나 Nyström 근사로 우회합니다.

    **표현 정리.** 재생 커널 힐베르트 공간(RKHS)에서 정규화된 손실의 최적해는 항상 학습 점들의 커널 합 \(\sum_i\alpha_ik(\mathbf{x}_i, \cdot)\)으로 표현됩니다. 커널 릿지 회귀, 가우시안 프로세스가 같은 틀에 속합니다.

    **신경망 탄젠트 커널(NTK).** 폭이 무한대인 신경망을 작은 학습률로 학습하면 파라미터가 초기값 근처에서 거의 움직이지 않고, 학습 동역학이 특정 커널(NTK)을 쓴 커널 회귀와 같아진다는 결과입니다(Jacot 외 2018). 신경망을 이론적으로 분석하는 도구가 되었지만, 실제 유한 폭 신경망은 특징이 크게 변하는 "특징 학습" 체제에서 작동하며 그것이 커널 방법보다 강한 이유로 여겨집니다.

## 관련 페이지

- [선형 회귀와 로지스틱 회귀](linear-models.md) — 힌지 손실과 로지스틱 손실 비교
- [최적화의 기초](../math/optimization.md) — 볼록 최적화
- [과적합과 정규화](overfitting.md) — C와 γ의 역할
