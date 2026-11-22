# 결정 트리와 앙상블

!!! spec "핵심 논문 · 연도"
    Breiman 외, *Classification and Regression Trees* (1984, CART) · Quinlan, "C4.5" (1993) · Breiman, "Bagging Predictors" (1996), "Random Forests" (2001) · Freund·Schapire, AdaBoost (1997) · Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" (2001) · Chen·Guestrin, "XGBoost" (2016) · Ke 외, "LightGBM" (2017) · Grinsztajn 외, "Why do tree-based models still outperform deep learning on tabular data?" (2022)

!!! basic "한눈에 보기"
    **결정 트리**는 스무고개입니다. "소득이 5천만 원 이상인가? → 예 → 연체 이력이 있는가? → 아니오 → 대출 승인." 질문을 따라 내려가면 답이 나옵니다. 사람이 읽고 이해하기 쉽다는 큰 장점이 있습니다.

    하지만 트리 하나는 학습 데이터에 쉽게 과적합합니다. 그래서 여러 트리를 묶는 **앙상블**을 씁니다.

    1. **랜덤 포레스트** — 조금씩 다른 데이터로 트리 수백 개를 따로 키우고 **투표**합니다. "여러 사람의 평균은 한 사람보다 정확하다."
    2. **그래디언트 부스팅** — 트리를 **차례로** 키우되, 새 트리는 앞 트리들이 **틀린 만큼**을 고치도록 배웁니다. 오답 노트를 반복하는 학생과 같습니다.

    엑셀 표 같은 데이터(금융, 제조 센서, 마케팅)에서는 지금도 그래디언트 부스팅(XGBoost, LightGBM)이 딥러닝을 이기는 경우가 많습니다.

## 트리는 어떻게 자라나 { .l2 }

각 노드에서 "어떤 특징을 어떤 값으로 나누면 양쪽이 가장 **순수**해지나"를 찾아 나눕니다.

| 기준 | 식 (클래스 비율 \(p_k\)) | 쓰임 |
|---|---|---|
| 지니 불순도 | \(1 - \sum_k p_k^2\) | CART, scikit-learn 기본 |
| 엔트로피 | \(-\sum_k p_k \log p_k\) | ID3, C4.5 |
| 분산 감소 | 자식 노드의 분산 합 | 회귀 트리 |

멈추는 조건(최대 깊이, 잎의 최소 샘플 수)을 두지 않으면 잎마다 샘플 하나가 남을 때까지 자라 학습 데이터를 완벽히 외웁니다.

## 배깅과 랜덤 포레스트 { .l2 }

1. 학습 데이터에서 **복원 추출**로 같은 크기의 표본을 여러 벌 만듭니다(부트스트랩).
2. 표본마다 트리를 키우되, 노드마다 **특징도 일부만 무작위로** 후보로 씁니다.
3. 분류는 다수결, 회귀는 평균으로 예측합니다.

트리끼리 상관이 낮을수록 평균의 분산이 줄어듭니다. 특징을 무작위로 고르는 것이 상관을 낮추는 핵심입니다. 부트스트랩에서 빠진 약 37%의 샘플(OOB)로 별도 검증셋 없이 성능을 추정할 수도 있습니다.

## 그래디언트 부스팅 { .l2 }

\[
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu\, h_m(\mathbf{x})
\]

새 트리 \(h_m\)은 현재 모델의 손실 기울기(MSE라면 **잔차** \(y - F_{m-1}(\mathbf{x})\))를 맞히도록 학습합니다. \(\nu\)(학습률, 보통 0.01–0.1)를 작게 하고 트리를 많이 쓰면 일반화가 좋아집니다. 이름 그대로 "함수 공간에서의 경사하강법"입니다.

| 항목 | 랜덤 포레스트 | 그래디언트 부스팅 |
|---|---|---|
| 트리 학습 | 병렬, 서로 독립 | 순차, 앞 트리에 의존 |
| 트리 깊이 | 깊게 (과적합을 평균으로 상쇄) | 얕게 (3–8) |
| 주로 줄이는 것 | 분산 | 편향 |
| 튜닝 민감도 | 낮음 (기본값도 괜찮음) | 높음 (학습률·트리 수·깊이) |
| 성능 | 좋음 | 표 데이터에서 보통 최고 |

## 왜 표 데이터에서는 트리가 강한가 { .l2 }

| 이유 | 설명 |
|---|---|
| 의미 없는 특징에 강함 | 쓸모없는 열은 분할에 거의 안 쓰임 |
| 불규칙한 경계 | "나이 ≥ 65" 같은 계단식 규칙을 자연스럽게 표현 |
| 단위 무관 | 정규화가 필요 없음 |
| 데이터가 적어도 됨 | 수천–수십만 행에서 잘 작동 |

## 코드로 보기 { .l2 }

```python
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score

# 비선형 회귀 벤치마크: 특징 10개 중 5개만 의미 있음
X, y = make_friedman1(n_samples=5000, n_features=10, noise=1.0, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

models = {
    "트리 1개 (깊이 제한 없음)": DecisionTreeRegressor(random_state=0),
    "랜덤 포레스트 (100개)": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0),
    "그래디언트 부스팅": HistGradientBoostingRegressor(random_state=0),
}
for name, m in models.items():
    m.fit(Xtr, ytr)
    print(f"{name:22s} 학습 R²={r2_score(ytr, m.predict(Xtr)):.3f}  테스트 R²={r2_score(yte, m.predict(Xte)):.3f}")
```

트리 1개는 학습 R²가 1.0(완벽히 외움)인데 테스트는 약 0.75에 그칩니다. 랜덤 포레스트는 약 0.90, 그래디언트 부스팅은 약 0.94까지 올라갑니다.

??? expert "전문가 노트 — XGBoost의 2차 근사, 특징 중요도의 함정, 트리와 신경망"
    **2차 근사.** XGBoost는 손실을 현재 예측 주변에서 2차 테일러 전개하고(기울기 \(g_i\), 헤시안 \(h_i\)), 잎 j의 최적 값을 \(w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}\)로 닫힌 식으로 구합니다. 분할 이득도 같은 식에서 나오므로 임의의 미분 가능한 손실을 같은 코드로 다룹니다. LightGBM은 특징 값을 구간(히스토그램)으로 묶고 잎 단위로 성장시켜 대용량에서 빠릅니다. scikit-learn의 `HistGradientBoosting*`도 같은 아이디어입니다.

    **특징 중요도의 함정.** 분할 횟수나 불순도 감소 기반 중요도는 값의 종류가 많은 특징(ID, 연속값)을 과대평가합니다. 순열 중요도(permutation importance)나 SHAP 값이 더 믿을 만하지만, 상관된 특징끼리는 중요도가 나눠지는 문제가 여전히 있습니다. 중요도는 인과 관계가 아닙니다.

    **트리와 신경망의 결합.** 표 데이터용 딥러닝(TabNet, FT-Transformer)과 사전 학습된 표 파운데이션 모델(TabPFN 계열)이 연구되고 있고, 작은 데이터셋에서는 부스팅과 경쟁하는 결과도 보고됩니다. 실무에서는 여전히 부스팅이 기준선입니다.

    **외삽 불가.** 트리는 학습 데이터 범위 밖의 값을 예측하지 못합니다(잎의 평균값만 냄). 시계열 추세처럼 범위를 벗어나는 예측이 필요하면 선형 항을 함께 쓰거나 다른 모델을 고려합니다.

## 관련 페이지

- [과적합과 정규화](overfitting.md) — 편향-분산 트레이드오프
- [평가 지표와 검증](evaluation.md) — R², 교차 검증
- [AI·머신러닝·딥러닝·생성형 AI](../getting-started/landscape.md) — 고전 ML과 딥러닝의 경계
