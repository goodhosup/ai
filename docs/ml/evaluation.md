# 평가 지표와 검증

!!! spec "핵심 논문 · 연도"
    Stone, "Cross-Validatory Choice and Assessment of Statistical Predictions" (1974) · Fawcett, "An introduction to ROC analysis" (2006) · Saito·Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot" (2015) · Kapoor·Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science" (2023)

!!! basic "한눈에 보기"
    환자 1,000명 중 10명만 암인 데이터에서, 모든 사람에게 "암 아님"이라고 답하는 모델의 정확도는 **99%**입니다. 그런데 이 모델은 암 환자를 한 명도 찾지 못합니다. 그래서 **무엇을 놓치면 안 되는지**에 맞는 지표를 골라야 합니다.

    - **정밀도(precision)**: 모델이 "암"이라고 한 사람 중 진짜 암 환자의 비율. "헛경보가 적은가?"
    - **재현율(recall)**: 실제 암 환자 중 모델이 찾아낸 비율. "놓친 환자가 적은가?"

    또 하나, 모델을 고르고 조정할 때 쓴 데이터로 최종 성적을 매기면 **자기 채점**이 됩니다. 시험 문제를 미리 본 셈이니, 마지막 평가용 데이터(테스트셋)는 끝까지 따로 둡니다.

## 데이터 분할 { .l2 }

| 세트 | 비율 예 | 용도 |
|---|---|---|
| 학습 (train) | 70–80% | 파라미터 학습 |
| 검증 (validation) | 10–15% | 모델·하이퍼파라미터 선택, 조기 종료 |
| 테스트 (test) | 10–15% | 마지막에 **한 번만** 성능 보고 |

데이터가 적으면 **k겹 교차 검증**을 씁니다. 데이터를 k등분해 하나씩 돌아가며 검증에 쓰고 k번의 평균을 냅니다.

| 분할 주의점 | 이유 |
|---|---|
| 시계열은 시간 순서로 분할 | 미래 데이터로 과거를 예측하면 성능이 부풀려짐 |
| 같은 환자·사용자는 한쪽에만 | 같은 사람의 다른 사진이 양쪽에 있으면 얼굴을 외워 맞힘 |
| 클래스 비율 유지 (층화) | 드문 클래스가 검증셋에 없을 수 있음 |
| 전처리는 학습셋으로만 fit | 정규화 평균을 전체 데이터로 계산하면 테스트 정보 누출 |

## 분류 지표 { .l2 }

혼동 행렬(confusion matrix)에서 출발합니다.

| | 예측 양성 | 예측 음성 |
|---|---|---|
| **실제 양성** | TP (맞게 찾음) | FN (놓침) |
| **실제 음성** | FP (헛경보) | TN |

| 지표 | 식 | 중요한 상황 |
|---|---|---|
| 정확도 | (TP+TN) / 전체 | 클래스가 균형일 때만 |
| 정밀도 | TP / (TP+FP) | 헛경보가 비쌀 때 (스팸 필터: 정상 메일을 버리면 안 됨) |
| 재현율 (민감도) | TP / (TP+FN) | 놓치면 큰일일 때 (암 진단, 불량 검출, 사기 탐지) |
| F1 | 정밀도와 재현율의 조화 평균 | 둘의 균형 |
| ROC-AUC | 모든 임곗값에서 TPR·FPR 곡선 아래 면적 | 순위 매기기 성능, 임곗값 무관 |
| PR-AUC | 정밀도-재현율 곡선 아래 면적 | 양성이 매우 드물 때 ROC-AUC보다 정직 |

정밀도와 재현율은 **임곗값**으로 맞바꿉니다. 확률 0.5 대신 0.2 이상을 양성으로 보면 재현율은 오르고 정밀도는 떨어집니다. 업무상 비용(놓친 환자 1명 vs 추가 검사 1건)을 따져 임곗값을 정합니다.

## 회귀 지표 { .l2 }

| 지표 | 식 | 특징 |
|---|---|---|
| MAE | \(\frac{1}{N}\sum\lvert y-\hat{y}\rvert\) | 단위가 원래와 같아 해석 쉬움, 이상치에 덜 민감 |
| RMSE | \(\sqrt{\frac{1}{N}\sum(y-\hat{y})^2}\) | 큰 오차를 더 크게 벌함 |
| R² | \(1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}\) | 평균값 예측 대비 얼마나 나은가 (1이 완벽, 0은 평균 수준) |
| MAPE | 평균 백분율 오차 | 실제값이 0 근처면 폭발 |

## 생성 모델은 어떻게 평가하나 { .l2 }

LLM처럼 답이 하나로 정해지지 않으면 위 지표를 그대로 쓰기 어렵습니다. 정답 비교(객관식, 수학 정답, 코드 테스트 통과율), 사람 평가, 다른 LLM을 심판으로 쓰는 방법(LLM-as-a-judge)을 섞어 씁니다 → [평가와 벤치마크](../apps/evaluation.md).

## 코드로 보기 { .l2 }

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

# 양성 약 2.5%인 불균형 데이터 (라벨 잡음 1% 포함)
X, y = make_classification(n_samples=20000, n_features=20, weights=[0.98],
                           class_sep=0.8, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

print("모두 음성이라고 답하면 정확도:", round(1 - yte.mean(), 3))
p = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
for th in (0.5, 0.2):
    pred = (p >= th).astype(int)
    print(f"임곗값 {th}: 정확도={accuracy_score(yte, pred):.3f} 정밀도={precision_score(yte, pred):.3f} "
          f"재현율={recall_score(yte, pred):.3f} F1={f1_score(yte, pred):.3f}")
print(f"ROC-AUC={roc_auc_score(yte, p):.3f}  PR-AUC={average_precision_score(yte, p):.3f}")
```

"모두 음성"도 정확도 0.975가 나오고, 학습한 모델의 정확도(0.978)와 거의 차이가 없습니다. 임곗값 0.5에서는 정밀도 1.0이지만 재현율이 0.11로 양성 대부분을 놓칩니다. 0.2로 내리면 재현율이 0.31로 오르고 정밀도는 0.54로 내려갑니다. ROC-AUC는 0.78로 괜찮아 보여도 PR-AUC는 0.38에 그쳐, 드문 양성을 찾는 일이 실제로는 어렵다는 것을 보여 줍니다.

??? expert "전문가 노트 — 데이터 누출, 검증셋 과적합, 통계적 유의성"
    **데이터 누출(leakage).** 예측 시점에는 알 수 없는 정보가 특징에 섞이는 것입니다. 예: "입원 기간"으로 "입원 필요 여부"를 예측, 환불 처리 후 생성되는 필드로 사기 탐지. Kapoor·Narayanan(2023)은 ML을 쓴 과학 논문 수백 편이 누출로 성능을 과대 보고했다고 정리했습니다. 성능이 "너무 좋으면" 먼저 누출을 의심합니다.

    **검증셋 과적합.** 검증셋으로 수백 번 하이퍼파라미터를 고르면 검증셋 자체에 과적합합니다. 공개 벤치마크 리더보드도 같은 문제를 겪습니다. 중첩 교차 검증이나 진짜로 한 번만 쓰는 테스트셋이 해법입니다.

    **불확실성 표기.** 테스트셋 1,000개에서 정확도 90%의 95% 신뢰구간은 대략 \(\pm 1.96\sqrt{0.9\times0.1/1000} \approx \pm 1.9\)%p입니다. 두 모델의 차이가 이보다 작으면 같은 테스트 문항에 대한 짝지은 검정(McNemar)이나 부트스트랩으로 유의성을 확인합니다.

    **보정과 임곗값.** 클래스 가중치나 오버샘플링(SMOTE)으로 불균형을 다루면 예측 확률이 왜곡됩니다. 순위만 필요하면 문제없지만, 확률 자체를 쓰려면 보정(Platt, isotonic) 후 임곗값을 정합니다.

## 관련 페이지

- [과적합과 정규화](overfitting.md)
- [평가와 벤치마크](../apps/evaluation.md) — LLM 평가
- [편향과 공정성](../society/fairness.md) — 집단별 지표 차이
