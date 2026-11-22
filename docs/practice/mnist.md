# 손글씨 숫자 분류 (MNIST)

!!! spec "준비물 · 소요 시간"
    [파이썬과 PyTorch 환경](setup.md) · CPU로 충분 (실행 약 10초) · 약 40분. 관련 이론: [퍼셉트론과 MLP](../dl/mlp.md), [역전파](../dl/backprop.md), [CNN](../arch/cnn.md)

!!! basic "한눈에 보기"
    손글씨 숫자 이미지를 보고 0–9 중 무엇인지 맞히는 문제는 딥러닝의 "Hello, World"입니다. 1998년 LeCun의 LeNet이 은행 수표의 숫자를 읽던 시절부터 지금까지, 새로운 사람이 신경망을 처음 배울 때 거치는 관문입니다.

    이 실습에서는 신경망 학습의 **전체 흐름**을 한 번에 경험합니다.

    1. 데이터 준비: 이미지를 숫자 텐서로 바꾸고 학습용·평가용으로 나눕니다.
    2. 모델 정의: 작은 CNN을 만듭니다.
    3. 학습 루프: 예측 → 손실 계산 → 역전파 → 가중치 갱신을 반복합니다.
    4. 평가: 한 번도 보지 않은 데이터로 정확도를 재고, 틀린 예를 살펴봅니다.

    이 네 단계는 GPT 같은 거대 모델을 학습할 때도 **똑같습니다**. 규모와 데이터만 다를 뿐입니다.

## 데이터에 대하여 { .l2 }

| 데이터셋 | 크기 | 이미지 | 얻는 법 |
|---|---|---|---|
| scikit-learn digits | 1,797장 | 8×8, 회색조 16단계 | `load_digits()` — 설치 시 포함, 인터넷 불필요 |
| MNIST | 학습 6만 + 평가 1만 | 28×28, 회색조 | `torchvision.datasets.MNIST(download=True)` |
| Fashion-MNIST | MNIST와 같음 | 28×28 옷·신발 사진 | MNIST보다 약간 어려운 대체재 |

아래 코드는 인터넷 연결 없이도 바로 돌아가도록 **scikit-learn의 digits**(작은 MNIST)를 씁니다. 진짜 MNIST로 바꾸는 방법은 코드 다음에 있습니다.

## 코드로 보기 { .l2 }

```python
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)                  # 8×8 손글씨 숫자 1,797장 (설치 시 포함)
X = torch.tensor(X / 16.0, dtype=torch.float32).reshape(-1, 1, 8, 8)
y = torch.tensor(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8×8 → 4×4
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 4×4 → 2×2
    nn.Flatten(), nn.Linear(32 * 2 * 2, 10),
)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()
print("파라미터 수:", sum(p.numel() for p in model.parameters()))

for epoch in range(1, 16):
    model.train()
    for i in torch.randperm(len(Xtr)).split(64):        # 미니배치 64
        loss = loss_fn(model(Xtr[i]), ytr[i])
        opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        print(f"epoch {epoch:2d}  train loss {loss.item():.3f}  test acc {acc:.3f}")

with torch.no_grad():
    pred = model(Xte).argmax(1)
wrong = (pred != yte).nonzero().flatten()
print("틀린 예:", [(int(yte[i]), int(pred[i])) for i in wrong[:5]], "(정답, 예측)")
```

파라미터 6,090개짜리 작은 CNN이 CPU에서 몇 초 만에 학습됩니다. 평가 정확도는 5에포크 0.949 → 10에포크 0.969 → 15에포크 **0.980**으로 오릅니다. 450장 중 9장 정도만 틀리는 셈입니다. 틀린 예를 보면 4를 9로, 5를 3으로 읽는 등 사람도 헷갈릴 만한 쌍이 많습니다. 8×8 해상도는 매우 거칠기 때문에, 직접 `Xte[i]`를 그려 보면 왜 틀렸는지 이해가 됩니다.

## 코드 읽기 { .l2 }

| 줄 | 하는 일 | 관련 개념 |
|---|---|---|
| `X / 16.0` | 픽셀 값(0–16)을 0–1로 정규화 | 입력 스케일을 맞춰야 학습이 안정적 |
| `reshape(-1, 1, 8, 8)` | (개수, 채널, 높이, 너비) 형태 | PyTorch 이미지 텐서 규칙 |
| `stratify=y` | 숫자별 비율을 학습·평가에 똑같이 | [평가 지표와 검증](../ml/evaluation.md) |
| `Conv2d` → `ReLU` → `MaxPool2d` | 지역 패턴 감지 → 비선형 → 크기 축소 | [CNN](../arch/cnn.md) |
| `CrossEntropyLoss` | 소프트맥스 + 음의 로그 가능도 | [정보이론](../math/information.md) |
| `opt.zero_grad(); loss.backward(); opt.step()` | 기울기 초기화 → 역전파 → 갱신 | [역전파](../dl/backprop.md), [옵티마이저](../dl/optimizers.md) |
| `model.eval()` / `torch.no_grad()` | 평가 모드, 기울기 계산 끔 | 메모리·속도 절약 |

## 진짜 MNIST로 바꾸기 { .l2 }

인터넷이 되는 환경이라면 `pip install torchvision` 후 데이터 준비 부분만 바꿉니다. 이미지가 28×28이 되므로 마지막 `Linear`의 입력은 32×7×7이 됩니다.

```py
from torchvision import datasets
tr = datasets.MNIST("data", train=True, download=True)
te = datasets.MNIST("data", train=False, download=True)
Xtr = tr.data.float().div(255).unsqueeze(1); ytr = tr.targets   # (60000, 1, 28, 28)
Xte = te.data.float().div(255).unsqueeze(1); yte = te.targets
# 모델 마지막 층: nn.Linear(32 * 7 * 7, 10)
```

같은 구조로 몇 에포크면 평가 정확도 99% 안팎에 도달합니다. CPU에서도 에포크당 1분 이내입니다.

## 해 볼 만한 실험 { .l2 }

1. `nn.Conv2d` 층을 빼고 `nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)`의 MLP로 바꿔 정확도를 비교해 보세요.
2. 학습률을 `3e-2`, `3e-4`로 바꿔 손실 곡선이 어떻게 달라지는지 보세요 → [최적화의 기초](../math/optimization.md).
3. 학습 데이터를 10%만 쓰면 평가 정확도가 얼마나 떨어지는지, `nn.Dropout`을 넣으면 나아지는지 보세요 → [과적합과 정규화](../ml/overfitting.md).
4. 혼동 행렬(`sklearn.metrics.confusion_matrix`)을 그려 어떤 숫자 쌍을 가장 많이 헷갈리는지 찾아보세요.

??? expert "전문가 노트 — DataLoader, 학습 루프의 표준 구성, MNIST 이후"
    **DataLoader.** 위 코드는 데이터가 작아 텐서를 직접 잘랐지만, 실제로는 `torch.utils.data.Dataset`과 `DataLoader`로 셔플·배치·병렬 로딩·데이터 증강을 처리합니다. 이미지 증강(회전, 이동)을 넣으면 MNIST 정확도가 99.5% 이상으로 오릅니다.

    **표준 학습 루프의 구성 요소.** 실무 코드에는 검증 세트로 조기 종료, 학습률 스케줄러(워밍업 + 코사인 감쇠), 체크포인트 저장, 혼합 정밀도(`torch.autocast`), 기울기 클리핑, 로그 기록(TensorBoard, Weights & Biases)이 추가됩니다. 이 요소들은 LLM 학습에서도 그대로 쓰입니다 → [초기화와 학습 안정성](../dl/training-stability.md).

    **MNIST 이후.** MNIST는 너무 쉬워서 모델 비교용으로는 더 이상 쓰이지 않습니다. 다음 단계로는 CIFAR-10(컬러 사물 사진 32×32), 사전학습된 ResNet·ViT를 내 데이터에 미세 조정하는 전이 학습, 그리고 텍스트로 넘어가 [작은 트랜스포머 직접 만들기](mini-transformer.md)를 권합니다.

## 관련 페이지

- [CNN](../arch/cnn.md) — 합성곱의 원리
- [역전파](../dl/backprop.md) — `loss.backward()` 안에서 일어나는 일
- [작은 트랜스포머 직접 만들기](mini-transformer.md) — 다음 실습
