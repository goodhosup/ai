# CNN

!!! spec "핵심 논문 · 연도"
    Fukushima, "Neocognitron" (1980) · LeCun 외, "Gradient-based learning applied to document recognition" (1998, LeNet-5) · Krizhevsky 외, "AlexNet" (2012) · Simonyan·Zisserman, "VGG" (2014) · He 외, "Deep Residual Learning for Image Recognition" (2015, ResNet) · Tan·Le, "EfficientNet" (2019) · Liu 외, "A ConvNet for the 2020s" (2022, ConvNeXt)

!!! basic "한눈에 보기"
    사진에서 고양이를 찾을 때, 사람은 사진 전체를 한 번에 보지 않고 **작은 창(돋보기)을 이리저리 옮기며** 귀, 눈, 수염 같은 부분을 찾습니다. **CNN(합성곱 신경망)**이 바로 이렇게 합니다.

    1. 작은 **필터**(예: 3×3)를 이미지 전체에 미끄러뜨리며 "여기에 세로 모서리가 있나?" 같은 패턴을 찾습니다. 같은 필터를 모든 위치에 재사용하므로 고양이가 왼쪽에 있든 오른쪽에 있든 찾을 수 있습니다.
    2. 층을 쌓을수록 모서리 → 질감 → 눈·귀 → 고양이 얼굴처럼 점점 큰 개념을 잡습니다.
    3. 중간중간 이미지를 **줄여서(풀링)** 계산량을 줄이고 더 넓은 범위를 봅니다.

    2012년 AlexNet 이후 10년간 이미지 인식의 표준이었고, 지금도 스마트폰·카메라·자동차의 온디바이스 비전에서 널리 쓰입니다.

## 합성곱 연산 { .l2 }

입력 채널 \(C_{in}\), 출력 채널 \(C_{out}\), 커널 크기 k×k인 합성곱 층은 출력 위치마다

\[
Y_{o}(i, j) = \sum_{c=1}^{C_{in}}\sum_{u,v} W_{o,c}(u, v)\,X_c(i+u, j+v) + b_o
\]

를 계산합니다.

| 하이퍼파라미터 | 뜻 | 출력 크기에 주는 영향 |
|---|---|---|
| 커널 크기 k | 필터 크기 (보통 3) | — |
| 스트라이드 s | 필터를 몇 칸씩 옮기나 | s = 2면 가로세로 절반 |
| 패딩 p | 가장자리에 0을 채움 | p = (k−1)/2면 크기 유지 |
| 채널 수 | 필터 개수 = 찾는 패턴 종류 수 | 출력 깊이 |

출력 크기: \(\left\lfloor\frac{H + 2p - k}{s}\right\rfloor + 1\)

## 왜 이미지에 잘 맞나 { .l2 }

| 성질 | 의미 | 효과 |
|---|---|---|
| 지역 연결 | 출력 한 칸은 입력의 작은 영역만 봄 | 파라미터 수 대폭 감소 |
| 가중치 공유 | 같은 필터를 모든 위치에 사용 | 위치가 바뀌어도 같은 패턴 검출 (이동 등변성) |
| 계층 구조 | 층이 깊어질수록 수용 영역(receptive field)이 커짐 | 작은 패턴 → 큰 개념 |

224×224×3 이미지를 MLP 첫 층(뉴런 1,000개)에 넣으면 약 1.5억 개 가중치가 필요하지만, 3×3 합성곱 64채널은 \(3\times3\times3\times64 + 64 = 1{,}792\)개면 됩니다.

## 대표 구조의 발전 { .l2 }

| 모델 | 연도 | 핵심 아이디어 | 깊이 |
|---|---|---|---|
| LeNet-5 | 1998 | 합성곱 + 풀링 + 완전 연결 | 7 |
| AlexNet | 2012 | GPU 학습, ReLU, 드롭아웃 | 8 |
| VGG | 2014 | 3×3 합성곱만 깊게 쌓기 | 16–19 |
| GoogLeNet | 2014 | 여러 크기 필터를 병렬로 (인셉션) | 22 |
| ResNet | 2015 | **잔차 연결**로 아주 깊게 | 50–152 |
| MobileNet | 2017 | 깊이별 분리 합성곱 (모바일용 경량화) | — |
| EfficientNet | 2019 | 깊이·폭·해상도를 함께 키우는 규칙 | — |
| ConvNeXt | 2022 | 트랜스포머의 설계 요소를 CNN에 도입 | — |

## CNN과 비전 트랜스포머 { .l2 }

2020년 **ViT**는 이미지를 16×16 조각(패치)으로 잘라 토큰처럼 트랜스포머에 넣었습니다. 데이터가 적으면 CNN이, 수억 장 규모로 사전 학습하면 ViT가 유리했습니다. 지금은 멀티모달 LLM의 이미지 인코더 대부분이 ViT 계열이고, 경량·실시간 비전(검출, 분할)에서는 CNN과 하이브리드가 여전히 강합니다 → [멀티모달](../apps/multimodal.md).

## 코드로 보기 { .l2 }

```python
import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 28 → 14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 14 → 7
        )
        self.head = nn.Linear(64 * 7 * 7, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))

model = SmallCNN()
x = torch.randn(8, 1, 28, 28)            # 흑백 28×28 이미지 8장
print(model(x).shape)                    # torch.Size([8, 10])
print("파라미터:", sum(p.numel() for p in model.parameters()))

# 잔차 블록: 출력 = 입력 + F(입력)
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                               nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c))
    def forward(self, x):
        return torch.relu(x + self.f(x))
print(ResBlock(64)(torch.randn(2, 64, 7, 7)).shape)
```

MNIST로 실제 학습하는 예는 [손글씨 숫자 분류 (MNIST)](../practice/mnist.md)에 있습니다.

??? expert "전문가 노트 — 연산량, 깊이별 분리 합성곱, im2col과 하드웨어"
    **연산량.** 출력 크기 H×W, 커널 k×k, 채널 \(C_{in}\to C_{out}\)인 합성곱의 곱셈-덧셈(MAC) 수는 \(H W k^2 C_{in} C_{out}\)입니다. ResNet-50은 224×224 입력 한 장에 약 4 GMAC(약 8 GFLOPs), 파라미터 약 2,560만 개입니다. CNN은 파라미터에 비해 연산이 많아(가중치 재사용이 큼) 연산 집약적이고, 이것이 NPU 설계에서 MAC 배열을 크게 잡는 이유입니다.

    **깊이별 분리 합성곱.** 일반 합성곱을 채널별 k×k 합성곱(depthwise)과 1×1 합성곱(pointwise)으로 나누면 연산이 약 \(\frac{1}{C_{out}} + \frac{1}{k^2}\)배로 줄어듭니다(3×3이면 약 8–9배 감소). MobileNet·EfficientNet의 핵심이지만, 산술 강도(연산/메모리 접근)가 낮아 GPU에서는 이론만큼 빠르지 않을 수 있습니다.

    **im2col과 하드웨어.** 합성곱은 입력 패치를 펼쳐 행렬로 만들면(im2col) 하나의 큰 행렬 곱이 되어 GEMM 라이브러리·텐서 코어로 가속됩니다. 3×3 합성곱에는 곱셈 수를 줄이는 Winograd 변환도 씁니다. NPU는 데이터 재사용 방식(가중치 고정, 출력 고정, 행 고정 데이터플로)에 따라 설계가 달라집니다 → [NPU와 온디바이스 AI](../systems/npu.md).

    **수용 영역과 유효 수용 영역.** 3×3 합성곱 n층의 이론적 수용 영역은 \(2n+1\)이지만, 실제로 출력에 크게 영향을 주는 영역(유효 수용 영역)은 가운데에 몰린 가우시안 모양이라 훨씬 작습니다(Luo 외 2016). 전역 정보가 필요하면 다운샘플링을 늘리거나 어텐션을 섞습니다.

## 관련 페이지

- [초기화와 학습 안정성](../dl/training-stability.md) — 잔차 연결
- [정규화 기법](../dl/normalization.md) — BatchNorm
- [멀티모달](../apps/multimodal.md) — 비전 트랜스포머
- [NPU와 온디바이스 AI](../systems/npu.md) — CNN 가속 하드웨어
