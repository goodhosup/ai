# 초기화와 학습 안정성

!!! spec "핵심 논문 · 연도"
    Glorot·Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010, Xavier 초기화) · He 외, "Delving Deep into Rectifiers" (2015, He 초기화) · He 외, "Deep Residual Learning" (2015, 잔차 연결) · Pascanu 외, "On the difficulty of training RNNs" (2013, 기울기 클리핑) · Micikevicius 외, "Mixed Precision Training" (2017) · Wortsman 외, "Small-scale proxies for large-scale Transformer training instabilities" (2023)

!!! basic "한눈에 보기"
    깊은 신경망 학습은 **수십 층짜리 확성기 체인**과 같습니다. 각 층이 소리를 조금만 키워도 끝에서는 귀가 터지고(**폭주**), 조금만 줄여도 끝에서는 아무것도 안 들립니다(**소실**).

    - **초기화**: 학습 시작 시 가중치를 "소리가 층을 지나도 크기가 유지되도록" 알맞은 크기의 난수로 정합니다.
    - **잔차 연결**: 층을 건너뛰는 지름길을 만들어 신호와 기울기가 직접 흐르게 합니다.
    - **기울기 클리핑**: 기울기가 갑자기 너무 커지면 상한으로 잘라 한 번에 크게 튀지 않게 합니다.

    대형 모델 학습에서는 손실이 갑자기 치솟거나 NaN이 나오는 일이 흔하며, 이를 미리 막고 감지하고 되돌리는 것이 엔지니어링의 큰 부분입니다.

## 초기화 { .l2 }

층 하나 \(y = W\mathbf{x}\)에서 입력이 n개이고 가중치가 분산 \(\sigma^2\)인 독립 난수라면 \(\mathrm{Var}(y) \approx n\,\sigma^2\,\mathrm{Var}(x)\)입니다. 층을 지나도 분산이 유지되려면 \(n\sigma^2 \approx 1\)이어야 합니다.

| 초기화 | 분산 | 맞는 활성화 |
|---|---|---|
| Xavier (Glorot) | \(\frac{2}{n_{in} + n_{out}}\) | tanh, 시그모이드, 선형 |
| He (Kaiming) | \(\frac{2}{n_{in}}\) | ReLU (절반이 0이 되므로 2배) |
| 작은 정규분포 | 표준편차 0.02 고정 | GPT-2 등 트랜스포머 관례 |
| 잔차 출력 축소 | 잔차 경로 마지막 층을 \(1/\sqrt{2L}\)배 | 깊은 트랜스포머 (GPT-2) |
| 0 초기화 | 마지막 층이나 LoRA의 B 행렬 | 학습 시작 시 원래 동작 유지 |

모든 가중치를 같은 값(예: 0)으로 두면 같은 층의 뉴런이 모두 같은 기울기를 받아 영원히 똑같이 움직입니다. 그래서 **무작위**가 필수입니다.

## 잔차 연결 { .l2 }

\[
\mathbf{x}_{l+1} = \mathbf{x}_l + F(\mathbf{x}_l)
\]

역전파에서 \(\partial\mathbf{x}_{l+1}/\partial\mathbf{x}_l = I + \partial F/\partial\mathbf{x}_l\)이므로, F의 기울기가 작아도 **항등 행렬 I**를 통해 기울기가 그대로 전달됩니다. ResNet(2015)이 152층을 학습할 수 있었던 이유이며, 트랜스포머의 모든 블록이 이 구조입니다.

## 학습이 망가지는 증상과 처방 { .l2 }

| 증상 | 흔한 원인 | 처방 |
|---|---|---|
| 손실이 처음부터 NaN | 학습률 과대, 0으로 나누기, log(0), FP16 오버플로 | 학습률 낮추기, ε 추가, BF16 사용, 입력 확인 |
| 손실이 안 줄어듦 | 학습률 과소, 기울기 소실, 라벨 오류, 버그 | 작은 데이터 하나에 과적합되는지부터 확인 |
| 손실 급등(spike) 후 회복 | 드물게 큰 기울기, 나쁜 배치 | 기울기 클리핑, β₂ 낮추기 |
| 손실 급등 후 발산 | 어텐션 로짓 폭주, 출력 로짓 폭주 | 체크포인트로 되돌리고 문제 배치 건너뛰기, QK-norm, z-loss |
| 검증 손실만 오름 | 과적합 | 정규화, 조기 종료 |

**"작은 배치 하나에 과적합되는가"**는 가장 강력한 디버깅 도구입니다. 샘플 몇 개로 손실을 0 근처까지 못 내리면 모델이나 코드에 버그가 있습니다.

## 혼합 정밀도 { .l2 }

| 형식 | 지수 비트 | 가수 비트 | 범위 | 쓰임 |
|---|---|---|---|---|
| FP32 | 8 | 23 | 넓음, 정밀 | 마스터 가중치, 옵티마이저 상태, 누적 |
| FP16 | 5 | 10 | 좁음 (최대 65,504) | 손실 스케일링 필요 |
| BF16 | 8 | 7 | FP32와 같은 범위, 덜 정밀 | 현재 학습의 표준 |
| FP8 (E4M3/E5M2) | 4/5 | 3/2 | 매우 좁음 | 최신 GPU의 대형 학습·추론 |

BF16은 FP32와 지수 비트가 같아 오버플로 걱정이 거의 없어, FP16 시절의 손실 스케일링이 필요 없습니다.

## 코드로 보기 { .l2 }

초기화 크기에 따라 50층을 지난 활성값의 크기가 어떻게 되는지, 그리고 잔차 연결이 있으면 어떻게 달라지는지 봅니다.

```python
import torch

torch.manual_seed(0)
n, depth = 256, 50
x0 = torch.randn(1000, n)

def run(std, residual=False):
    x = x0.clone()
    for _ in range(depth):
        W = torch.randn(n, n) * std
        h = torch.relu(x @ W)
        x = x + h / depth ** 0.5 if residual else h
    return x.std().item()

print(f"너무 작게 (0.01)   : {run(0.01):.2e}")       # 소실
print(f"너무 크게 (0.2)    : {run(0.2):.2e}")        # 폭주
print(f"He (√(2/n)={(2/n)**0.5:.3f}): {run((2 / n) ** 0.5):.2e}")   # 유지
print(f"잔차 + He          : {run((2 / n) ** 0.5, residual=True):.2e}")

# 기울기 클리핑: 전체 노름을 1.0으로 제한
p = torch.nn.Parameter(torch.randn(10)); p.grad = torch.randn(10) * 100
before = torch.nn.utils.clip_grad_norm_([p], max_norm=1.0)
print(f"클리핑 전 노름 {before:.1f} → 후 {p.grad.norm():.2f}")
```

표준편차 0.01이면 50층 뒤 값이 0(소실), 0.2면 \(10^{17}\)(폭주)이 됩니다. He 초기화는 50층 뒤에도 1 근처(약 0.6)를 유지합니다. 잔차 연결을 쓰면 층마다 값이 더해져 서서히 커지는데(약 8), 이것이 Pre-LN 트랜스포머가 블록마다 입력을 다시 정규화하는 이유입니다.

??? expert "전문가 노트 — 대형 모델의 불안정성, z-loss, μP, 결정성"
    **어텐션 로짓 폭주.** 큰 학습률에서 Q·K의 크기가 커지며 어텐션 softmax가 원-핫으로 포화하고 손실이 급등하는 현상이 보고됐습니다(Dehghani 외 2023, ViT-22B). Q와 K에 LayerNorm을 거는 **QK-norm**이 효과적이며 여러 최신 모델이 채택했습니다.

    **출력 로짓 발산과 z-loss.** 출력 softmax의 정규화 상수 \(Z = \sum e^{z_i}\)가 커지면 수치 문제가 생깁니다. PaLM은 \(10^{-4}\cdot\log^2 Z\)를 손실에 더해 log Z를 0 근처로 붙잡았습니다. Gemma 2는 로짓에 \(c\cdot\tanh(z/c)\) 형태의 soft-capping을 썼습니다.

    **작은 모델로 미리 보기.** Wortsman 외(2023)는 대형 모델에서만 나타나는 줄 알았던 불안정성을 작은 모델에 큰 학습률을 줘서 재현할 수 있음을 보였습니다. 학습률에 대한 손실 곡선이 넓고 평평할수록(학습률 민감도가 낮을수록) 안정적인 설계입니다. μP를 쓰면 폭이 달라도 최적 학습률이 거의 유지되어, 작은 모델로 하이퍼파라미터를 찾고 큰 모델로 옮길 수 있습니다.

    **결정성.** 같은 시드로도 GPU 원자적 덧셈 순서, 커널 선택(cuDNN 벤치마크), 분산 통신 순서 때문에 결과가 비트 단위로 달라질 수 있습니다. 완전 재현이 필요하면 `torch.use_deterministic_algorithms(True)`를 쓰되 속도가 느려질 수 있습니다. 대형 학습에서는 손실 급등 시 체크포인트로 돌아가 데이터 순서를 바꿔 재개하는 것이 표준 운영 절차입니다.

## 관련 페이지

- [활성화 함수](activations.md) — 기울기 소실
- [정규화 기법](normalization.md)
- [옵티마이저 (SGD·Adam)](optimizers.md) — 워밍업, β₂
- [CNN](../arch/cnn.md) — ResNet과 잔차 연결
- [양자화와 경량화](../systems/quantization.md) — FP8, INT8
