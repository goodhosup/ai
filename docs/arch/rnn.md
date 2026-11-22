# RNN과 LSTM

!!! spec "핵심 논문 · 연도"
    Elman, "Finding Structure in Time" (1990) · Bengio·Simard·Frasconi, "Learning long-term dependencies with gradient descent is difficult" (1994) · Hochreiter·Schmidhuber, "Long Short-Term Memory" (1997) · Cho 외, GRU (2014) · Sutskever·Vinyals·Le, "Sequence to Sequence Learning" (2014) · Gu·Dao, "Mamba" (2023) · Beck 외, "xLSTM" (2024)

!!! basic "한눈에 보기"
    문장은 단어의 **순서**가 중요합니다. "개가 사람을 물었다"와 "사람이 개를 물었다"는 완전히 다릅니다. **RNN(순환 신경망)**은 단어를 하나씩 읽으면서, 지금까지 읽은 내용을 **메모장(은닉 상태)** 하나에 요약해 들고 다닙니다. 새 단어를 읽을 때마다 메모장을 고쳐 씁니다.

    문제는 메모장이 작고 계속 덮어쓴다는 점입니다. 긴 문장의 앞부분은 금세 잊힙니다. **LSTM**은 메모장에 **문(게이트)** 세 개를 달아 "무엇을 지울지, 무엇을 새로 적을지, 무엇을 꺼내 쓸지"를 학습으로 정하게 해 기억을 훨씬 오래 유지합니다.

    2014–2017년 번역·음성 인식의 표준이었지만, 한 단어씩 순서대로 처리해야 해서 병렬화가 어렵다는 약점 때문에 트랜스포머에 자리를 넘겼습니다.

## 기본 RNN { .l2 }

\[
\mathbf{h}_t = \tanh(W_h\mathbf{h}_{t-1} + W_x\mathbf{x}_t + \mathbf{b}),\qquad \mathbf{y}_t = W_y\mathbf{h}_t
\]

같은 가중치 \(W_h, W_x\)를 모든 시점에서 재사용합니다. 학습은 시간 축으로 펼쳐 역전파합니다(BPTT).

**기울기 소실·폭주**: 기울기가 T 시점 거슬러 올라가려면 \(W_h\)를 (대략) T번 곱해야 합니다. \(W_h\)의 가장 큰 특잇값이 1보다 작으면 지수적으로 사라지고, 크면 폭주합니다. 그래서 기본 RNN은 수십 단계 이상 떨어진 관계를 배우기 어렵습니다.

## LSTM과 GRU { .l2 }

LSTM은 은닉 상태와 별도로 **셀 상태** \(\mathbf{c}_t\)라는 "컨베이어 벨트"를 둡니다.

\[
\begin{aligned}
\mathbf{f}_t &= \sigma(W_f[\mathbf{h}_{t-1}, \mathbf{x}_t]) &&\text{망각 게이트: 무엇을 지울까}\\
\mathbf{i}_t &= \sigma(W_i[\mathbf{h}_{t-1}, \mathbf{x}_t]) &&\text{입력 게이트: 무엇을 적을까}\\
\tilde{\mathbf{c}}_t &= \tanh(W_c[\mathbf{h}_{t-1}, \mathbf{x}_t]) &&\text{새 후보 내용}\\
\mathbf{c}_t &= \mathbf{f}_t\odot\mathbf{c}_{t-1} + \mathbf{i}_t\odot\tilde{\mathbf{c}}_t &&\text{셀 상태 갱신 (덧셈!)}\\
\mathbf{o}_t &= \sigma(W_o[\mathbf{h}_{t-1}, \mathbf{x}_t]),\quad \mathbf{h}_t = \mathbf{o}_t\odot\tanh(\mathbf{c}_t) &&\text{출력 게이트}
\end{aligned}
\]

셀 상태가 **곱셈이 아니라 덧셈으로** 갱신되기 때문에, 망각 게이트가 1에 가까우면 기울기가 거의 그대로 먼 과거까지 흐릅니다. ResNet의 잔차 연결과 같은 원리입니다.

| 구조 | 게이트 수 | 파라미터 (같은 은닉 크기) | 특징 |
|---|---|---|---|
| 기본 RNN | 0 | 1× | 단순, 장기 의존성 약함 |
| LSTM | 3 + 셀 | 4× | 가장 널리 쓰인 RNN |
| GRU | 2 | 3× | LSTM 단순화, 성능 비슷 |

## 시퀀스-투-시퀀스와 그 한계 { .l2 }

번역은 입력 길이와 출력 길이가 다릅니다. **인코더 RNN**이 원문 전체를 읽어 마지막 은닉 상태 하나(문맥 벡터)로 요약하고, **디코더 RNN**이 그것에서 번역문을 한 단어씩 생성했습니다(2014). 긴 문장을 고정 크기 벡터 하나에 담는 것이 병목이었고, 이를 풀기 위해 **어텐션**이 등장했습니다 → [어텐션](attention.md).

| 항목 | RNN·LSTM | 트랜스포머 |
|---|---|---|
| 학습 병렬화 | 시점마다 순차 → 느림 | 모든 위치 동시에 → 빠름 |
| 먼 위치 사이 경로 | T 단계 | 1 단계 |
| 생성 시 메모리 | 고정 크기 상태 | 문맥 길이에 비례 (KV 캐시) |
| 생성 시 토큰당 계산 | 일정 | 문맥 길이에 비례 |

마지막 두 줄은 RNN이 유리한 점입니다. 이 장점을 살리면서 병렬 학습도 가능한 **상태 공간 모델(SSM)** 계열(Mamba 등)이 트랜스포머의 대안·보완으로 연구되고 있습니다.

## 코드로 보기 { .l2 }

길이 100인 입력에서, 마지막 시점의 출력이 각 시점의 입력에 대해 갖는 기울기 크기를 기본 RNN과 LSTM으로 비교합니다. 이 기울기가 작으면 먼 과거의 입력을 학습에 반영할 수 없습니다.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
T, H = 100, 64
x = torch.randn(16, T, 1, requires_grad=True)

def grad_profile(cell):
    torch.manual_seed(0)
    rnn = cell(1, H, batch_first=True)
    if cell is nn.LSTM:                       # 관례: 망각 게이트 편향을 1로 (기억 유지 쪽으로 시작)
        rnn.bias_ih_l0.data[H:2 * H] = 1.0
    h, _ = rnn(x)
    h[:, -1].sum().backward()                 # 마지막 시점 출력의 기울기를 입력으로 거슬러 보냄
    g = x.grad.abs().mean(dim=(0, 2))         # 시점별 입력 기울기 크기
    x.grad = None
    return g

for cell in (nn.RNN, nn.LSTM):
    g = grad_profile(cell)
    print(f"{cell.__name__:4s} 기울기  t=99: {g[99]:.1e}  t=90: {g[90]:.1e}  t=50: {g[50]:.1e}  t=0: {g[0]:.1e}")
```

기본 RNN은 10걸음만 거슬러 가도 기울기가 100분의 1로 줄고, 처음 시점(t = 0)에서는 \(10^{-21}\)로 사실상 0입니다. LSTM도 줄어들지만 t = 0에서 약 \(10^{-7}\)로 14자릿수 가까이 큽니다. 망각 게이트 편향을 1로 시작하는 것은 "처음에는 기억을 유지하는 쪽으로" 학습을 시작하게 하는 오래된 관례입니다.

??? expert "전문가 노트 — 선형 RNN의 부활과 상태 공간 모델"
    **선형 순환과 병렬 스캔.** 순환이 \(\mathbf{h}_t = A_t\mathbf{h}_{t-1} + B_t\mathbf{x}_t\)처럼 \(\mathbf{h}\)에 대해 **선형**이면, 결합 법칙 덕분에 병렬 스캔(prefix sum)으로 O(log T) 깊이에 계산할 수 있어 학습이 트랜스포머처럼 병렬화됩니다. 추론은 고정 크기 상태로 토큰당 O(1)입니다. S4(2021)는 연속 시간 상태 공간 모델을 이산화해 긴 시퀀스 벤치마크에서 강했고, Mamba(2023)는 \(A_t, B_t\)를 입력에 따라 바꾸는 **선택적** SSM으로 언어 모델링에서 같은 규모 트랜스포머에 근접했습니다.

    **하이브리드.** 순수 SSM은 문맥에서 특정 정보를 정확히 찾아 복사하는 과제(연상 기억, "바늘 찾기")에서 어텐션보다 약하다는 분석이 있어, 최근 대형 모델은 어텐션 층 일부와 SSM·선형 어텐션 층을 섞는 하이브리드 구조를 쓰기도 합니다(Jamba, 일부 2025년 오픈 모델). 긴 문맥에서 KV 캐시를 크게 줄이는 것이 동기입니다 → [긴 문맥과 메모리](../llm/long-context.md).

    **LSTM의 재조명.** xLSTM(2024)은 지수 게이트와 행렬 메모리로 LSTM을 확장해 언어 모델링 성능을 끌어올렸습니다. RWKV는 RNN처럼 추론하고 트랜스포머처럼 학습하는 구조로 오픈소스 생태계를 이루었습니다.

    **기울기 클리핑의 기원.** Pascanu 외(2013)가 RNN 기울기 폭주를 다루며 기울기 노름 클리핑을 제안했고, 이것이 지금 트랜스포머 학습에서도 표준 기법이 되었습니다.

## 관련 페이지

- [어텐션](attention.md) — seq2seq의 병목을 푼 방법
- [트랜스포머](transformer.md)
- [긴 문맥과 메모리](../llm/long-context.md) — SSM·선형 어텐션
- [초기화와 학습 안정성](../dl/training-stability.md) — 기울기 클리핑
