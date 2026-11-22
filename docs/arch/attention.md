# 어텐션

!!! spec "핵심 논문 · 연도"
    Bahdanau·Cho·Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (2014) · Luong 외, "Effective Approaches to Attention-based NMT" (2015) · Vaswani 외, "Attention Is All You Need" (2017, 스케일드 닷-프로덕트·멀티 헤드) · Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019, MQA) · Ainslie 외, "GQA" (2023) · Olsson 외, "In-context Learning and Induction Heads" (2022)

!!! basic "한눈에 보기"
    번역가는 긴 문장을 번역할 때 원문 전체를 한 번 읽고 외워서 쓰지 않습니다. 단어 하나를 쓸 때마다 **원문의 관련 부분을 다시 쳐다봅니다**. "은행"을 번역할 때는 앞의 "돈을 찾으러"를 보고 bank로, "강가의"를 보고 riverbank로 정합니다.

    **어텐션**은 이것을 계산으로 옮긴 것입니다.

    1. 지금 처리 중인 단어가 **질문(Query)**을 던집니다. "내 뜻을 정하는 데 누가 도움이 되지?"
    2. 다른 모든 단어는 자기를 설명하는 **꼬리표(Key)**와 넘겨줄 **내용(Value)**을 가집니다.
    3. 질문과 꼬리표가 잘 맞는 단어일수록 큰 **가중치**를 받고, 내용을 가중치대로 섞어 가져옵니다.

    한 문장 안의 단어들끼리 서로 이렇게 주목하는 것이 **셀프 어텐션**이고, 트랜스포머의 심장입니다.

## 어텐션의 탄생: 번역의 병목 풀기 { .l2 }

RNN 인코더-디코더는 원문 전체를 벡터 하나로 요약했습니다([RNN과 LSTM](rnn.md)). Bahdanau 외(2014)는 디코더가 단어를 생성할 때마다 **인코더의 모든 은닉 상태를 가중 평균**해 쓰게 했습니다.

\[
\alpha_{t,i} = \text{softmax}_i\big(\text{score}(\mathbf{s}_{t-1}, \mathbf{h}_i)\big),\qquad
\mathbf{c}_t = \sum_i \alpha_{t,i}\,\mathbf{h}_i
\]

가중치 \(\alpha\)를 시각화하면 원문과 번역문의 단어 정렬(alignment)이 자연스럽게 드러났습니다. 긴 문장 번역 품질이 크게 좋아졌습니다.

## 점수 함수의 종류 { .l2 }

| 방식 | score(q, k) | 비고 |
|---|---|---|
| 덧셈 (Bahdanau) | \(\mathbf{v}^\top\tanh(W_1\mathbf{q} + W_2\mathbf{k})\) | 작은 MLP |
| 내적 (Luong) | \(\mathbf{q}^\top\mathbf{k}\) | 빠름 |
| 스케일드 내적 (트랜스포머) | \(\mathbf{q}^\top\mathbf{k}/\sqrt{d_k}\) | 행렬 곱 하나로 전부 계산, 현재 표준 |

## Q·K·V: 검색 엔진 비유 { .l2 }

| 요소 | 검색 엔진에서 | 어텐션에서 |
|---|---|---|
| Query | 검색어 | 현재 토큰이 찾는 정보 \(\mathbf{q} = W_Q\mathbf{x}\) |
| Key | 문서의 색인 키워드 | 각 토큰의 "이런 정보가 있어요" \(\mathbf{k} = W_K\mathbf{x}\) |
| Value | 문서 본문 | 실제로 전달할 내용 \(\mathbf{v} = W_V\mathbf{x}\) |
| 결과 | 가장 잘 맞는 문서 하나 | 모든 Value의 **부드러운 가중 평균** |

일반 검색이 1등 하나를 고르는(hard) 반면, 어텐션은 softmax로 **모두를 조금씩** 섞으므로(soft) 미분 가능하고 역전파로 학습됩니다.

## 셀프·교차·인과 어텐션 { .l2 }

| 종류 | Q의 출처 | K·V의 출처 | 쓰는 곳 |
|---|---|---|---|
| 셀프 어텐션 | 같은 시퀀스 | 같은 시퀀스 | 트랜스포머 인코더·디코더 |
| 인과(마스크드) 셀프 어텐션 | 같은 시퀀스 | 같은 시퀀스의 **앞부분만** | GPT 계열 디코더 |
| 교차 어텐션 | 디코더 | 인코더 출력 | 번역(인코더-디코더), 이미지 생성의 텍스트 조건 |

## 멀티 헤드와 KV 공유 { .l2 }

헤드를 여러 개 두면 각 헤드가 다른 관계를 봅니다. 추론 시에는 Key·Value를 저장하는 KV 캐시가 메모리를 많이 차지해, K·V 헤드를 줄이는 변형이 나왔습니다.

| 방식 | Q 헤드 | K·V 헤드 | KV 캐시 |
|---|---|---|---|
| MHA (멀티 헤드) | h | h | 1× |
| GQA (그룹 쿼리) | h | g (예: h/8) | g/h |
| MQA (멀티 쿼리) | h | 1 | 1/h |
| MLA (잠재 어텐션) | h | 저차원 잠재 벡터로 압축 | 크게 감소 |

현재 대부분의 오픈 LLM이 GQA를 쓰고, 일부(DeepSeek 계열)는 MLA를 씁니다 → [추론 최적화](../systems/inference.md).

## 코드로 보기 { .l2 }

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
T, d = 4, 8
x = torch.randn(T, d)                                 # 토큰 4개
Wq, Wk, Wv = (torch.randn(d, d) / d ** 0.5 for _ in range(3))
Q, K, V = x @ Wq, x @ Wk, x @ Wv

scores = Q @ K.T / d ** 0.5                           # (4, 4) 누가 누구에게 얼마나
weights = F.softmax(scores, dim=-1)
out = weights @ V                                     # (4, 8)
print(weights.round(decimals=2))
print("행 합:", weights.sum(-1))

# 인과 마스크
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
causal = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
print(causal.round(decimals=2))

# PyTorch 내장 함수와 비교
ref = F.scaled_dot_product_attention(Q[None], K[None], V[None], is_causal=True)[0]
print(torch.allclose(ref, causal @ V, atol=1e-6))
```

인과 마스크를 걸면 가중치 행렬의 대각선 위쪽이 모두 0이 됩니다. 첫 토큰은 자기 자신만 봅니다.

??? expert "전문가 노트 — 어텐션의 계산 복잡도, 유도 헤드, 어텐션 싱크"
    **복잡도.** 시퀀스 길이 T, 차원 d에서 \(QK^\top\)과 가중합은 각각 \(O(T^2d)\) 연산, 점수 행렬은 \(O(T^2)\) 메모리입니다. T가 d보다 훨씬 커지면(긴 문맥) 이 항이 FFN의 \(O(Td^2)\)를 넘어섭니다. 희소 어텐션(지역 창 + 전역 토큰), 선형 어텐션(커널 근사로 softmax 제거), 슬라이딩 윈도 어텐션이 대안이며, FlashAttention은 수식은 그대로 두고 메모리를 \(O(T)\)로 줄입니다 → [긴 문맥과 메모리](../llm/long-context.md).

    **유도 헤드(induction head).** Olsson 외(2022)는 "…A B … A → B"처럼 앞에서 본 패턴을 이어 쓰는 두 개 헤드의 회로(이전 토큰 헤드 + 유도 헤드)를 찾았고, 이 회로가 학습 중 갑자기 형성되는 시점에 문맥 내 학습(in-context learning) 능력이 함께 급상승함을 보였습니다. 기계적 해석 가능성 연구의 대표 사례입니다.

    **어텐션 싱크.** 많은 LLM에서 첫 토큰(또는 BOS)이 내용과 무관하게 큰 어텐션 가중치를 받습니다. softmax는 합이 1이어야 하므로 "아무 데도 주목하고 싶지 않을 때" 버릴 곳이 필요하기 때문으로 해석됩니다(Xiao 외 2023, StreamingLLM). 긴 문맥 스트리밍에서 첫 토큰을 캐시에 남겨 두면 품질이 유지되는 이유이며, 학습 가능한 싱크 토큰이나 게이트를 넣는 설계도 나왔습니다.

    **어텐션 가중치 ≠ 설명.** 어텐션 가중치가 크다고 그 토큰이 출력에 결정적인 것은 아닙니다(Value 크기, 이후 층의 처리에 따라 다름). 해석에는 기울기 기반 기여도나 개입 실험이 더 믿을 만합니다.

## 관련 페이지

- [트랜스포머](transformer.md) — 어텐션을 쌓은 전체 구조
- [벡터·행렬·텐서](../math/linear-algebra.md) — 내적과 유사도
- [추론 최적화](../systems/inference.md) — KV 캐시, GQA, MLA
- [긴 문맥과 메모리](../llm/long-context.md)
