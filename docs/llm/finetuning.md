# 미세 조정 (SFT·LoRA)

!!! spec "핵심 논문 · 연도"
    Howard·Ruder, "ULMFiT" (2018) · Wei 외, "Finetuned Language Models Are Zero-Shot Learners" (2021, FLAN) · Ouyang 외, "Training language models to follow instructions with human feedback" (2022, InstructGPT) · Hu 외, "LoRA: Low-Rank Adaptation of LLMs" (2021) · Dettmers 외, "QLoRA" (2023) · Zhou 외, "LIMA: Less Is More for Alignment" (2023) · Liu 외, "DoRA" (2024)

!!! basic "한눈에 보기"
    사전 학습을 마친 모델은 아는 것은 많지만 **대화할 줄 모릅니다**. "프랑스의 수도는?"이라고 물으면 답 대신 "독일의 수도는? 이탈리아의 수도는?"처럼 문제집을 이어 쓸 수도 있습니다. 인터넷 글을 이어 쓰는 법만 배웠기 때문입니다.

    **미세 조정(fine-tuning)**은 이 박식한 모델에게 **일하는 방식**을 가르칩니다.

    - **지도 미세 조정(SFT)**: "질문 → 모범 답변" 예시 수천–수십만 개로 추가 학습합니다. 질문에 답하고, 지시를 따르고, 정해진 형식으로 말하게 됩니다.
    - **LoRA**: 모델 전체를 고치는 대신 **작은 보조 행렬만** 학습합니다. 수십억 개 파라미터 중 0.1–1%만 학습하므로 GPU 한 장으로도 가능하고, 결과물도 수십 MB짜리 "어댑터 파일"이라 바꿔 끼우기 쉽습니다.

## SFT { .l2 }

데이터는 (프롬프트, 응답) 쌍이고, 손실은 사전 학습과 같은 다음 토큰 예측이되 **응답 부분의 토큰에만** 계산합니다(프롬프트 부분은 마스킹).

| 데이터 종류 | 예 | 가르치는 것 |
|---|---|---|
| 지시 따르기 | "다음 글을 세 줄로 요약해 줘" → 요약 | 과제 수행 |
| 대화 | 여러 턴의 사용자-어시스턴트 대화 | 문맥 유지, 말투 |
| 도구 사용 | 질문 → 함수 호출 JSON → 결과 → 답 | 에이전트 행동 |
| 안전 | 위험한 요청 → 적절한 거절·대안 | 안전한 행동 |
| 도메인 | 사내 문서 형식, 의료·법률 문체 | 특화 |

LIMA(2023)는 잘 고른 1,000개 예시만으로도 상당한 대화 능력이 생긴다고 보고했습니다. **양보다 질**이 중요하며, 미세 조정은 새 지식을 넣기보다 **이미 아는 것을 꺼내 쓰는 형식**을 가르친다는 해석이 많습니다.

## 언제 미세 조정을 하나 { .l2 }

| 목표 | 먼저 시도할 것 | 미세 조정이 맞는 경우 |
|---|---|---|
| 최신·사내 지식 반영 | RAG (검색해서 넣기) | 거의 없음 (지식 주입은 RAG가 효율적) |
| 출력 형식·말투 고정 | 프롬프트 + 예시 | 프롬프트가 너무 길어지거나 일관성이 부족할 때 |
| 특정 과제 정확도 | 프롬프트, 더 큰 모델 | 데이터가 충분하고 작은 모델로 비용을 줄이고 싶을 때 |
| 지연·비용 절감 | 작은 모델 + 프롬프트 | 큰 모델의 출력을 작은 모델에 증류 |

## LoRA { .l2 }

미세 조정 중 가중치 변화량 \(\Delta W\)가 **저랭크**라고 가정하고, 원래 가중치 \(W_0\)는 고정한 채 두 개의 얇은 행렬만 학습합니다.

\[
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}BA,\qquad B\in\mathbb{R}^{d\times r},\; A\in\mathbb{R}^{r\times k},\; r \ll d
\]

| 항목 | 전체 미세 조정 | LoRA (r = 16) |
|---|---|---|
| 학습 파라미터 (4096×4096 행렬 하나) | 16.8M | 131K (0.78%) |
| 옵티마이저 메모리 | 전체 파라미터의 12바이트/개 | 어댑터만 |
| 저장 | 모델 전체 사본 | 어댑터 수십–수백 MB |
| 추론 속도 | 그대로 | 병합하면 그대로 (\(W_0 + BA\)를 미리 더함) |
| 성능 | 기준 | 대부분의 과제에서 비슷 |

B를 0으로 초기화하므로 학습 시작 시점에는 원래 모델과 똑같이 동작합니다.

**QLoRA**는 고정된 \(W_0\)를 4비트로 양자화해 메모리를 더 줄입니다. 65B 모델을 48 GB GPU 한 장에서 미세 조정할 수 있게 했습니다.

## 코드로 보기 { .l2 }

선형 층에 LoRA를 직접 붙여, 학습 파라미터 수와 "처음에는 원래 출력과 같다"는 성질을 확인합니다.

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                      # 원래 가중치 고정
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # 0 초기화
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merge(self):                                     # 추론용: W0 + scale·BA
        self.base.weight.data += self.scale * self.B @ self.A
        return self.base

torch.manual_seed(0)
base = nn.Linear(4096, 4096, bias=False)
lora = LoRALinear(base, r=16)
x = torch.randn(2, 4096)
print("처음 출력이 원래와 같은가:", torch.allclose(lora(x), base(x)))
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
total = sum(p.numel() for p in lora.parameters())
print(f"학습 파라미터 {trainable:,} / 전체 {total:,} ({100 * trainable / total:.2f}%)")

with torch.no_grad():
    lora.B.normal_(0, 0.01)                              # 학습했다고 치고
    y = lora(x)
    merged = lora.merge()
print("병합 후 결과 동일:", torch.allclose(merged(x), y, atol=1e-4))
```

실무에서는 Hugging Face PEFT 라이브러리로 어텐션·FFN의 선형 층 전체에 한 번에 붙입니다.

??? expert "전문가 노트 — 하이퍼파라미터, 망각, 채팅 템플릿, 모델 병합"
    **하이퍼파라미터.** LoRA의 r은 8–64, α는 r의 1–2배, 학습률은 전체 미세 조정보다 10배 정도 큰 1e-4–2e-4가 흔한 출발점입니다. 어텐션의 Q·V에만 붙이던 초기 관행보다 **모든 선형 층**에 붙이는 것이 대체로 낫습니다. rsLoRA는 스케일을 \(\alpha/\sqrt{r}\)로 바꿔 r이 클 때의 학습을 안정시키고, DoRA는 가중치를 크기와 방향으로 나눠 방향만 LoRA로 학습합니다.

    **치명적 망각.** 좁은 데이터로 많이 학습하면 원래 능력(일반 지식, 다른 언어, 안전 행동)이 손상됩니다. 학습률을 낮추고, 에폭을 1–3으로 제한하고, 일반 데이터를 섞고(replay), LoRA처럼 변화량을 제한하면 줄어듭니다. "LoRA는 덜 배우고 덜 잊는다"(Biderman 외 2024)는 관찰이 있습니다. 소량의 미세 조정만으로도 안전 정렬이 무너질 수 있다는 연구(Qi 외 2023)는 오픈 모델 배포에서 중요한 고려 사항입니다.

    **채팅 템플릿과 손실 마스킹.** 학습과 추론에서 같은 템플릿(역할 토큰, 줄바꿈, EOS)을 써야 합니다. 응답 끝의 EOS 토큰을 손실에 포함하지 않으면 모델이 멈추지 않고 계속 생성합니다. 여러 샘플을 하나의 긴 시퀀스로 이어 붙이는 패킹(packing)은 효율적이지만, 샘플 사이에 어텐션이 새지 않도록 문서 경계 마스크를 쓰는 것이 좋습니다.

    **모델 병합.** 같은 기반 모델에서 따로 미세 조정한 모델들의 가중치를 평균하거나(model soup), 변화량끼리 더하는(task arithmetic, TIES, DARE) 방법으로 학습 없이 능력을 합칠 수 있습니다. 오픈 모델 커뮤니티에서 널리 쓰입니다.

## 관련 페이지

- [선호 학습 (RLHF·DPO)](alignment-training.md) — SFT 다음 단계
- [벡터·행렬·텐서](../math/linear-algebra.md) — 저랭크 근사
- [양자화와 경량화](../systems/quantization.md) — QLoRA의 4비트
- [RAG](../apps/rag.md) — 지식 주입은 미세 조정보다 RAG
