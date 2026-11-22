# LLM API 쓰기

!!! spec "준비물 · 소요 시간"
    [파이썬과 PyTorch 환경](setup.md) · LLM 회사의 API 키 (유료, 사용량 과금) · 약 40분. 관련 이론: [프롬프트 설계](../llm/prompting.md), [에이전트와 도구 사용](../apps/agents.md), [RAG](../apps/rag.md)

!!! basic "한눈에 보기"
    가장 강력한 LLM은 내 컴퓨터에서 돌리기에는 너무 큽니다. 대신 모델 회사의 서버에 **API**로 요청을 보내고 답을 받습니다. 식당에 비유하면 주방(모델)을 직접 차리지 않고 배달 앱(API)으로 주문하는 것입니다.

    API를 쓰는 흐름은 어느 회사든 거의 같습니다.

    1. 회사 콘솔에서 가입하고 **API 키**를 발급받습니다. 키는 비밀번호처럼 다룹니다.
    2. 공식 **SDK**(파이썬 라이브러리)를 설치합니다.
    3. "모델 이름 + 대화 메시지 목록 + 최대 출력 길이"를 담아 요청을 보냅니다.
    4. 답을 받고, 사용한 **토큰 수만큼** 비용을 냅니다.

    이 페이지의 예시는 Anthropic의 Claude API를 쓰지만, 다른 회사의 API도 메시지 목록을 주고받는 구조가 거의 같아 쉽게 옮길 수 있습니다. 모델 이름과 가격은 **2026년 11월 기준**이며 자주 바뀌므로 공식 문서를 확인하세요.

## 준비 { .l2 }

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."    # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
```

API 키를 코드에 직접 쓰거나 깃 저장소에 올리면 안 됩니다. 환경 변수나 `.env` 파일(깃에서 제외)로 관리하세요. SDK는 환경 변수 `ANTHROPIC_API_KEY`를 자동으로 읽습니다.

## 첫 요청 { .l2 }

아래 코드는 API 키가 있어야 실행되므로 이 책의 자동 검증에서는 제외했습니다(요청 형식은 SDK로 확인했습니다).

```py
import anthropic

client = anthropic.Anthropic()                     # ANTHROPIC_API_KEY 환경 변수 사용

response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    system="당신은 친절한 AI 튜터입니다. 한국어로 짧게 답하세요.",
    messages=[{"role": "user", "content": "트랜스포머를 한 문장으로 설명해 줘."}],
)

if response.stop_reason == "refusal":              # 안전 정책으로 거절된 경우
    print("거절됨:", response.stop_details)
else:
    for block in response.content:                 # 응답은 블록 목록 (생각, 텍스트, 도구 호출 등)
        if block.type == "text":
            print(block.text)
print("토큰:", response.usage.input_tokens, "입력 /", response.usage.output_tokens, "출력")
```

| 인자 | 뜻 |
|---|---|
| `model` | 모델 이름. 큰 모델일수록 똑똑하고 비쌈 |
| `max_tokens` | 출력 토큰 상한. 너무 작으면 답이 중간에 잘림 (`stop_reason == "max_tokens"`) |
| `system` | 시스템 프롬프트: 역할, 말투, 규칙 |
| `messages` | `user`와 `assistant`가 번갈아 나오는 대화 기록 |
| `stop_reason` | 끝난 이유: `end_turn`(정상), `max_tokens`(잘림), `tool_use`(도구 호출), `refusal`(거절) |

**API는 기억이 없습니다.** 대화를 이어 가려면 이전 질문과 답을 `messages`에 모두 넣어 다시 보내야 합니다. 그래서 대화가 길어질수록 입력 토큰(=비용)이 늘어납니다.

## 스트리밍: 답을 글자가 나오는 대로 받기 { .l2 }

긴 답을 기다리지 않고 생성되는 대로 화면에 보여 주려면 스트리밍을 씁니다. 긴 출력에서 요청 시간 초과를 피하는 효과도 있습니다.

```py
with client.messages.stream(
    model="claude-opus-5-5",
    max_tokens=64000,
    messages=[{"role": "user", "content": "LLM의 역사를 다섯 단계로 설명해 줘."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final = stream.get_final_message()             # 끝나면 전체 응답과 사용량
```

## 그 밖의 주요 기능 { .l2 }

| 기능 | 하는 일 | 관련 페이지 |
|---|---|---|
| 도구 사용 (함수 호출) | 도구의 이름·설명·JSON 스키마를 주면 모델이 호출 요청을 돌려줌 | [에이전트와 도구 사용](../apps/agents.md) |
| 구조화된 출력 | 지정한 JSON 스키마에 맞는 답만 받기 | 데이터 추출, 분류 |
| 이미지·PDF 입력 | 사진, 차트, 문서를 메시지에 넣기 | [멀티모달](../apps/multimodal.md) |
| 추론 깊이 조절 (effort) | 어려운 문제는 더 오래 생각, 쉬운 문제는 빠르게 | [추론 모델과 사고 사슬](../llm/reasoning.md) |
| 프롬프트 캐싱 | 매번 같은 긴 앞부분(시스템 프롬프트, 문서)을 캐시해 비용·지연 절감 | [추론 최적화](../systems/inference.md) |
| 배치 API | 급하지 않은 대량 요청을 비동기로 싸게 처리 | |

## 코드로 보기 { .l2 }

API 비용은 **토큰 수 × 단가**로 정해지며, 출력 토큰이 입력보다 보통 5배 비쌉니다. 서비스를 만들기 전에 월 비용을 대략 계산해 보는 습관이 중요합니다.

```python
PRICE = {                     # 100만 토큰당 달러 (입력, 출력), 2026년 11월 기준 공개 가격
    "claude-opus-5-5":   (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5":  (0.10,  0.50),
}

def monthly_cost(model, requests_per_day, in_tok, out_tok, cached_frac=0.0, cache_discount=0.9):
    p_in, p_out = PRICE[model]
    in_cost = in_tok * (1 - cached_frac * cache_discount) * p_in / 1e6   # 캐시된 입력은 약 90% 할인
    return (in_cost + out_tok * p_out / 1e6) * requests_per_day * 30

# 사내 문서 Q&A 봇: 하루 2,000건, 질문+검색 문서 6,000토큰 입력, 500토큰 답변
for m in PRICE:
    base = monthly_cost(m, 2000, 6000, 500)
    cached = monthly_cost(m, 2000, 6000, 500, cached_frac=0.5)   # 공통 시스템 프롬프트 절반 캐시
    print(f"{m:18s} 월 ${base:8,.0f}   (프롬프트 캐싱 시 ${cached:8,.0f})")
```

하루 2,000건의 사내 문서 Q&A 봇이면 가장 큰 모델은 월 약 **2,040달러**, 중간 모델은 1,020달러, 가장 작은 모델은 51달러입니다. 입력의 절반을 차지하는 공통 시스템 프롬프트를 캐시하면 각각 약 30% 줄어듭니다(캐시 쓰기 비용은 단순화를 위해 생략). 이 경우 비용의 대부분이 **입력**에서 나오므로, 검색 문서를 줄이거나 캐싱을 쓰는 것이 출력 길이를 줄이는 것보다 효과적입니다. 실제로는 작은 모델로 품질이 충분한지 [평가 세트](../apps/evaluation.md)로 먼저 확인하고 모델을 고릅니다.

??? expert "전문가 노트 — 오류 처리, 거절 대비, 운영 팁"
    **오류 처리.** SDK는 연결 오류, 429(요청 한도 초과), 5xx 오류를 기본 2회 자동 재시도합니다. 그래도 실패하면 예외를 종류별로 나눠 처리합니다: `anthropic.RateLimitError`(잠시 후 재시도, `retry-after` 헤더 확인), `anthropic.BadRequestError`(요청 형식 오류: 고쳐야 함, 재시도 무의미), `anthropic.APIConnectionError`(네트워크). 응답 객체의 `_request_id`를 로그에 남겨 두면 문제를 문의할 때 유용합니다.

    **거절 대비 (서버 측 대체 모델).** 최신 모델은 안전 분류기가 요청을 거절하면 `stop_reason == "refusal"`을 돌려줍니다. Claude API는 거절 시 같은 요청을 다른 모델로 자동 재실행하는 베타 기능을 제공합니다.

    ```py
    response = client.beta.messages.create(
        model="claude-opus-5-5", max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                       # 거절 유형에 따라 대체 모델 자동 선택
        messages=[{"role": "user", "content": "..."}],
    )
    ```

    **운영 팁.** ① 모델 이름은 설정 파일로 빼 두면 새 모델로 바꾸기 쉽습니다. ② 사용자 입력이 섞인 프롬프트에서는 지시와 데이터를 태그 등으로 분명히 구분하세요(프롬프트 인젝션 → [정렬과 AI 안전](../society/safety.md)). ③ 개인정보·기밀을 보내기 전에 회사의 데이터 처리 조건(학습 사용 여부, 보관 기간)을 확인하세요(→ [저작권과 개인정보](../society/copyright-privacy.md)). ④ 응답 품질, 지연, 토큰 사용량을 기록해 두면 비용 최적화와 회귀 점검에 씁니다.

## 관련 페이지

- [프롬프트 설계](../llm/prompting.md) — 좋은 지시문 쓰기
- [RAG](../apps/rag.md) — 문서를 검색해 API에 함께 보내기
- [에이전트와 도구 사용](../apps/agents.md) — 도구 호출 루프
- [로컬 LLM 실행](local-llm.md) — API 없이 내 컴퓨터에서
