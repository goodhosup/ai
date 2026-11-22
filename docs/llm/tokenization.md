# 토큰화

!!! spec "핵심 논문 · 연도"
    Gage, "A New Algorithm for Data Compression" (1994, BPE) · Sennrich 외, "Neural Machine Translation of Rare Words with Subword Units" (2016) · Schuster·Nakajima (2012) / Wu 외 (2016), WordPiece · Kudo·Richardson, "SentencePiece" (2018) · Kudo, "Unigram LM" (2018) · Radford 외, GPT-2 (2019, 바이트 수준 BPE) · Petrov 외, "Language Model Tokenizers Introduce Unfairness Between Languages" (2023)

!!! basic "한눈에 보기"
    LLM은 글자를 한 글자씩 보지도, 단어 단위로 보지도 않습니다. 그 중간 크기의 조각인 **토큰**으로 봅니다. "unbelievable"은 "un" + "believ" + "able"처럼 잘리고, 자주 쓰는 단어 "the"는 통째로 토큰 하나입니다.

    - 단어 단위로 자르면 사전이 끝없이 커지고, 처음 보는 단어(신조어, 오타)를 처리할 수 없습니다.
    - 글자 단위로 자르면 문장이 너무 길어져 계산이 비싸집니다.
    - **서브워드 토큰**은 자주 나오는 조각은 크게, 드문 것은 잘게 잘라 둘의 장점을 취합니다.

    토큰 수는 **비용과 속도**에 직결됩니다. API 요금, 문맥 길이 제한, 생성 속도가 모두 토큰 단위입니다. 영어는 대략 단어 1개 ≈ 1.3토큰이지만, 한국어는 같은 내용에 토큰이 더 많이 드는 경우가 많습니다.

## BPE: 가장 흔한 방법 { .l2 }

**바이트 쌍 부호화(BPE)**는 학습 말뭉치에서 가장 자주 붙어 나오는 쌍을 반복해서 합칩니다.

1. 모든 텍스트를 글자(또는 바이트) 단위로 쪼갭니다.
2. 가장 자주 등장하는 인접 쌍(예: "t" + "h")을 찾아 새 토큰 "th"로 합칩니다.
3. 원하는 어휘 크기(예: 10만 개)가 될 때까지 2를 반복합니다.
4. 합친 순서(병합 규칙)를 저장해 두고, 새 텍스트도 같은 순서로 합칩니다.

| 방식 | 특징 | 쓰는 모델 예 |
|---|---|---|
| 바이트 수준 BPE | UTF-8 바이트에서 시작 → 어떤 문자도 처리 가능 (미등록 토큰 없음) | GPT-2 이후 GPT 계열, LLaMA 3 |
| WordPiece | 우도를 가장 많이 올리는 쌍을 합침 | BERT |
| Unigram (SentencePiece) | 큰 후보 사전에서 확률이 낮은 토큰을 제거 | T5, 일부 다국어 모델 |

## 어휘 크기의 트레이드오프 { .l2 }

| 어휘 | 장점 | 단점 |
|---|---|---|
| 작음 (3만) | 임베딩·출력 층이 작음 | 문장이 길어짐 (토큰 수 증가), 다국어에 불리 |
| 큼 (15–25만) | 문장이 짧아짐, 다국어 효율 | 임베딩·출력 층이 커짐, 드문 토큰 학습 부족 |

최근 LLM은 다국어·코드 효율을 위해 10만–25만 개 어휘를 많이 씁니다(2026년 기준).

## 한국어와 토큰 { .l2 }

한글은 UTF-8에서 글자 하나가 3바이트입니다. 학습 데이터에 한국어가 적으면 병합 규칙이 한국어 조각을 충분히 만들지 못해, 글자 하나가 여러 토큰으로 쪼개지기도 합니다. 그 결과 같은 내용을 처리할 때 영어보다 **토큰이 더 들고, 비용도 더 들고, 문맥 창도 빨리 찹니다**. 다국어 데이터를 늘린 최근 토크나이저와 한국어 특화 모델은 이 격차를 크게 줄였습니다.

## 특수 토큰과 대화 형식 { .l2 }

| 토큰 예 | 역할 |
|---|---|
| 시작 토큰 (BOS, 예: `<s>`) | 문서 시작 |
| 종료 토큰 (EOS, 예: `</s>`) | 생성 종료 |
| 역할 표시 (system·user·assistant) | 대화에서 누가 말하는지 구분 (채팅 템플릿) |
| 도구 호출·사고 구간 표시 | 함수 호출 JSON, 추론 과정의 시작·끝 |

모델마다 채팅 템플릿이 달라, 오픈 모델을 쓸 때는 반드시 그 모델의 템플릿을 적용해야 합니다.

## 코드로 보기 { .l2 }

BPE 학습을 직접 구현해, 작은 말뭉치에서 병합 규칙을 배우고 새 단어를 토큰화합니다.

```python
from collections import Counter

corpus = "low low low lower lowest newer newer newest widest wider".split()
words = Counter(tuple(w) + ("</w>",) for w in corpus)     # 단어 끝 표시

def merge_step(words):
    pairs = Counter()
    for w, n in words.items():
        for a, b in zip(w, w[1:]):
            pairs[a, b] += n
    best = max(pairs, key=pairs.get)
    new = {}
    for w, n in words.items():
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                out.append(w[i] + w[i + 1]); i += 2
            else:
                out.append(w[i]); i += 1
        new[tuple(out)] = n
    return new, best

merges = []
for _ in range(10):
    words, best = merge_step(words)
    merges.append(best)
print("병합 순서:", ["".join(m) for m in merges])

def tokenize(word):
    toks = list(word) + ["</w>"]
    for a, b in merges:                                    # 학습한 순서대로 적용
        i = 0
        while i < len(toks) - 1:
            if (toks[i], toks[i + 1]) == (a, b):
                toks[i:i + 2] = [a + b]
            else:
                i += 1
    return toks
print(tokenize("lowest"), tokenize("newer"), tokenize("slow"))

# 한글은 UTF-8로 글자당 3바이트
print(len("안녕하세요".encode("utf-8")), "바이트,", len("hello".encode("utf-8")), "바이트")
```

학습에 있던 "lowest"는 큰 조각으로, 처음 보는 "slow"는 "s" + 아는 조각으로 나뉩니다. 바이트 수준 BPE에서는 어떤 문자열도 최악의 경우 바이트 단위로 표현되므로 "모르는 토큰"이 없습니다.

??? expert "전문가 노트 — 토큰화가 만드는 이상 현상"
    **글자 세기와 철자.** "strawberry에 r이 몇 개?" 같은 질문을 LLM이 틀리던 이유는 모델이 글자가 아니라 "str", "aw", "berry" 같은 토큰을 보기 때문입니다. 숫자도 토크나이저에 따라 "12345"가 "123" + "45"처럼 불규칙하게 잘려 산술이 어려워지므로, 최근 토크나이저는 숫자를 한 자리씩(또는 세 자리씩 일정하게) 자르는 규칙을 넣습니다.

    **글리치 토큰.** 토크나이저 학습 말뭉치에는 많았지만 모델 학습 데이터에는 거의 없던 토큰(예: 특정 사용자명)은 임베딩이 거의 학습되지 않아 이상한 출력을 유발했습니다(GPT-3의 " SolidGoldMagikarp" 사례). 토크나이저와 모델을 같은 분포의 데이터로 학습해야 하는 이유입니다.

    **공백과 프롬프트 경계.** "Hello"와 " Hello"(앞 공백)는 다른 토큰입니다. 프롬프트가 공백으로 끝나면 모델이 평소와 다른 토큰 경계에서 생성을 시작해 품질이 떨어질 수 있습니다(token healing 기법이 이를 보정).

    **언어 간 불공정.** Petrov 외(2023)는 같은 문장을 번역했을 때 언어에 따라 토큰 수가 최대 10배 이상 차이 남을 보였습니다. API 요금, 응답 속도, 사용 가능한 문맥 길이가 언어에 따라 달라지는 구조적 불평등입니다.

    **토크나이저 없는 모델.** 바이트를 직접 입력하되 내부에서 동적으로 묶는 구조(MegaByte, Byte Latent Transformer 2024)가 연구되고 있습니다. 토큰화의 이상 현상을 없애지만 계산 효율을 맞추는 것이 과제입니다.

## 관련 페이지

- [임베딩과 표현 학습](../arch/embeddings.md) — 토큰 ID → 벡터
- [정보이론](../math/information.md) — 토크나이저와 손실 비교 (bits per byte)
- [LLM API 쓰기](../practice/llm-api.md) — 토큰 단위 요금
