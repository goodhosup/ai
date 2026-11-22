# 로컬 LLM 실행 (라즈베리파이 포함)

!!! spec "준비물 · 소요 시간"
    PC·Mac 또는 라즈베리파이 5 (8GB 이상 권장, 16GB면 여유) · 저장 공간 10GB 이상 · 인터넷(모델 내려받기용) · 약 1시간. 관련 이론: [양자화와 경량화](../systems/quantization.md), [추론 최적화](../systems/inference.md), [오픈 웨이트 모델](../next/open-weights.md)

!!! basic "한눈에 보기"
    가중치가 공개된 **오픈 웨이트 모델**을 내려받으면 인터넷 연결 없이 내 컴퓨터에서 LLM을 돌릴 수 있습니다. 비용이 들지 않고, 입력한 내용이 밖으로 나가지 않으며, 원하는 대로 바꿔 볼 수 있습니다.

    가능하게 만든 열쇠는 두 가지입니다.

    - **양자화**: 16비트 가중치를 4비트 안팎으로 줄여 모델 크기를 1/4로 만듭니다. 8B 모델이 16GB에서 약 5GB가 됩니다.
    - **llama.cpp**: CPU에서도 빠르게 돌도록 최적화된 C/C++ 추론 엔진입니다. **Ollama**, LM Studio 같은 쉬운 도구들이 이를 바탕으로 만들어졌습니다.

    라즈베리파이 5에서도 1–4B급 작은 모델은 초당 수 토큰~십여 토큰으로 대화가 됩니다. 크고 똑똑한 모델과 비교할 수는 없지만, 요약, 분류, 간단한 질의응답, 홈 오토메이션 명령 해석 같은 일에는 충분히 쓸 만합니다.

## 방법 1: Ollama (가장 쉬움) { .l2 }

```bash
curl -fsSL https://ollama.com/install.sh | sh   # Linux·라즈베리파이 (macOS·Windows는 설치 파일)
ollama run gemma3:1b                            # 처음 실행 시 모델을 내려받고 대화 시작
ollama run qwen3:1.7b                           # 다른 작은 모델
ollama list                                     # 내려받은 모델 목록
```

Ollama는 백그라운드에서 HTTP 서버(기본 포트 11434)를 띄우므로 파이썬에서도 부를 수 있습니다. 아래 코드는 Ollama가 실행 중이어야 하므로 이 책의 자동 검증에서 제외했습니다.

```py
import requests

r = requests.post("http://localhost:11434/api/chat", json={
    "model": "gemma3:1b",
    "messages": [{"role": "user", "content": "라즈베리파이를 한 문장으로 소개해 줘."}],
    "stream": False,
})
print(r.json()["message"]["content"])
```

## 방법 2: llama.cpp (직접 빌드, 세밀한 제어) { .l2 }

```bash
sudo apt install -y git cmake build-essential libcurl4-openssl-dev
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build --config Release -j4    # 파이 5에서 수 분~십여 분
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF          # Hugging Face에서 GGUF 받아 대화
./build/bin/llama-server -hf ggml-org/gemma-3-1b-it-GGUF --port 8080   # 웹 UI + OpenAI 호환 API
```

모델 파일 형식은 **GGUF**입니다. 같은 모델도 `Q4_K_M`(4비트, 균형), `Q5_K_M`, `Q8_0`(8비트, 거의 무손실) 등 여러 양자화 버전으로 배포됩니다. 메모리가 허락하는 한 큰 모델의 Q4가 작은 모델의 Q8보다 대체로 낫습니다.

## 어떤 모델을 고를까 { .l2 }

| 기기 | 권장 크기 (4비트) | 예 (2026년 11월 기준 공개 모델 계열) |
|---|---|---|
| 라즈베리파이 5 (8GB) | 0.5–3B | Gemma 3 1B, Qwen3 0.6B–1.7B, Llama 3.2 1B·3B |
| 라즈베리파이 5 (16GB) | 최대 7–8B (느림) | Qwen3 4B·8B, Llama 3.1 8B |
| 16–32GB 노트북 | 7–14B, 30B급 MoE | Qwen3 시리즈, Gemma 3 12B, gpt-oss-20b |
| 64GB+ 고메모리 PC·Mac | 30–70B, 대형 MoE | Llama 3.3 70B, Qwen3 대형 MoE |

한국어가 중요하다면 한국어 성능을 강조한 모델(EXAONE, Kanana 등 국내 공개 모델과 다국어 모델)을 직접 비교해 보세요. 모델 목록과 순위는 빠르게 바뀝니다 → [오픈 웨이트 모델](../next/open-weights.md).

## 코드로 보기 { .l2 }

[추론 최적화](../systems/inference.md)에서 본 것처럼 디코딩 속도의 상한은 **메모리 대역폭 ÷ 모델 크기**입니다. 기기별로 어떤 모델이 들어가고 얼마나 빠를지 미리 계산해 봅니다.

```python
def estimate(params_b, bits, bandwidth_gbs, ram_gb, efficiency=0.5, overhead_gb=1.0):
    weights_gb = params_b * bits / 8 * 1.1          # 양자화 블록 스케일 등 약 10% 추가
    fits = weights_gb + overhead_gb <= ram_gb * 0.8   # OS와 KV 캐시 몫을 남김
    tok_s = bandwidth_gbs * efficiency / weights_gb   # 디코딩은 토큰마다 가중치 전체를 읽음
    return weights_gb, fits, tok_s

devices = {"라즈베리파이 5 (8GB)": (17, 8), "노트북 (DDR5, 32GB)": (80, 32), "Max급 Mac (64GB)": (400, 64)}
models = {"1B Q4": (1, 4.5), "3B Q4": (3, 4.5), "8B Q4": (8, 4.5), "32B Q4": (32, 4.5)}
for dev, (bw, ram) in devices.items():
    row = []
    for name, (p, b) in models.items():
        w, fits, t = estimate(p, b, bw, ram)
        row.append(f"{name} {t:5.1f}tok/s" if fits else f"{name}   (메모리 부족)")
    print(f"{dev:20s} | " + " | ".join(row))
```

라즈베리파이 5(LPDDR4X, 이론 대역폭 약 17GB/s)에서는 1B 모델이 초당 약 **14토큰**, 3B는 약 5토큰, 8B는 약 2토큰으로 예상되고, 32B는 메모리에 들어가지 않습니다. 실제 파이 5에서 llama.cpp로 측정한 보고들도 이 범위와 비슷합니다. 노트북은 8B를 초당 8토큰 정도로 쾌적하게 쓰고, 대역폭이 큰 통합 메모리 Mac은 32B도 초당 10토큰 수준입니다. 1B 모델의 수백 토큰/s 같은 값은 대역폭만 본 **상한**이며, 작은 모델에서는 연산과 소프트웨어 오버헤드가 먼저 한계가 됩니다. 프롬프트 처리(프리필)는 연산 한계라 CPU만 있는 파이에서는 긴 문서를 넣으면 첫 토큰까지 수십 초가 걸릴 수 있습니다.

## 라즈베리파이 팁 { .l2 }

| 항목 | 권장 |
|---|---|
| 운영체제 | 64비트 Raspberry Pi OS (Bookworm 이상) |
| 냉각 | 공식 액티브 쿨러 필수. 과열 시 클럭이 떨어져 속도가 크게 줄어듦 |
| 저장 장치 | microSD보다 NVMe SSD(M.2 HAT)가 모델 로딩이 훨씬 빠름 |
| 스레드 | 코어 4개 모두 사용 (`-t 4`). 다른 무거운 작업과 동시에 돌리지 않기 |
| 메모리 | 스왑에 의존하면 극도로 느려짐. 모델 + 1–2GB 여유 확보 |
| 원격 사용 | `llama-server`나 Ollama를 띄우고 같은 네트워크의 PC·폰 브라우저에서 접속 |

??? expert "전문가 노트 — 성능 튜닝, 가속기, 로컬 RAG"
    **튜닝.** 문맥 길이(`-c`)를 필요한 만큼만 잡아 KV 캐시 메모리를 줄이고, KV 캐시 양자화(`--cache-type-k q8_0` 등)로 긴 문맥을 확보합니다. ARM CPU용으로 재배열된 양자화 형식(예: Q4_0의 ARM 최적화 경로)이 파이에서 더 빠를 수 있으므로 같은 모델의 여러 양자화 버전을 벤치마크(`llama-bench`)해 보세요. 작은 초안 모델을 이용한 추측 디코딩도 llama.cpp에서 지원됩니다 → [추론 최적화](../systems/inference.md).

    **가속기.** 파이용 AI HAT(NPU)은 주로 비전 모델(객체 인식 등)용으로, LLM 디코딩 가속에는 제약이 큽니다. PC에서는 NVIDIA GPU(CUDA), Apple Silicon(Metal), AMD·Intel GPU(Vulkan) 백엔드로 레이어 일부 또는 전부를 GPU에 올려(`-ngl`) 크게 가속할 수 있습니다 → [NPU와 온디바이스 AI](../systems/npu.md).

    **로컬 RAG.** 로컬 LLM에 작은 임베딩 모델(Ollama로도 실행 가능)과 벡터 검색을 붙이면 내 문서에 대해 답하는 완전 오프라인 비서를 만들 수 있습니다. 작은 모델일수록 검색 품질과 프롬프트의 명확성이 결과를 크게 좌우합니다 → [RAG](../apps/rag.md). `llama-server`와 Ollama는 OpenAI 호환 API를 제공하므로, 대부분의 LLM 앱 프레임워크에서 주소만 바꿔 로컬 모델을 연결할 수 있습니다.

## 관련 페이지

- [양자화와 경량화](../systems/quantization.md) — Q4, Q8의 의미
- [추론 최적화](../systems/inference.md) — 대역폭이 속도를 정하는 이유
- [오픈 웨이트 모델](../next/open-weights.md) — 어떤 모델이 공개되어 있나
- [LLM API 쓰기](llm-api.md) — 큰 모델은 API로
