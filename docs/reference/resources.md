# 참고 자료

!!! basic "이 페이지의 용도"
    이 책 다음에 볼 만한 강의, 책, 사이트를 수준별로 골랐습니다. 대부분 무료입니다. 링크와 내용은 2026년 11월 기준이며, 사이트 이름으로 검색하면 쉽게 찾을 수 있습니다.

## 강의 { .l2 }

| 자료 | 수준 | 내용 |
|---|---|---|
| 3Blue1Brown, "Neural Networks" 시리즈 (유튜브) | 입문 | 신경망·역전파·트랜스포머를 애니메이션으로 직관적으로 설명 |
| Andrej Karpathy, "Neural Networks: Zero to Hero" (유튜브) | 입문~중급 | 역전파 엔진부터 GPT까지 직접 코딩. [작은 트랜스포머](../practice/mini-transformer.md)의 다음 단계 |
| fast.ai, "Practical Deep Learning for Coders" | 입문~중급 | 코드 먼저, 이론은 나중에. 실용 중심 |
| Stanford CS231n (비전) / CS224n (자연어) / CS336 (LLM 직접 만들기) | 중급~고급 | 강의 영상·과제 공개 |
| Hugging Face Learn (LLM·에이전트·확산 모델 코스) | 중급 | 오픈 모델을 실제로 다루는 실습 |
| DeepLearning.AI 단기 강좌 | 입문~중급 | 프롬프트, RAG, 에이전트 등 응용 주제별 1–2시간 강좌 |

## 책 { .l2 }

| 책 | 수준 | 특징 |
|---|---|---|
| Aston Zhang 외, *Dive into Deep Learning* (d2l.ai) | 입문~중급 | 무료 온라인, 코드 포함, 한국어 번역판 있음 |
| Simon Prince, *Understanding Deep Learning* (2023) | 중급 | 무료 PDF, 그림이 훌륭한 현대적 교과서 |
| Christopher Bishop·Hugh Bishop, *Deep Learning: Foundations and Concepts* (2023) | 중급~고급 | 확률적 관점의 정통 교과서 |
| Goodfellow·Bengio·Courville, *Deep Learning* (2016) | 고급 | 고전. 트랜스포머 이전이지만 기초 이론에 강함 |
| Sutton·Barto, *Reinforcement Learning: An Introduction* (2판, 2018) | 중급 | 강화 학습의 표준 교과서, 무료 공개 |
| Sebastian Raschka, *Build a Large Language Model (From Scratch)* (2024) | 중급 | GPT를 처음부터 구현 |
| Chip Huyen, *AI Engineering* (2025) | 중급 | 파운데이션 모델로 제품 만들기: 평가, RAG, 에이전트, 비용 |
| 위키독스, 「딥 러닝을 이용한 자연어 처리 입문」 | 입문~중급 | 한국어 무료 온라인 책 |

## 계속 따라가기 { .l2 }

| 사이트 | 용도 |
|---|---|
| arXiv (cs.LG, cs.CL, cs.CV) | 새 논문. Hugging Face Papers의 일별 추천 목록으로 걸러 보기 |
| Epoch AI | 학습 연산량, 하드웨어, 데이터 추세 데이터 |
| METR | 에이전트의 긴 작업 수행 능력 평가 |
| Artificial Analysis | 모델별 품질·속도·가격 독립 비교 |
| LMArena | 사용자 쌍대 비교 기반 리더보드 |
| Hugging Face Hub | 오픈 웨이트 모델·데이터셋, 오픈 모델 리더보드 |
| 각 개발사의 기술 블로그·시스템 카드 | 모델의 능력, 평가, 안전 조치에 대한 1차 자료 |
| 국제 AI 안전 보고서, 각국 AI 안전연구소 | 안전·정책 동향 |
| 과학기술정보통신부, 개인정보보호위원회 | 한국의 AI 정책·가이드라인 |

## 도구 { .l2 }

| 도구 | 용도 | 본문 |
|---|---|---|
| PyTorch, JAX | 딥러닝 프레임워크 | [파이썬과 PyTorch 환경](../practice/setup.md) |
| scikit-learn | 전통적 머신러닝 | [머신러닝](../ml/index.md) |
| Hugging Face Transformers, PEFT, TRL | 사전 학습 모델 사용, LoRA, RLHF·DPO | [미세 조정](../llm/finetuning.md) |
| vLLM, SGLang, llama.cpp, Ollama | LLM 서빙과 로컬 실행 | [추론 최적화](../systems/inference.md), [로컬 LLM 실행](../practice/local-llm.md) |
| 각 회사 SDK (anthropic, openai, google-genai 등) | LLM API | [LLM API 쓰기](../practice/llm-api.md) |
