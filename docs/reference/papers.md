# 핵심 논문 연표

!!! basic "이 페이지의 용도"
    각 페이지 맨 위 "핵심 논문 · 연도" 상자에 나온 논문 가운데 분야의 방향을 바꾼 것들을 시간순으로 모았습니다. 제목을 검색하면 대부분 arXiv나 학회 사이트에서 무료로 읽을 수 있습니다. 논문 읽는 법은 [논문 읽는 법](../getting-started/reading-papers.md)을 참고하세요.

## 1943–2011: 토대 { .l2 }

| 연도 | 논문·사건 | 의미 | 본문 |
|---|---|---|---|
| 1943 | McCulloch·Pitts, 신경 세포의 논리 계산 | 인공 뉴런 모델의 시작 | [MLP](../dl/mlp.md) |
| 1950 | Turing, "Computing Machinery and Intelligence" | 튜링 테스트, "기계가 생각할 수 있는가" | [AGI 논쟁](../next/agi.md) |
| 1958 | Rosenblatt, 퍼셉트론 | 학습하는 최초의 신경망 | [MLP](../dl/mlp.md) |
| 1986 | Rumelhart·Hinton·Williams, 역전파 | 다층 신경망 학습의 표준 | [역전파](../dl/backprop.md) |
| 1992 | Watkins·Dayan, Q-learning | 가치 기반 강화 학습 | [강화 학습 기초](../ml/reinforcement.md) |
| 1995 | Cortes·Vapnik, SVM | 마진 최대화와 커널 | [SVM](../ml/svm.md) |
| 1997 | Hochreiter·Schmidhuber, LSTM | 장기 의존성 문제 해결 | [RNN과 LSTM](../arch/rnn.md) |
| 1998 | LeCun 외, LeNet-5 | 실용 CNN, 손글씨 인식 | [CNN](../arch/cnn.md) |
| 2001 | Breiman, 랜덤 포레스트 | 앙상블의 표준 | [결정 트리와 앙상블](../ml/trees.md) |
| 2006 | Hinton 외, 심층 신뢰 신경망 | 딥러닝 부활의 신호탄 | [AI의 역사](../getting-started/history.md) |
| 2009 | Deng 외, ImageNet | 대규모 데이터셋의 힘 | [AI의 역사](../getting-started/history.md) |
| 2010 | Glorot·Bengio, 초기화 / Nair·Hinton, ReLU | 깊은 망 학습의 실마리 | [초기화와 학습 안정성](../dl/training-stability.md) |

## 2012–2019: 딥러닝 혁명 { .l2 }

| 연도 | 논문 | 의미 | 본문 |
|---|---|---|---|
| 2012 | Krizhevsky·Sutskever·Hinton, AlexNet | GPU 딥러닝이 ImageNet을 압도 | [CNN](../arch/cnn.md), [GPU](../systems/gpu.md) |
| 2013 | Mikolov 외, word2vec | 단어 임베딩 | [임베딩](../arch/embeddings.md) |
| 2013 | Kingma·Welling, VAE / Mnih 외, DQN | 생성 모델과 심층 강화 학습 | [VAE](../arch/vae.md), [강화 학습](../ml/reinforcement.md) |
| 2014 | Goodfellow 외, GAN | 적대적 생성 | [GAN](../arch/gan.md) |
| 2014 | Bahdanau 외, 어텐션 / Sutskever 외, seq2seq | 기계 번역과 어텐션의 등장 | [어텐션](../arch/attention.md) |
| 2014 | Kingma·Ba, Adam / Srivastava 외, Dropout | 표준 옵티마이저와 정규화 | [옵티마이저](../dl/optimizers.md) |
| 2015 | Ioffe·Szegedy, BatchNorm / He 외, ResNet | 매우 깊은 망의 학습 | [정규화 기법](../dl/normalization.md), [CNN](../arch/cnn.md) |
| 2016 | Silver 외, AlphaGo | 바둑에서 인간 최고수를 이김 | [강화 학습](../ml/reinforcement.md) |
| 2017 | Vaswani 외, "Attention Is All You Need" | **트랜스포머** | [트랜스포머](../arch/transformer.md) |
| 2017 | Christiano 외, 사람 선호 기반 RL / Schulman 외, PPO | RLHF의 기초 | [선호 학습](../llm/alignment-training.md) |
| 2017 | Shazeer 외, Sparsely-Gated MoE | 조건부 계산 | [MoE](../llm/moe.md) |
| 2018 | Radford 외, GPT / Devlin 외, BERT | 사전 학습 + 미세 조정 패러다임 | [LLM 개요](../llm/index.md) |
| 2019 | Radford 외, GPT-2 / Shoeybi 외, Megatron / Rajbhandari 외, ZeRO | 대형 언어 모델과 분산 학습 | [분산 학습](../systems/distributed.md) |

## 2020–2023: 스케일링과 생성형 AI { .l2 }

| 연도 | 논문 | 의미 | 본문 |
|---|---|---|---|
| 2020 | Kaplan 외, 스케일링 법칙 / Brown 외, GPT-3 | 규모의 법칙, 문맥 내 학습 | [사전 학습과 스케일링 법칙](../llm/pretraining.md) |
| 2020 | Ho 외, DDPM / Lewis 외, RAG / Dosovitskiy 외, ViT | 확산 모델, 검색 증강, 비전 트랜스포머 | [확산 모델](../arch/diffusion.md), [RAG](../apps/rag.md) |
| 2021 | Radford 외, CLIP / Hu 외, LoRA / Jumper 외, AlphaFold 2 | 멀티모달, 효율적 미세 조정, 과학 응용 | [멀티모달](../apps/multimodal.md), [미세 조정](../llm/finetuning.md) |
| 2021 | Chen 외, Codex·HumanEval / Su 외, RoPE | 코드 생성, 회전 위치 인코딩 | [코딩 어시스턴트](../apps/coding.md) |
| 2022 | Hoffmann 외, Chinchilla | 계산 최적 데이터·모델 비율 | [사전 학습과 스케일링 법칙](../llm/pretraining.md) |
| 2022 | Ouyang 외, InstructGPT / Bai 외, Constitutional AI | 지시 따르기와 정렬 | [선호 학습](../llm/alignment-training.md) |
| 2022 | Wei 외, 사고 사슬 프롬프팅 / Yao 외, ReAct | 단계적 추론, 행동하는 LLM | [추론 모델](../llm/reasoning.md), [에이전트](../apps/agents.md) |
| 2022 | Dao 외, FlashAttention / Rombach 외, Latent Diffusion | 메모리 효율 어텐션, Stable Diffusion | [어텐션](../arch/attention.md) |
| 2022 | Frantar 외, GPTQ / Dettmers 외, LLM.int8() | LLM 양자화 | [양자화](../systems/quantization.md) |
| 2023 | Touvron 외, LLaMA / Rafailov 외, DPO / Kwon 외, vLLM | 오픈 웨이트, 간단한 선호 학습, 효율적 서빙 | [오픈 웨이트](../next/open-weights.md), [추론 최적화](../systems/inference.md) |
| 2023 | Gu·Dao, Mamba / Jimenez 외, SWE-bench | 상태 공간 모델, 실제 코딩 과제 평가 | [긴 문맥](../llm/long-context.md) |

## 2024–2026: 추론과 에이전트 { .l2 }

| 연도 | 논문·보고서 | 의미 | 본문 |
|---|---|---|---|
| 2024 | DeepSeek-AI, DeepSeek-V2·V3 | MLA, 저비용 대형 MoE, FP8 학습 | [MoE](../llm/moe.md), [양자화](../systems/quantization.md) |
| 2024 | Templeton 외, Scaling Monosemanticity | 희소 오토인코더로 대형 모델 해석 | [정렬과 AI 안전](../society/safety.md) |
| 2024 | OpenAI, o1 시스템 카드 / Snell 외, 추론 시 계산 최적화 | 추론 시 계산의 스케일링 | [추론 모델](../llm/reasoning.md) |
| 2024 | Greenblatt 외, Alignment Faking / Hubinger 외, Sleeper Agents | 기만적 행동의 실증 연구 | [정렬과 AI 안전](../society/safety.md) |
| 2025 | DeepSeek-AI, DeepSeek-R1 | 강화 학습만으로 추론 능력, 공개 | [추론 모델](../llm/reasoning.md) |
| 2025 | Kwa 외 (METR), 긴 작업 수행 능력 측정 | 에이전트 능력의 지수적 증가 | [AGI 논쟁](../next/agi.md) |
| 2025 | Bengio 외, 국제 AI 안전 보고서 | 범용 AI 위험에 대한 국제 과학 합의 | [정렬과 AI 안전](../society/safety.md) |
| 2025 | Phan 외, Humanity's Last Exam | 최고 난도 지식·추론 벤치마크 | [평가와 벤치마크](../apps/evaluation.md) |
