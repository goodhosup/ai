---
hide:
  - navigation
---

# AI Handbook

**머신러닝 기초부터 딥러닝, 트랜스포머와 LLM, 에이전트, 그리고 AI 반도체와 온디바이스 AI까지.**
처음 공부하는 사람은 개념부터, 현업 엔지니어는 수식과 논문 디테일까지 한 페이지 안에서 단계적으로 읽을 수 있도록 만든 한글 인공지능 핸드북입니다.

## 이 핸드북을 읽는 법

모든 페이지는 같은 틀을 따르고, 난이도를 신호 막대 아이콘으로 구분합니다.

| 표시 | 대상 | 내용 |
|---|---|---|
| <span class="lv lv1">기초</span> | 처음 공부하는 사람, 비전공자 | 개념, 비유, "왜 이렇게 만들었나" — 수식 없이 |
| <span class="lv lv2">중급</span> | 전공자, 개발자, 엔지니어 | 구조, 핵심 수식, 표, 그림, 짧은 코드 |
| <span class="lv lv3">전문가</span> | ML 엔지니어, 연구자, 시스템·반도체 엔지니어 | 유도 과정, 논문 디테일, 하이퍼파라미터, 실무 팁 (접힌 상자) |

!!! spec "논문 박스"
    각 페이지 맨 위의 이 상자에는 그 주제의 출발점이 된 핵심 논문과 연도, 주요 발전 연표가 적혀 있습니다.

    본문을 읽다가 원문을 확인하고 싶을 때 바로 찾아갈 수 있습니다.

약어 위에 마우스를 올리면 풀네임이 툴팁으로 나옵니다. 예: LLM, RAG, MoE

## 섹션

<div class="grid cards" markdown>

-   :material-flag-checkered: **시작하기**

    ---

    AI의 역사, AI·머신러닝·딥러닝·생성형 AI의 관계, 논문 읽는 법, 학습 로드맵

    [:octicons-arrow-right-24: 시작하기](getting-started/index.md)

-   :material-function-variant: **수학 기초**

    ---

    벡터·행렬, 미분과 경사하강법, 확률, 엔트로피, 최적화

    [:octicons-arrow-right-24: 수학 기초](math/index.md)

-   :material-chart-scatter-plot: **머신러닝**

    ---

    회귀, 트리와 앙상블, SVM, 군집, 과적합, 평가 지표, 강화 학습 기초

    [:octicons-arrow-right-24: 머신러닝](ml/index.md)

-   :material-graph: **딥러닝 기초**

    ---

    MLP, 역전파, 활성화 함수, Adam, 정규화, 학습 안정성

    [:octicons-arrow-right-24: 딥러닝 기초](dl/index.md)

-   :material-layers-triple: **신경망 구조**

    ---

    CNN, RNN·LSTM, 어텐션, 트랜스포머, 임베딩, VAE, GAN, 확산 모델

    [:octicons-arrow-right-24: 신경망 구조](arch/index.md)

-   :material-message-text: **LLM**

    ---

    토큰화, 사전 학습과 스케일링 법칙, SFT·LoRA, RLHF·DPO, 프롬프트, 추론 모델, MoE

    [:octicons-arrow-right-24: LLM](llm/index.md)

-   :material-robot: **LLM 응용**

    ---

    RAG, 에이전트와 도구 사용, 멀티모달, 코딩 어시스턴트, 평가와 벤치마크

    [:octicons-arrow-right-24: LLM 응용](apps/index.md)

-   :material-chip: **시스템과 하드웨어**

    ---

    GPU, 분산 학습, KV 캐시와 추론 최적화, 양자화, NPU와 온디바이스 AI, HBM

    [:octicons-arrow-right-24: 시스템과 하드웨어](systems/index.md)

-   :material-scale-balance: **안전과 사회**

    ---

    정렬과 AI 안전, 편향, 저작권과 개인정보, EU AI Act와 AI 기본법

    [:octicons-arrow-right-24: 안전과 사회](society/index.md)

-   :material-code-braces: **실습**

    ---

    PyTorch 환경, MNIST, 작은 트랜스포머 만들기, LLM API, 라즈베리파이에서 로컬 LLM

    [:octicons-arrow-right-24: 실습](practice/index.md)

-   :material-rocket-launch: **최신 동향**

    ---

    2025–2026 모델 지형, 오픈 웨이트 모델, AGI 논쟁

    [:octicons-arrow-right-24: 최신 동향](next/index.md)

-   :material-bookshelf: **레퍼런스**

    ---

    핵심 논문 연표, 수식 모음, 약어 사전, 참고 자료

    [:octicons-arrow-right-24: 레퍼런스](reference/index.md)

</div>

## 자매 핸드북

- [Bluetooth Handbook](https://goodhosup.github.io/bluetooth/) — Classic · LE · LE Audio
- [Cellular Handbook](https://goodhosup.github.io/cellular/) — LTE · 5G NR · 6G
- [Wi-Fi Handbook](https://goodhosup.github.io/wifi/) — IEEE 802.11 · Wi-Fi 6/7/8

!!! note "출처와 저작권"
    본문과 그림은 직접 작성했습니다. 논문은 제목·저자·연도로 출처를 밝히며, 원문의 그림이나 표를 복제하지 않습니다.
    모델 이름과 성능 수치는 빠르게 바뀌므로 각 페이지에 적힌 기준 시점을 확인하세요.
