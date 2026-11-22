# 수식 모음

!!! basic "이 페이지의 용도"
    본문에 흩어져 있는 핵심 수식을 한곳에 모았습니다. 각 수식의 의미와 유도는 오른쪽에 연결한 페이지에서 설명합니다. 기호: 굵은 소문자 $\mathbf{x}$는 벡터, 대문자 $W$는 행렬, $\sigma$는 시그모이드, $\eta$는 학습률입니다.

## 수학 기초 { .l2 }

| 이름 | 수식 | 본문 |
|---|---|---|
| 내적과 코사인 유사도 | $\mathbf{a}\cdot\mathbf{b}=\sum_i a_i b_i,\quad \cos\theta=\dfrac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}$ | [벡터·행렬·텐서](../math/linear-algebra.md) |
| 경사하강법 | $\theta \leftarrow \theta - \eta\,\nabla_\theta L(\theta)$ | [미분과 경사하강법](../math/calculus.md) |
| 연쇄 법칙 | $\dfrac{\partial L}{\partial x}=\dfrac{\partial L}{\partial y}\dfrac{\partial y}{\partial x}$ | [역전파](../dl/backprop.md) |
| 베이즈 정리 | $P(A\mid B)=\dfrac{P(B\mid A)\,P(A)}{P(B)}$ | [확률과 통계](../math/probability.md) |
| 엔트로피 | $H(p)=-\sum_x p(x)\log p(x)$ | [정보이론](../math/information.md) |
| 교차 엔트로피 | $H(p,q)=-\sum_x p(x)\log q(x)$ | [정보이론](../math/information.md) |
| KL 발산 | $D_{\mathrm{KL}}(p\,\Vert\,q)=\sum_x p(x)\log\dfrac{p(x)}{q(x)}$ | [정보이론](../math/information.md) |

## 머신러닝 { .l2 }

| 이름 | 수식 | 본문 |
|---|---|---|
| 선형 회귀 (최소제곱) | $\hat{\mathbf{w}}=(X^\top X)^{-1}X^\top\mathbf{y}$ | [선형·로지스틱 회귀](../ml/linear-models.md) |
| 릿지 회귀 | $\min_\mathbf{w}\ \lVert X\mathbf{w}-\mathbf{y}\rVert^2+\lambda\lVert\mathbf{w}\rVert^2$ | [과적합과 정규화](../ml/overfitting.md) |
| 로지스틱 회귀 | $P(y=1\mid\mathbf{x})=\sigma(\mathbf{w}^\top\mathbf{x}+b),\ \ \sigma(z)=\dfrac{1}{1+e^{-z}}$ | [선형·로지스틱 회귀](../ml/linear-models.md) |
| 소프트맥스 | $\mathrm{softmax}(\mathbf{z})_i=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | [정보이론](../math/information.md) |
| SVM (소프트 마진) | $\min\ \tfrac12\lVert\mathbf{w}\rVert^2+C\sum_i\max(0,\,1-y_i(\mathbf{w}^\top\mathbf{x}_i+b))$ | [SVM과 커널](../ml/svm.md) |
| 정밀도·재현율·F1 | $P=\dfrac{TP}{TP+FP},\ R=\dfrac{TP}{TP+FN},\ F_1=\dfrac{2PR}{P+R}$ | [평가 지표와 검증](../ml/evaluation.md) |
| 벨만 방정식 | $Q(s,a)=r+\gamma\max_{a'}Q(s',a')$ | [강화 학습 기초](../ml/reinforcement.md) |

## 딥러닝 { .l2 }

| 이름 | 수식 | 본문 |
|---|---|---|
| 뉴런 | $y=f(\mathbf{w}^\top\mathbf{x}+b)$ | [퍼셉트론과 MLP](../dl/mlp.md) |
| ReLU · GELU | $\mathrm{ReLU}(x)=\max(0,x),\ \ \mathrm{GELU}(x)=x\,\Phi(x)$ | [활성화 함수](../dl/activations.md) |
| 모멘텀 SGD | $\mathbf{v}\leftarrow\beta\mathbf{v}+\nabla L,\ \ \theta\leftarrow\theta-\eta\mathbf{v}$ | [옵티마이저](../dl/optimizers.md) |
| Adam | $m\leftarrow\beta_1 m+(1-\beta_1)g,\ v\leftarrow\beta_2 v+(1-\beta_2)g^2,\ \theta\leftarrow\theta-\eta\dfrac{\hat m}{\sqrt{\hat v}+\epsilon}$ | [옵티마이저](../dl/optimizers.md) |
| LayerNorm | $\mathrm{LN}(\mathbf{x})=\gamma\odot\dfrac{\mathbf{x}-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ | [정규화 기법](../dl/normalization.md) |
| RMSNorm | $\mathrm{RMSNorm}(\mathbf{x})=\gamma\odot\dfrac{\mathbf{x}}{\sqrt{\tfrac1d\sum_i x_i^2+\epsilon}}$ | [정규화 기법](../dl/normalization.md) |
| He 초기화 | $W\sim\mathcal{N}\!\left(0,\ \tfrac{2}{n_{\text{in}}}\right)$ | [초기화와 학습 안정성](../dl/training-stability.md) |

## 신경망 구조와 LLM { .l2 }

| 이름 | 수식 | 본문 |
|---|---|---|
| 합성곱 출력 크기 | $\left\lfloor\dfrac{n+2p-k}{s}\right\rfloor+1$ | [CNN](../arch/cnn.md) |
| 스케일드 닷프로덕트 어텐션 | $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}+M\right)V$ | [어텐션](../arch/attention.md) |
| 트랜스포머 블록 (Pre-LN) | $\mathbf{h}=\mathbf{x}+\mathrm{Attn}(\mathrm{LN}(\mathbf{x})),\ \ \mathbf{y}=\mathbf{h}+\mathrm{FFN}(\mathrm{LN}(\mathbf{h}))$ | [트랜스포머](../arch/transformer.md) |
| VAE 목적 (ELBO) | $\mathbb{E}_{q(z\mid x)}[\log p(x\mid z)]-D_{\mathrm{KL}}(q(z\mid x)\,\Vert\,p(z))$ | [오토인코더와 VAE](../arch/vae.md) |
| GAN 목적 | $\min_G\max_D\ \mathbb{E}[\log D(x)]+\mathbb{E}[\log(1-D(G(z)))]$ | [GAN](../arch/gan.md) |
| 확산 모델 학습 | $\mathbb{E}_{t,x_0,\epsilon}\lVert\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\ t)\rVert^2$ | [확산 모델](../arch/diffusion.md) |
| 언어 모델 손실 | $L=-\dfrac1T\sum_{t}\log p_\theta(x_t\mid x_{<t}),\ \ \mathrm{PPL}=e^{L}$ | [사전 학습과 스케일링 법칙](../llm/pretraining.md) |
| 학습 연산량 | $C\approx 6ND$ (N: 파라미터, D: 토큰) | [사전 학습과 스케일링 법칙](../llm/pretraining.md) |
| Chinchilla 손실 | $L(N,D)=E+\dfrac{A}{N^\alpha}+\dfrac{B}{D^\beta}$ | [사전 학습과 스케일링 법칙](../llm/pretraining.md) |
| LoRA | $W'=W+\dfrac{\alpha}{r}BA,\ \ B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k}$ | [미세 조정](../llm/finetuning.md) |
| RLHF 목적 | $\max_\pi\ \mathbb{E}[r(x,y)]-\beta\,D_{\mathrm{KL}}(\pi\,\Vert\,\pi_{\text{ref}})$ | [선호 학습](../llm/alignment-training.md) |
| DPO 손실 | $-\log\sigma\!\left(\beta\log\dfrac{\pi(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\dfrac{\pi(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)$ | [선호 학습](../llm/alignment-training.md) |
| 온도 샘플링 | $p_i=\dfrac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$ | [프롬프트 설계](../llm/prompting.md) |

## 시스템 · 평가 · 사회 { .l2 }

| 이름 | 수식 | 본문 |
|---|---|---|
| 루프라인 | $\text{성능}=\min(\text{최대 연산},\ \text{산술 강도}\times\text{대역폭})$ | [GPU와 행렬 연산](../systems/gpu.md) |
| KV 캐시 크기 | $2\times L\times n_{kv}\times d_h\times T\times B\times\text{바이트}$ | [추론 최적화](../systems/inference.md) |
| 추측 디코딩 기대 토큰 | $\dfrac{1-\alpha^{\gamma+1}}{1-\alpha}$ | [추론 최적화](../systems/inference.md) |
| 대칭 양자화 | $q=\mathrm{round}(w/s),\ \ s=\dfrac{\max\lvert w\rvert}{2^{b-1}-1}$ | [양자화와 경량화](../systems/quantization.md) |
| 정답률 표준오차 | $\mathrm{SE}=\sqrt{p(1-p)/n}$ | [평가와 벤치마크](../apps/evaluation.md) |
| pass@k 불편 추정 | $1-\dbinom{n-c}{k}\Big/\dbinom{n}{k}$ | [코딩 어시스턴트](../apps/coding.md) |
| Elo 기대 승률 | $E_A=\dfrac{1}{1+10^{(R_B-R_A)/400}}$ | [평가와 벤치마크](../apps/evaluation.md) |
| 대조 손실 (InfoNCE) | $-\log\dfrac{\exp(s_{ii}/\tau)}{\sum_j\exp(s_{ij}/\tau)}$ | [멀티모달](../apps/multimodal.md) |
| 차분 프라이버시 (라플라스) | $\tilde f(D)=f(D)+\mathrm{Lap}(\Delta f/\varepsilon)$ | [저작권과 개인정보](../society/copyright-privacy.md) |
