# 딥러닝 기초

이 섹션은 신경망이 무엇이고 어떻게 학습되는지를 다룹니다. 뉴런과 층, 역전파로 기울기를 구하는 법, 활성화 함수와 옵티마이저의 선택, 깊은 신경망을 안정적으로 학습시키는 정규화와 초기화 기법까지, CNN이든 트랜스포머든 모든 구조가 공유하는 부품입니다.

| 페이지 | 핵심 질문 |
|---|---|
| [퍼셉트론과 MLP](mlp.md) | 뉴런 하나는 무엇을 하고, 층을 쌓으면 왜 아무 함수나 흉내 낼 수 있나? |
| [역전파](backprop.md) | 수십억 개 파라미터의 기울기를 어떻게 한 번에 구하나? |
| [활성화 함수](activations.md) | 비선형 함수는 왜 필요하고, 시그모이드에서 ReLU·GELU·SwiGLU로 왜 바뀌었나? |
| [옵티마이저 (SGD·Adam)](optimizers.md) | 모멘텀과 적응형 학습률은 무엇을 고치고, AdamW와 학습률 스케줄은 왜 표준이 됐나? |
| [정규화 기법 (BatchNorm·LayerNorm·Dropout)](normalization.md) | 층 사이 값의 분포를 맞추면 왜 학습이 쉬워지고, 트랜스포머는 왜 LayerNorm을 쓰나? |
| [초기화와 학습 안정성](training-stability.md) | 기울기 소실·폭주, 손실 급등, NaN은 왜 생기고 어떻게 막나? |

!!! basic "PyTorch 학습 루프의 뼈대"
    이 섹션의 모든 내용은 아래 다섯 줄 어딘가에 들어갑니다.

    ```python
    for x, y in loader:                 # 1. 미니배치 가져오기
        loss = loss_fn(model(x), y)     # 2. 순전파 + 손실 (MLP, 활성화, 정규화)
        optimizer.zero_grad()
        loss.backward()                 # 3. 역전파 (기울기 계산)
        optimizer.step()                # 4. 파라미터 업데이트 (SGD, Adam)
    ```
