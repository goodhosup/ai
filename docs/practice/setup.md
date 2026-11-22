# 파이썬과 PyTorch 환경

!!! spec "준비물 · 소요 시간"
    PC(Windows·macOS·Linux) 또는 라즈베리파이 5 · 인터넷 연결 · 디스크 여유 5GB 이상 · 약 30분. GPU는 없어도 이 책의 모든 예제가 CPU에서 돌아갑니다.

!!! basic "한눈에 보기"
    이 책의 코드는 모두 **파이썬**으로 되어 있습니다. 파이썬은 AI 분야의 공용어이고, 그 위에서 두 가지 라이브러리를 주로 씁니다.

    - **NumPy**: 숫자 배열(벡터·행렬) 계산. 이 책의 수학·머신러닝 예제 대부분.
    - **PyTorch**: 신경망을 만들고 학습시키는 라이브러리. 자동 미분과 GPU 계산을 제공합니다.
    - 그 밖에 **scikit-learn**(전통적 머신러닝), **matplotlib**(그래프)을 씁니다.

    처음 설치할 때 가장 흔한 실수는 시스템 파이썬에 패키지를 마구 설치하다가 버전이 꼬이는 것입니다. 그래서 프로젝트마다 **가상환경**이라는 독립된 상자를 만들어 그 안에만 설치합니다.

## 설치 순서 { .l2 }

```mermaid
flowchart LR
    P["1. 파이썬 3.10+ 설치"] --> V["2. 가상환경 만들기<br/>(venv 또는 uv)"] --> T["3. PyTorch 설치<br/>(CPU / CUDA / MPS)"] --> L["4. 나머지 라이브러리"] --> C["5. 확인 코드 실행"]
```

**1. 파이썬.** Windows는 python.org 설치 파일(“Add python.exe to PATH” 체크), macOS는 Homebrew(`brew install python`), Ubuntu·라즈베리파이 OS는 기본 포함된 `python3`를 쓰면 됩니다. 버전은 3.10 이상을 권합니다.

**2. 가상환경과 라이브러리.**

```bash
mkdir ai-practice && cd ai-practice
python3 -m venv .venv                 # 가상환경 만들기
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch numpy scikit-learn matplotlib jupyter
```

빠른 패키지 관리자 **uv**를 쓴다면 `uv venv` 후 `uv pip install ...`로 같은 일을 몇 배 빠르게 할 수 있습니다.

**3. PyTorch 설치 선택.** `pip install torch`만으로 대부분 동작합니다. 환경별 차이는 다음과 같습니다.

| 환경 | 설치 | 장치 이름 |
|---|---|---|
| NVIDIA GPU (Windows/Linux) | pytorch.org의 설치 안내에서 CUDA 버전에 맞는 명령 복사 | `cuda` |
| Apple Silicon Mac | `pip install torch` (MPS 내장) | `mps` |
| CPU만 (일반 PC, 라즈베리파이 5) | `pip install torch` (Linux는 CPU 전용 휠로 용량 절약 가능) | `cpu` |
| 클라우드 노트북 (Colab 등) | 이미 설치되어 있음. 무료 GPU 사용 가능 | `cuda` |

## 코드로 보기 { .l2 }

설치가 끝났으면 아래 코드로 확인합니다. 버전, 사용할 장치, 간단한 행렬 곱이 출력되면 준비 완료입니다.

```python
import sys, platform
import numpy as np, torch

print("Python ", sys.version.split()[0], "|", platform.machine())
print("NumPy  ", np.__version__)
print("PyTorch", torch.__version__)

if torch.cuda.is_available():
    device = "cuda"                                   # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"                                    # Apple Silicon GPU
else:
    device = "cpu"
print("장치   ", device, "| CPU 스레드", torch.get_num_threads())

a = torch.randn(512, 512, device=device)
b = torch.randn(512, 512, device=device)
c = a @ b                                             # 행렬 곱이 되면 준비 완료
print("행렬 곱 결과 크기:", tuple(c.shape), "| dtype:", c.dtype)
```

이 책을 쓴 환경(GPU 없는 리눅스 서버)에서는 `장치 cpu`, `행렬 곱 결과 크기: (512, 512)`가 출력되었습니다. 이후 실습 코드는 `device` 변수를 쓰지 않고 CPU 기준으로 작성했으므로 어디서나 그대로 돌아갑니다. GPU가 있다면 모델과 텐서에 `.to(device)`를 붙여 옮기면 됩니다.

## 작업 방식 { .l2 }

| 도구 | 언제 | 비고 |
|---|---|---|
| Jupyter Notebook / Lab | 실험, 그래프 확인, 학습 | `jupyter lab`으로 실행. 셀 단위로 결과 확인 |
| VS Code + Python 확장 | 코드가 길어질 때, 디버깅 | 노트북 파일도 열 수 있음 |
| 터미널에서 `python file.py` | 반복 실행, 라즈베리파이 원격 작업 | SSH로 접속해 실행 |
| Google Colab 등 | GPU가 필요할 때 | 무료 사용량 제한 있음 |

??? expert "전문가 노트 — 재현성, 의존성 고정, 원격 GPU"
    **재현성.** 결과를 다시 얻으려면 난수 시드를 고정합니다(`torch.manual_seed`, `np.random.default_rng(seed)`). 이 책의 예제도 모두 시드를 고정했습니다. 다만 GPU의 일부 연산은 비결정적이어서 완전히 같은 결과가 필요하면 `torch.use_deterministic_algorithms(True)`를 켭니다(느려질 수 있음). 라이브러리 버전이 달라도 숫자가 조금씩 달라질 수 있으므로, 본문의 출력과 소수점 아래가 다르더라도 경향이 같다면 정상입니다.

    **의존성 고정.** 프로젝트가 안정되면 `pip freeze > requirements.txt` 또는 uv의 `uv lock`으로 버전을 고정해 다른 컴퓨터에서도 같은 환경을 만듭니다. CUDA 버전·드라이버·PyTorch 빌드의 조합이 맞지 않으면 GPU를 인식하지 못하는 문제가 가장 흔하므로, `nvidia-smi`로 드라이버를 확인하고 PyTorch 설치 안내의 조합을 따르세요. 시스템 수준 의존성이 복잡하면 Docker 컨테이너(예: NVIDIA의 PyTorch 이미지)를 씁니다.

    **원격 GPU.** 큰 모델 실습에는 클라우드 GPU 인스턴스를 시간 단위로 빌려 쓸 수 있습니다. SSH 접속 + VS Code 원격 개발 또는 Jupyter 포트 포워딩으로 로컬처럼 작업합니다. 비용 사고를 막기 위해 사용 후 인스턴스 중지를 습관화하세요.

## 관련 페이지

- [손글씨 숫자 분류 (MNIST)](mnist.md) — 첫 신경망 학습
- [벡터·행렬·텐서](../math/linear-algebra.md) — NumPy·PyTorch 텐서의 기초
- [GPU와 행렬 연산](../systems/gpu.md) — GPU가 빠른 이유
- [로컬 LLM 실행](local-llm.md) — 라즈베리파이 환경
