# AI Handbook — 머신러닝 · 딥러닝 · LLM

머신러닝 기초부터 딥러닝, 트랜스포머와 LLM, 에이전트, AI 반도체와 온디바이스 AI까지 다루는 한글 인공지능 핸드북입니다.
각 페이지는 **기초 → 중급 → 전문가** 순서로 깊어집니다. 기획은 [PLAN.md](PLAN.md)를 보세요.

- 웹 사이트: https://goodhosup.github.io/ai/ (저장소 생성 후)
- 자매 핸드북: [Bluetooth Handbook](https://goodhosup.github.io/bluetooth/) · [Cellular Handbook](https://goodhosup.github.io/cellular/) · [Wi-Fi Handbook](https://goodhosup.github.io/wifi/)

## 로컬 실행

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python tools/build_figures.py   # 그림 생성
.venv/bin/mkdocs serve                    # http://127.0.0.1:8000
```

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `mkdocs.yml` | 사이트 설정과 **전체 목차(nav)** — 페이지 추가는 여기서 시작 |
| `PLAN.md` | 기획안 (독자, 수준, 문체, 목차, 결정 사항) |
| `docs/` | 본문 마크다운 |
| `docs/assets/figures/` | 스크립트로 생성한 SVG 그림 (직접 수정하지 않음) |
| `diagrams/scripts/` | 그림 생성 스크립트 (matplotlib). 공통 색·스타일은 `style.py` |
| `includes/abbreviations.md` | 사이트 전체 약어 툴팁 (`tools/gen_abbreviations.py`가 약어 사전에서 생성) |
| `tools/scaffold.py` | nav에 있는데 파일이 없는 페이지를 스텁으로 생성 |
| `tools/build_figures.py` | 모든 그림 스크립트 실행 |
| `tools/gen_abbreviations.py` | `docs/reference/glossary.md` 표 → 약어 툴팁 파일 생성 |
| `tools/export_wikidocs.py` | 위키독스 책 형식으로 내보내기 (`--render`로 Mermaid PNG) |
| `tools/publish_wikidocs.py` | 위키독스 API로 업로드 (`WIKIDOCS_TOKEN` 환경 변수) |
| `docs/javascripts/diagrams.mjs` | Mermaid 렌더러 (큰 다이어그램 가독성, 다크 모드, 전체 화면) |

## 페이지 작성 규칙

1. 맨 위에 `!!! spec "핵심 논문 · 연도"` 박스로 그 주제의 출발점이 된 논문(제목·저자·연도)과 주요 발전 연표를 적습니다.
2. 본문은 세 단계로 씁니다.
    - `!!! basic "한눈에 보기"` — 개념과 비유 (수식 없이)
    - 본문 `## 제목 { .l2 }` (헤딩 옆에 중급 배지) — 구조, 수식 최소한, 표, 그림, 짧은 PyTorch 코드
    - `??? expert "전문가 노트 — …"` — 접히는 전문가 노트 (유도 과정, 논문 디테일, 하이퍼파라미터, 실무 팁)
3. 끝에는 해당하면 **비교** 표와 **관련 페이지** 링크를 붙입니다.
4. 흐름·절차는 Mermaid 다이어그램으로(라벨에 괄호가 있으면 따옴표로 감쌈), 곡선·분포·구조 그림은 `diagrams/scripts/`의 스크립트로 그립니다.
5. 코드 예제는 실행해 보고 넣습니다. 버전에 민감한 API는 버전을 적습니다.
6. 모델 이름·성능·가격처럼 빨리 바뀌는 내용은 "YYYY년 MM월 기준"을 적고, 아직 검증되지 않은 주장은 단정하지 않습니다.
7. 논문의 그림·문장을 복사하지 않습니다. 논문은 제목·저자·연도(필요하면 arXiv 번호)로만 인용합니다.

## 배포

`main` 브랜치에 push하면 GitHub Actions(`.github/workflows/deploy.yml`)가 그림을 생성하고 사이트를 빌드해 GitHub Pages에 올립니다.
저장소의 **Settings → Pages → Source**를 `GitHub Actions`로 설정하세요.
