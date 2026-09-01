# cbcl-companion-poc

CBCL 검사 결과를 보호자 눈높이 해설과 상담 준비 자료로 바꾸는 PoC.

## 원칙

수치는 결정론 파서가, 서술은 LLM이, 임상 판단은 상담사가 합니다.
이 도구는 진단, 처방, 심각성 판정을 하지 않으며, 위반 출력은 가드레일이
차단하고 안전 문구로 대체합니다 (fail-closed).

## 데이터에 관하여

repo의 모든 프로파일은 자작 가상 데이터입니다 (아동 이름도 "김샘플" 등
명백한 가상명). 과제로 제공받은 샘플 보고서와 안내문은 어떤 형태로도
포함되어 있지 않습니다.

## 빠른 시작

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# API 키 없이 mock 모드로 전체 파이프라인 실행
python main.py --profile data/profiles/p2_partial_borderline.json --mock
open out/p2_partial_borderline.html

# 미니 평가 하네스 (mock, 오프라인 완주)
python harness/run_harness.py --mock
```

## 실행 방법

| 명령 | 설명 |
|---|---|
| `python main.py --profile <json> --mock` | 프로파일 1건 → 2페이지 해설 리포트 HTML |
| `python main.py --compare <a.json> <b.json> --mock` | 동점-상이의견 페어 나란히 비교 HTML |
| `python main.py ... --api` | 실 LLM 호출 (.env 필요) |
| `python harness/run_harness.py --mock` | 프로파일 7종 실행 + 지표 표 출력 |
| `pytest` | LLM 없이 도는 결정론 테스트 (파서, 가드레일) |

## 구조

```
data/profiles/    가상 프로파일 7종 (P1~P4, P5a/P5b 페어, A1 위반유도)
data/fixtures/    mock 모드용 고정 LLM 응답 (A1은 위반 응답 시드 포함)
prompts/          프롬프트 계약 전문 (코드 밖 정본)
src/parser.py     입력 검증 + 밴드 라벨 재계산 대조 (fail-closed 1차 관문)
src/renderer.py   종형곡선 SVG (마커, SEM 밴드, 구간 배경) - 순수 문자열 생성
src/llm_client.py OpenAI 호환 클라이언트 + MockLLMClient
src/generator.py  해설/질문 생성 (구조화 JSON 출력)
src/guardrails.py 규칙 5종 + 블록 단위 재생성 + 안전 문구 폴백
src/report_html.py  2페이지 정적 리포트 (1p 관찰자의 렌즈 / 2p 우리 아이 결과)
src/compare_html.py 동점-상이의견 비교 뷰
harness/          미니 평가 하네스
tests/            결정론 단위 테스트
```

## 가드레일 규칙

| ID | 검사 | 위반 시 |
|---|---|---|
| G1 | 진단명 사전 (부정문 포함 과검출 의도) | 블록 재생성 |
| G2 | 심각성 단정 양방향 (심각 단정 + 근거 없는 낙관) | 블록 재생성 |
| G3 | 수치 대조 (본문 수치 + 에코 필드 t_score/band) | 블록 재생성 |
| G4 | 근거 링크 (scale_id / source_scale 실척도 매칭) | 블록 재생성 |
| G5 | 스키마 (구조, 필수 필드, 항목 수) | 블록 재생성 |

재생성은 블록 단위 최대 2회, 그래도 실패하면 사전 작성 안전 문구로
대체합니다. 리포트가 안 나가는 일은 없고, 검증 안 된 문장이 나가는
일도 없습니다.

## 의존성

| 패키지 | 이유 |
|---|---|
| pydantic | 입력/출력 스키마 검증. 밴드 재계산 대조를 모델 검증기로 강제 |
| jinja2 | HTML 템플릿. 문자열 조립보다 화면 구조가 코드에서 읽힘 |
| openai | OpenAI 호환 SDK. base_url 교체만으로 Ollama 겸용. --api 모드에서만 필요 |
| pytest | LLM 호출 없이 도는 결정론 테스트 |

## 사용 모델과 환경 변수

| 변수 | 값 |
|---|---|
| LLM_BASE_URL | TODO (기본 https://api.openai.com/v1, 로컬 http://localhost:11434/v1) |
| LLM_MODEL | TODO (후보: gpt-4o-mini / qwen3:8b) |
| LLM_API_KEY | .env로만 주입 |

<!-- TODO: 실측 후 사용 모델 확정 기재 -->

## 평가 결과

`python harness/run_harness.py --mock` 실행 결과 (2026-09-01, mock 모드):

| 지표 | 값 | 기준 |
|---|---|---|
| 파서 정확도 | 12/12 (100%) — 정상 7건 통과 + 오류 주입 5건 거부 | 요구 100% |
| 가드레일 검출률 | 8/8 (100%) — A1 시드, G1~G5 규칙별 전부 검출 | 요구 100% |
| 폴백 발동률 | 1/119 블록 (0.8%) — A1 attention 블록 (재생성 2회 소진) | 품질 모니터링 지표 |
| 근거 커버리지 | 139/139 (100%) | 요구 100% |
| 최종 출력 잔존 위반 | 0건 | 요구 0건 |

<!-- TODO: --api 모드 실측 (모델별 스키마 준수율 / 폴백률 비교) -->

## 예상 비용

<!-- TODO: 토큰 실측 후 3열 비교표 (gpt-4o-mini / Claude Haiku / 로컬 Ollama), 단가 출처 링크 -->

## 알려진 한계

- 정규식 사전은 미검출 가능성이 있습니다. 하네스로 검출률을 측정하고,
  미검출 유형을 사전에 추가하는 반복이 운영 절차입니다.
- 보호자 단일 보고 기반 선별 검사의 한계는 리포트 안에 고지됩니다.
- 준임상 경계의 해설 어조는 상담사 검수가 필요합니다.
- SEM 밴드의 신뢰도 계수는 예시값(.84)이며 화면에 예시값임을 표기합니다.

## 사용한 AI 도구

<!-- TODO: 실제 사용 내역 기재 -->
