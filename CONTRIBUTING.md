# 기여 규칙

3일 규모의 PoC에 맞는 최소 규칙입니다. "검증 체계를 먼저 세우고 사람 검토를
중요한 코드에 집중한다"는 README의 작업 방식을 저장소 운영에 옮긴 것입니다.

## 이력에 관하여

초기 스캐폴드부터 실측 단계까지는 main에 직접 커밋했고, 이후 변경은 PR로 관리합니다.

## 브랜치

- `main`에는 직접 커밋하지 않습니다. 변경은 PR로만 들어갑니다.
- 브랜치 이름은 `feat/` · `fix/` · `chore/` 접두 + 짧은 설명 (예: `feat/g3-korean-numerals`).
- 머지는 squash merge. 커밋 메시지는 PR 제목을 따릅니다.
- 브랜치 보호(PR 필수·CI 통과 필수)는 현재 요금제의 private 저장소에서 설정할 수
  없어 규칙으로만 지킵니다. CI 결과는 PR 화면에서 확인합니다.

## 커밋 메시지

현재 이력에서 쓰는 형식 그대로입니다.

    <type>: <무엇을, 한국어로 한 줄>

`type`은 `feat` / `fix` / `docs` / `chore`. 본문이 필요하면 왜 바꿨는지만 씁니다
(무엇을 바꿨는지는 diff에 있습니다).

## 검증 (PR 전 필수)

1. `pytest -q` 통과.
2. `python harness/run_harness.py --mock` 결론이 "모든 요구 기준 충족" (exit 0).
   CI(`.github/workflows/ci.yml`)가 같은 두 명령을 Python 3.11에서 다시 돌립니다.
3. 가드레일 규칙(`src/guardrails.py`)을 바꾸면 B축 시드(`data/fixtures/seeded/`)를
   반드시 함께 추가하고 `harness/run_harness.py`의 `EXPECTED_B_SEEDS`를 갱신합니다.
   시드 없는 규칙 변경은 검출률 100%가 무엇을 잰 것인지 알 수 없습니다.
4. 프롬프트(`prompts/`)를 바꾸면 `--api` 실측을 1회 이상 다시 하고 결과(모델,
   프로파일, 재생성, 폴백 블록, 잔존 위반)를 PR 본문 또는 README 실측 표에 남깁니다.
   mock 픽스처는 프롬프트 변화를 반영하지 못합니다.

## 사람 검토 우선순위

AI 도구가 작성한 코드는 검증이 먼저입니다. 틀리면 즉시 실패하는 코드(파서, 스키마)는
테스트에 맡기고, **틀려도 에러가 나지 않는 코드**는 사람이 직접 읽습니다.

1. 규칙 사전 - `src/guardrails.py`의 정규식·어휘 사전. 빠진 항목은 조용히 통과합니다.
2. 프롬프트 계약 - `prompts/*.md`. 모델 행동이 바뀌어도 테스트는 모릅니다.
3. 지표 계산 - `harness/run_harness.py`, `src/quality.py`. 분모가 잘못되면 100%가 나옵니다.
4. 가상 데이터 - `data/profiles/`, `data/fixtures/`. 시드가 비면 검출률이 0/0으로 통과합니다
   (그래서 하네스가 시드 총수를 assert 합니다).

## 문서

설계 결정은 `docs/decisions/`에 ADR 형식(맥락 / 결정 / 근거 / 대안 / 결과)으로 남깁니다.
안내는 `docs/README.md`.
