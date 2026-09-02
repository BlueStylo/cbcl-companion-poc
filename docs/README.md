# docs

코드 밖에 남겨야 하는 것만 둡니다. 실행 방법·구조·평가 결과는 루트 README에 있습니다.

## 결정 기록 (`decisions/`)

설계 결정을 ADR 형식(맥락 / 결정 / 근거 / 대안 / 결과)으로 짧게 남깁니다. README와 코드에서
확인할 수 있는 사실만 적고, 번호는 결정 순서입니다.

| 번호 | 결정 |
|---|---|
| [0001](decisions/0001-single-call-and-three-areas.md) | 태스크당 단일 호출 구조와 결정론·LLM·검증의 3영역 분리 |
| [0002](decisions/0002-fail-closed-block-regen-and-fallback.md) | fail-closed: 블록 단위 재생성과 안전 문구 폴백 |
| [0003](decisions/0003-sem-symmetric-band.md) | SEM 대칭 밴드 채택, 비대칭 완화안 기각 |
| [0004](decisions/0004-g3-numeric-equivalence.md) | G3 에코 t_score의 수치 동치 정의 정확화 (비교 완화 아님) |
| [0005](decisions/0005-reduce-llm-scope.md) | LLM 범위 축소: 척도별 해설은 고정 문구, LLM은 인용이 필요한 곳만 |
| [0006](decisions/0006-harness-metrics-redefinition.md) | 검증 지표 재정의: 커버리지 2단, FP 게이트, 시드 수 assert |

새 결정은 다음 번호로 파일을 추가하고 이 표에 한 줄을 더합니다. 규칙·지표·프롬프트 계약을
바꾸는 PR은 해당 ADR 번호를 본문에 적습니다.

## 예시 산출물

mock 모드 산출물(HTML)은 `examples/` 아래에 커밋합니다. 아직 없다면 #10 이 그 작업이며,
커밋되면 이 절을 링크로 바꿉니다. 그 전까지는 `python main.py --profile <json> --mock` 으로
직접 생성합니다 (`out/`, gitignore).

## 운영 규칙

브랜치·커밋·검증·사람 검토 우선순위는 [CONTRIBUTING.md](../CONTRIBUTING.md), 알려진 한계와
예정 작업은 GitHub 이슈(`known-limitation` / `task` 라벨)에 있습니다.
