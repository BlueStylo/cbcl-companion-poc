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
| [0007](decisions/0007-explorer-console.md) | 평가자용 탐색 콘솔: 슬라이더 입력, 즉시 결정론 렌더, 템플릿 목 (입력/결과 패널 분리 추가 결정 포함) |
| [0008](decisions/0008-scale-card-order.md) | 척도 카드 정보 순서(결론 → 쉬운 구간 → 수치 → 곡선 → 해설 → 진단 아님 한 줄), 곡선 아래만 채색과 SEM 가로 범위선, 곡선 읽는 법 블록 1회 |
| [0009](decisions/0009-fixed-before-counseling.md) | 상담 전 안내는 고정 문구로: LLM 범위에서 제외 (explain 스키마는 overview 하나, 검증 블록 5개에서 4개) |
| [0010](decisions/0010-llm-questions-only.md) | LLM 범위를 질문으로 축소: 연결 문단·상담사 요약, 이어서 관찰 포인트도 결정론 조립 (prep 1종, 블록 1개, G3 수치 금지와 G10 근거 강제 재정의, G11 질문 방향과 G12 위기 어휘 출력 신설, 입력 위기 사전 활용형 일반화, 미리보기 게이트) |

새 결정은 다음 번호로 파일을 추가하고 이 표에 한 줄을 더합니다. 규칙·지표·프롬프트 계약을
바꾸는 PR은 해당 ADR 번호를 본문에 적습니다.

## 예시 산출물

mock 모드 산출물(HTML)은 [`examples/mock`](../examples/mock)에 커밋되어 있습니다 (프로파일 8종과
동점 페어 비교 뷰). 직접 만들려면 `python main.py --profile <json> --mock` 입니다 (`out/`, gitignore).
실LLM 산출물 [`examples/api`](../examples/api)는 질문 1블록 구조(ADR 0010 보강: LLM은 상담사에게 물어볼 질문만, 관찰 포인트·연결 문단·상담사 요약은 결정론 조립)로 2026-09-02 20:15에 서버에서 재실측한 gemma4:12b 산출물입니다. 6런 전부 첫 시도 통과였고 수치는 README "--api 실측" 절에 있습니다.

## 운영 규칙

브랜치·커밋·검증·사람 검토 우선순위는 [CONTRIBUTING.md](../CONTRIBUTING.md), 알려진 한계와
예정 작업은 GitHub 이슈(`known-limitation` / `task` 라벨)에 있습니다.
