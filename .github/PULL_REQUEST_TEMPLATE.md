## 변경 이유

<!-- 무엇을 왜 바꾸는지 2~3줄. 설계 결정이 있으면 docs/decisions/ 의 ADR 번호 -->

## 검증

- [ ] `pytest -q` 통과 (로컬)
- [ ] `python harness/run_harness.py --mock` 결론 "모든 요구 기준 충족" (로컬, exit 0)
- [ ] 가드레일 규칙(`src/guardrails.py`)을 바꿨다면: B축 시드(`data/fixtures/seeded/`) 추가 + `EXPECTED_B_SEEDS` 갱신
- [ ] 프롬프트(`prompts/`)를 바꿨다면: `--api` 실측 1회 이상, 결과를 아래 또는 README 실측 표에 기재

실측 (해당 시): <!-- 모델 · 프로파일 · 재생성 · 폴백 블록 · 잔존 위반 · 토큰 · 벽시계 -->

## 위험과 알려진 한계

<!-- 틀려도 에러가 나지 않는 부분. 규칙 사전·프롬프트 계약·지표 계산·가상 데이터를 건드렸다면 무엇을 사람이 읽었는지 -->

## 관련 이슈

<!-- Closes #N / Refs #N -->
