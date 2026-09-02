---
name: 결함
about: 파서·가드레일·하네스·리포트가 기대와 다르게 동작함
title: "[결함] "
labels: bug
---

## 재현

<!-- 명령 그대로. 예: python main.py --profile data/profiles/p2_partial_borderline.json --mock -->

## 기대

## 실제

<!-- 출력·로그 발췌. run_stats.json 의 위반 분포가 있으면 붙여넣기 -->

## 가드레일 관련 여부

- [ ] 가드레일 규칙(G1~G9) 또는 위기 신호 게이트와 관련 있음
- 관련 있다면: 미검출(위반이 통과) / 오검출(정상 문장이 차단) / 폴백 오동작 중 어느 쪽인지
- 미검출이면: 재현 문장을 B축 시드(`data/fixtures/seeded/`)로 추가할 수 있는지
