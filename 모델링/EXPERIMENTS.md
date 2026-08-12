# 실험 로그 (E-번호 순차 기록)

**시작일: 2026-08-12** (각 실험 항목에도 날짜 표기)

`다른참가자/LGAIMERS` 저장소의 `EXPERIMENTS.md` 컨벤션을 참고해 여기서부터 새로 기록한다. 이전까지의 방대한 실험(raw ID, matchup, uncertainty, Trackman 8가지 표현×매핑 3단계, model diversity, calibration, regime weighting, recent_k_pitch_rate 등 40개 이상)은 [`HANDOFF.md`](HANDOFF.md)에 이미 상세히 기록돼 있으니 거기를 참고. 여기는 그 이후 실험만 순번을 매겨 기록한다.

형식: 각 실험은 변경점, rolling OOT(2022/2023/2024, 가중치 0.2/0.3/0.5) 또는 single-split(2019-23→24) 결과, 채택/기각 여부, 커밋(선택)을 남긴다.

기준선: CatBoost baseline(raw 44피처, l2_leaf_reg=15) — 2019-23→24 single-split 734.49, rolling weighted 818.65. 실제 LB 789.23(현재 최종 제출).

---

## E001 (2026-08-12) — game_type × post_regime 교호작용

**출처**: `다른참가자/LGAIMERS`의 model_test.md에서 이 팀의 최대 단일 개선(E3→E4, Brier -0.00197)으로 보고된 피처. 저쪽은 로지스틱+HistGradientBoosting 조합이라 명시적 교호작용이 크게 도움됨.

**변경점**: `game_type_regime = game_type + "_" + (season>=2023)` 를 CatBoost native categorical로 baseline 44개에 추가.

**결과 (single-split 2019-23→24)**: score=696.52, brier=0.248067 (baseline 734.49 대비 -37.97)

**상태**: 기각. CatBoost는 이미 `type="Interaction"` 분석에서 `season×asof_*`를 최상위로 자체 발견하고 있어서(이 세션 초반 확인), 명시적 교호작용이 CatBoost에는 중복 정보 + 노이즈로 작용. 선형/얕은 트리 모델에서 통했던 개선이 CatBoost엔 이미 내장돼 있었던 것으로 해석.

---

## E002 (2026-08-12) — R/M/O(reverse/middle/outside) hazard 분해, standalone

**출처**: `다른참가자/LGAIMERS`의 model_test_2.md, E16-H1. `control_success`를 단일 이진분류로 안 풀고, 원래 target 정의의 3가지 실패유형(reverse/middle/outside)을 hazard(순차조건부) 구조로 분해:
`P(success) = (1-qR)(1-qM)(1-qO)`, qR/qM/qO는 각각 별도 CatBoost 분류기.

**라벨 복원**: 원본 데이터엔 투구별 R/M/O 라벨이 없어서, `asof_pitcher_reverse_rate`/`middle_rate`(이미 baseline 44개 피처에 있음)의 행간 변화량을 역산해서 복원(`rmo_labels.py`). row_id가 시간순이고 `asof_pitcher_n`이 행마다 정확히 +1씩 증가한다는(이 세션에서 확인) 전제를 이용. 복원율 99.89%(이전 실험의 99.85%와 거의 일치), success 행에서 R/M 전부 0인 비율 100%로 정합성 확인.

**결과 (single-split 2019-23→24, standalone)**: score=575.52, brier=0.248369 (baseline 734.49 대비 -158.97). 동일 subset으로 학습한 단일 CatBoost(710.83)보다도 나쁨.

**상태**: standalone은 기각. 다만 이전 실험도 H1 standalone은 기각하고 "innovation"(베이스 모델 예측에 대한 보정치)으로만 채택했음 — 같은 방식으로 E003에서 재시도.

---

## E003 (2026-08-12) — R/M/O hazard를 innovation(보정치)으로

**변경점**: `p_final = p_catboost + beta * (p_rmo_hazard - p_catboost)`, beta 스윕.

**결과 (single-split 2019-23→24)**:

| beta | score |
| --- | --- |
| 0.0 (baseline) | 734.49 |
| 0.1 | **737.63** |
| 0.15 | 737.61 |
| 0.25 | 734.40 |
| 0.5 | 707.87 |
| 1.0 | 575.52 |

**상태**: 소폭 개선(+3.14, beta=0.1)이지만 hand_matchup(+18.40)보다도 작은 폭이라 노이즈 가능성 있음. rolling OOT 재확인 및 제출은 보류 — 더 큰 레버(v2 HistGradientBoosting 복제)를 먼저 확인.
