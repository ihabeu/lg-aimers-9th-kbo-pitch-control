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

---

## E004 (2026-08-12) — 다른참가자 V14 아키텍처 그대로 재현 (성공, 채택 후보)

**출처**: 사용자가 공유한 ChatGPT 개발 로그에서 받은 `kaggle_kbo_v14_recent_shared_domain_residual_extratrees.ipynb`. 다른참가자가 실제 LB **1044.25629**를 받았다고 보고한 버전. 코드를 거의 그대로(변수명/구조 유지) `모델링/v3_domain_experiments/run_v14_asis.py`로 옮겨서 discover_csv가 우리 `data/train.csv`, `data/test.csv`를 그대로 읽도록만 실행 경로를 맞췄다 — 로직은 손대지 않음.

**아키텍처**:
1. **3-domain 라우팅**: `game_type='F'`는 그대로 F. `game_type='R'` 중에서 `<=2022`의 F에 100% 관여한 팀 하나(우리 데이터에서 team_id=13)를 골라 R_ANCHOR로, 나머지를 R_CORE로 분리. 2024 비중: R_CORE 70.5% / R_ANCHOR 17.7% / F 11.8%.
2. **recent shared base**: 검증 연도 직전 "한 시즌만"으로 학습한 3개 서로 다른 inductive bias 모델(exact-ASOF LightGBM, exact-ASOF Ridge, domain-trend-centered LightGBM)을 단순 equal-blend. 전체 과거 데이터를 안 씀 — 최근 regime만 반영.
3. **residual adapter**: base의 오차(`y - p_base`)를 domain별로 따로 ExtraTrees가 학습. pitcher-disjoint 2-fold cross-fit(3 seed 평균)으로 검증, `p_final = p_base + 0.5 * correction`.

**결과 — 2023 학습 → 2024 검증 (source_year=2023, valid=2024)**:

| 단계 | BSS |
| --- | --- |
| base (equal blend) | **848.35** |
| + residual adapter (3-seed consensus) | **917.91** (+69.56) |

도메인별 gain: R_CORE +29.68, R_ANCHOR +157.15, F +178.14. bootstrap p05=+55.28, prob_positive=1.0. `V14_ARCHITECTURE_PROMISING=True` (원본 gate 그대로 통과).

우리 baseline 비교: single-split(2019-23→24) 734.49, rolling weighted 818.65, 실제 LB 789.23. **848.35(base만)~917.91(adapter까지)은 이 세션에서 재현된 어떤 이전 실험 레시피보다도 크게 앞선 첫 사례**임 — 이전 다른참가자 "안전 앙상블"/HGB 재현은 전부 우리 baseline보다 낮았음(HANDOFF.md 참고).

**결과 — 2022 학습 → 2023 검증 (source_year=2022, valid=2023, dual-fold 안정성 체크)**:

| 단계 | BSS |
| --- | --- |
| base | **-375.86** (F domain -8896.7로 완전 붕괴) |
| + residual adapter | **697.73** (+1073.59, adapter가 대부분 복구) |

`V14_ARCHITECTURE_PROMISING=False` (base_score>=820 게이트 미달).

**해석**: 2022→2023 base 붕괴는 아키텍처 결함이 아니라 F의 2023 regime shift(성공률 0.71→0.47, EDA로 이미 확인) 자체가 2022 데이터만으로는 원천적으로 예측 불가능한 change-point이기 때문. 반대로 2023→2024와 2024→2025(실제 배포)는 학습 소스 연도가 이미 post-2023 regime을 관측한 뒤라 이 문제가 없음 — 그래서 2023→2024 결과가 실제 배포 조건과 훨씬 가깝다고 판단.

**상태**: 채택 후보, 패키징 완료(→ E007에서 v1.1로 튜닝, 최종 제출은 v1.1 기준).

**주의**: 2024는 이 세션에서 이미 여러 실험에 반복 노출된 연도라 완전한 blind holdout은 아님. 그래도 rolling(우리 자체 지표)이 아닌 dual-fold(2022→23, 2023→24) 양쪽 원본 그대로 재현했다는 점에서 이전 실패한 재현들보다 신뢰도가 높음.

---

## E005 (2026-08-12) — game_type='F'(2군) 완전 제거 학습

**배경**: test.csv 5행 전부 `game_type='R'`(1군)로 확인됨. "학습에서 F를 빼면 R 예측이 더 좋아지지 않을까" 가설 검증.

**방법**: baseline CatBoost(44피처, depth=6, l2_leaf_reg=15) 그대로, rolling OOT(2022/23/24, 0.2/0.3/0.5) 각 폴드에서 (A) R+F 같이 학습 후 R행만 평가 vs (B) F를 학습에서 제거하고 R행만 평가. 조기종료도 R행 기준으로 함(평가 대상과 동일 분포).

**결과**:

| | 2022 | 2023 | 2024 | weighted |
| --- | --- | --- | --- | --- |
| A) R+F 학습 | 533.24 | 484.70 | 742.14 | **623.13** |
| B) R only 학습 | 506.56 | 561.77 | 625.68 | **582.68** |

**상태**: 기각. F를 빼면 오히려 -40.45로 더 나쁨 — 직관과 반대로 F 데이터가 R 예측에도 (아마 공유 트리 구조/정규화 효과로) 도움이 됨. `모델링/v3_domain_experiments/exclude_f_domain.py`.

---

## E006 (2026-08-12) — V14 아키텍처에서도 F 완전 제거 재검증

**배경**: 실제 채점 서버 `data/test.csv`에는 퓨처스(F) 행이 없다는 확인(사용자 전달) + E005가 단순 CatBoost baseline이었어서, 더 정교한 V14(3-domain recent-shared-base + residual adapter, E004)에서도 F를 아예 학습에서 빼면 결과가 달라지는지 재검증.

**방법**: `ANCHOR_TEAM_ID`(=13)만 과거 F 이력으로 식별한 뒤(도메인 지식으로 1회 사용), 이후 전 과정에서 `screen`을 R행만 남기고 F를 통째로 드롭. domain3는 R_CORE/R_ANCHOR 2-way만 존재. 2022→2023, 2023→2024 두 폴드 모두 실행. `모델링/v3_domain_experiments/run_v14_r_only.py`.

**결과 — 도메인별 BSS 직접 비교 (adapter 적용 후, 같은 행 기준이라 오차 없이 비교 가능)**:

| 폴드 | 도메인 | F 포함(E004) | F 제거(E006) | 차이 |
| --- | --- | --- | --- | --- |
| 2023→2024 | R_CORE | 721.88 | 656.12 | **-65.76** |
| 2023→2024 | R_ANCHOR | 883.51 | 866.00 | **-17.51** |
| 2022→2023 | R_CORE | 805.50 | 706.66 | **-98.84** |
| 2022→2023 | R_ANCHOR | 772.95 | 553.69 | **-219.26** |

**상태**: 기각. 두 폴드, 두 도메인 전부 일관되게 F 제거가 더 나쁨(-17.5 ~ -219.3). E005와 동일한 방향 — 실제 평가가 R만 본다는 사실과 별개로, 같은 선수(pitcher_id/batter_id)가 R/F 양쪽에 등판하는 경우가 있어서 F에서의 결과가 그 선수 실력 추정치에 추가 정보로 작용하는 것으로 해석. **V14는 원안대로 F를 포함한 3-domain 구조를 유지.**

---

## E007 (2026-08-12) — V14 하이퍼파라미터 튜닝 (v1.1)

**배경**: E004는 참고 코드를 그대로 복붙한 것이라 우리 데이터에 맞춘 튜닝이 전혀 없었음. CatBoost baseline 때 했던 l2_leaf_reg 스윕처럼, V14의 조절 가능한 값들을 스윕.

**방법**: 비싼 fit(LightGBM×2, Ridge, ExtraTrees)은 한 번만 하고 그 예측값을 재사용해서 값싼 조합(블렌드 가중치, ADAPTER_SHRINK)부터 스윕, 그다음 재학습이 필요한 것(ridge_alpha, ExtraTrees 구조)을 순서대로. `모델링/submit_v14_domain_residual/tune_v14.py`, `tune_v14_combined.py`.

**결과**:

| 대상 | 원래(E004) | 튜닝값 | 개별 효과(2023→2024) |
| --- | --- | --- | --- |
| ridge_alpha | 10000 | 10000 (이미 최적, 100~300000 스윕) | 없음 |
| base 블렌드(lgb/ridge/trend) | 균등(1/3,1/3,1/3) | (0.3, 0.4, 0.3) | +2.13 (거의 노이즈) |
| ExtraTrees max_depth | 10 | 14 | +5.22 (1-seed 스크린) |
| **ADAPTER_SHRINK** | 0.5 | **0.8** | **+13~19** (가장 큰 레버) |

세 개(블렌드+depth14+shrink0.8)를 합쳐서 3-seed로 재검증:

| 폴드 | E004(원본) | E007(튜닝) |
| --- | --- | --- |
| 2023→2024 (primary) | 917.91 | **936.93** (+19.02) |
| 2022→2023 (stress) | 697.73 | **1009.72** (shrink=0.8 기준, +311.99) |

primary 폴드는 shrink=0.75~0.85 구간이 전부 936점대로 평탄(knife-edge 아님), stress 폴드도 같은 조합으로 크게 개선 — 한쪽 폴드에만 맞춰서 다른 쪽이 깨지는 과거 실패 패턴(recent_k_pitch_rate, V25.5)과 다름. (stress 폴드는 shrink=0.9까지도 계속 상승 중이라 이 폴드만의 진짜 최적점은 더 높을 수 있으나, primary 폴드 기준으로 0.8을 채택.)

**버그 발견/수정**: `build_artifacts.py`에서 블렌드 가중치를 바꿨는데 `script.py`는 예전 균등 블렌드가 하드코딩돼 있어서 결과가 어긋났음(둘 다 `model/v14_artifacts.joblib`을 쓰지만 blend 가중치 자체가 아티팩트에 저장 안 돼 있었음). `base_blend_weights`를 아티팩트에 추가 저장하고 `script.py`가 거기서 읽도록 수정, 재검증(로컬 test.csv 5행 재현값 소수점까지 일치)으로 확인.

**상태**: 채택, 실제 제출 완료. `submit_v14_domain_residual/submit.zip`(v1.1) 실제 LB **1032.0064496443** — 기존 champion(789.23) 대비 +242.77, 로컬(936.93)보다 실제가 더 높게 나옴(local/actual 정합 사례). 새 champion으로 확정(`HANDOFF.md` 참고). 참고팀 자체 보고값(1044.26, 자기 데이터/검증 기준이라 직접 비교 불가)과는 -12.26 차이인데, 튜닝 전(E004) 버전을 실제 제출해본 적이 없어서 이번 튜닝이 실제 LB에 도움이 됐는지는 아직 분리 확인 안 됨.
