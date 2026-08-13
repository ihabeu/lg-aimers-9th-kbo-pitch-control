# 실험 로그 (E-번호 순차 기록)

**시작일: 2026-08-12** (각 실험 항목에도 날짜 표기)

형식: 각 실험은 변경점, rolling OOT(2022/2023/2024, 가중치 0.2/0.3/0.5) 또는 single-split(2019-23→24), dual-fold(primary 2023→2024/stress 2022→2023) 결과, 채택/기각 여부를 남긴다.

기준선: CatBoost baseline(raw 44피처, l2_leaf_reg=15) — 2019-23→24 single-split 734.49, rolling weighted 818.65. 실제 LB 789.23.

---

## E001 (2026-08-12) — game_type × post_regime 교호작용

**출처**: 다른 참가자가 공개한 실험 로그에서 큰 단일 개선으로 보고된 피처(로지스틱+HistGradientBoosting 조합 기준). 명시적 교호작용이 그쪽 모델엔 크게 도움이 됨.

**변경점**: `game_type_regime = game_type + "_" + (season>=2023)` 를 CatBoost native categorical로 baseline 44개에 추가.

**결과 (single-split 2019-23→24)**: score=696.52, brier=0.248067 (baseline 734.49 대비 -37.97)

**상태**: 기각. CatBoost는 이미 `type="Interaction"` 분석에서 `season×asof_*`를 최상위로 자체 발견하고 있어서(이 세션 초반 확인), 명시적 교호작용이 CatBoost에는 중복 정보 + 노이즈로 작용. 선형/얕은 트리 모델에서 통했던 개선이 CatBoost엔 이미 내장돼 있었던 것으로 해석.

---

## E002 (2026-08-12) — R/M/O(reverse/middle/outside) hazard 분해, standalone

**출처**: 다른 참가자의 공개 실험 로그. `control_success`를 단일 이진분류로 안 풀고, 원래 target 정의의 3가지 실패유형(reverse/middle/outside)을 hazard(순차조건부) 구조로 분해:
`P(success) = (1-qR)(1-qM)(1-qO)`, qR/qM/qO는 각각 별도 CatBoost 분류기.

**라벨 복원**: 원본 데이터엔 투구별 R/M/O 라벨이 없어서, `asof_pitcher_reverse_rate`/`middle_rate`(이미 baseline 44개 피처에 있음)의 행간 변화량을 역산해서 복원(`rmo_labels.py`). row_id가 시간순이고 `asof_pitcher_n`이 행마다 정확히 +1씩 증가한다는(이 세션에서 확인) 전제를 이용. 복원율 99.89%, success 행에서 R/M 전부 0인 비율 100%로 정합성 확인.

**결과 (single-split 2019-23→24, standalone)**: score=575.52, brier=0.248369 (baseline 734.49 대비 -158.97). 동일 subset으로 학습한 단일 CatBoost(710.83)보다도 나쁨.

**상태**: standalone은 기각. 참고 자료도 standalone은 기각하고 "innovation"(베이스 모델 예측에 대한 보정치)으로만 채택했음 — 같은 방식으로 E003에서 재시도.

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

**상태**: 소폭 개선(+3.14, beta=0.1)이지만 hand_matchup(+18.40)보다도 작은 폭이라 노이즈 가능성 있음. rolling OOT 재확인 및 제출은 보류.

---

## E004 (2026-08-12) — game_type='F'(2군) 완전 제거 학습

**배경**: test.csv 5행 전부 `game_type='R'`(1군)로 확인됨. "학습에서 F를 빼면 R 예측이 더 좋아지지 않을까" 가설 검증.

**방법**: baseline CatBoost(44피처, depth=6, l2_leaf_reg=15) 그대로, rolling OOT(2022/23/24, 0.2/0.3/0.5) 각 폴드에서 (A) R+F 같이 학습 후 R행만 평가 vs (B) F를 학습에서 제거하고 R행만 평가. 조기종료도 R행 기준으로 함(평가 대상과 동일 분포).

**결과**:

| | 2022 | 2023 | 2024 | weighted |
| --- | --- | --- | --- | --- |
| A) R+F 학습 | 533.24 | 484.70 | 742.14 | **623.13** |
| B) R only 학습 | 506.56 | 561.77 | 625.68 | **582.68** |

**상태**: 기각. F를 빼면 오히려 -40.45로 더 나쁨 — 직관과 반대로 F 데이터가 R 예측에도(공유 트리 구조/정규화 효과로 추정) 도움이 됨. `모델링/v3_domain_experiments/exclude_f_domain.py`.

---

## E005 (2026-08-13) — champion + segment(core/hybrid/dev) residual corrector

base 모델(CatBoost champion, 789.23)은 그대로 두고, 그 오차만 segment별로 별도 학습해서 보정하는 2-stage 구조를 독립적으로 설계·구현. segment 기준(team 13이 F 참여율 이상치)은 이 세션 초반 우리 자체 EDA에서 발견.

로컬 primary 기준: base 734.49 → 3-way segment + ExtraTrees corrector **801.93**. corrector 모델 종류(ExtraTrees/RandomForest/XGBoost/LightGBM), segment 세분화(2/3/4-way), capacity, shrink, corrector 입력 피처(base_pred)까지 전부 로컬 dual-fold(2023→2024 primary, 2022→2023 stress)로 검증 — 상세 실험 로그는 `HANDOFF.md`의 "E019~E020" 절.

제출 패키지: `submit_segment_residual_corrector/submit.zip`. **상태**: 실제 LB **879.7995048079** — 새 유효 champion(789.23 대비 +90.57). 로컬(801.93)보다 실제가 더 높게 나옴.

---

## E006 (2026-08-13) — 멀티모델(CatBoost+LightGBM+XGBoost) 가중 블렌드 base + 기존 corrector — 실LB에서 기각

E005 champion(879.80) 이후, model_diversity 실험에서 확인한 CatBoost/LightGBM/XGBoost 간 residual 상관관계(0.83~0.96, 같은 계열 모델끼리의 0.998+보다 낮음)를 근거로 base 단계 자체를 3-model 블렌드로 바꾸는 실험. 균등(1/3) 가중치로 먼저 검증: primary -4.57 / stress +85.61 — stress 대폭 개선인데 primary만 손해. CatBoost 비중을 높인 7개 가중치 스윕한 결과 (0.8~0.4, .1~.3, .1~.3) 전 구간이 두 폴드 모두 기존 champion(801.93/755.63)을 이김.

채택: weight(cat=0.6, lgb=0.2, xgb=0.2) — 로컬 primary 815.15(+13.22), stress 833.05(+77.42). LightGBM/XGBoost 하이퍼파라미터·가중치 그리드 전부 자체 설정.

배포 패키지 `submit_multimodel_blend_corrector/submit.zip` 빌드, 로컬 sanity 확인 후 실제 제출.

**실제 제출 결과**: Public LB **869.7143690742** — champion(879.80) 미달, **-10.09**. 로컬은 두 폴드 다 이겼는데 실제는 오히려 낮음 — local/actual 재괴리 사례(6차 hand_matchup과 같은 패턴). **유효 champion은 계속 879.80(`submit_segment_residual_corrector/submit.zip`), 이 멀티모델 블렌드는 기각.**

---

## E007 (2026-08-13) — Diversity Lab: Ridge/Logistic/ElasticNet residual correlation 진단 — 기각

CatBoost/LightGBM/XGBoost 간 residual 상관관계(0.83~0.96, E006)보다 더 낮은 다양성을 기대하고, 완전히 다른 귀납적 편향(선형)인 Ridge(l2)/Logistic(규제없음)/ElasticNet을 `modeling/elastic_net.py`의 기존 전처리 파이프라인으로 테스트(`v3_domain_experiments/diversity_lab_linear.py`).

PRIMARY(2023→2024): BSS≈336, corr(vs CatBoost)=0.718 — 트리 모델끼리보다 확실히 낮은 상관관계.
STRESS(2022→2023): **BSS=0.00**, corr=0.104 — 상관관계는 더 낮지만 신호 자체가 없음(노이즈 수준).

원인: `elastic_net.py`에 이미 기록된 game_type×season_regime 반전 때문에, stress fold 학습 데이터(<2023)엔 "post2023" 카테고리가 아예 없어서 선형모델이 이 구간을 완전 미지의 상태로 예측 — 낮은 상관관계가 "다양성"이 아니라 "신호 부재"에서 나온 것. **기각.** 블렌드 후보가 되려면 낮은 상관관계뿐 아니라 두 폴드 모두 최소한의 실질 신호(BSS)가 있어야 한다는 기준을 stress에서 탈락.

---

## E008 (2026-08-13) — Diversity Lab: MLP — 참고용 보류

`v3_domain_experiments/diversity_lab_mlp.py`, 은닉층(64,32) MLP. PRIMARY: BSS=441.11, corr(vs CatBoost)=0.7509. STRESS: BSS=0.00, corr=0.2160.

재해석: CatBoost 단독도 STRESS에서 corrector 없이는 10.25로 사실상 붕괴 — stress 폴드는 corrector 없이는 모든 base 모델이 거의 무너지는 구간이라, MLP만의 결함이 아님. 다만 표준 정확도(441)가 트리(734)보다 크게 낮아 블렌드 후보로는 매력 낮음. **보류, 우선순위 낮음.**

---

## E009 (2026-08-13) — Diversity Lab: LSTM(투구 시퀀스, window=10) — 학습 미흡, 결론 보류

`v3_domain_experiments/diversity_lab_lstm.py`. row_id 기준 투수별 과거 10개 투구 시퀀스로 many-to-one 예측. PRIMARY/STRESS 둘 다 BSS=0.00, corr(vs CatBoost) -0.08/0.48. loss가 3 epoch 동안 ln(2)=0.693 근처에서 거의 안 움직여 사실상 학습이 안 됨.

**신호가 없다는 결론이 아니라 "이 정도 투자(3 epoch, hidden=64)로는 학습이 안 됐다"는 것** — 시퀀스 자체에 신호가 없는지, 더 학습시켜야 나오는지 미확정. 추가 투자 대비 기대값 낮아 우선순위 낮게 보류.

---

## E010 (2026-08-13) — calibration 진단: E006 실LB 실패는 calibration 문제가 아님

E006(멀티모델 블렌드, 로컬 두 폴드 다 이겼는데 실제 LB -10.09)이 과신/과소신(calibration) 문제였는지 진단(`calibration_diagnostic.py`). champion(CatBoost단독)과 기각된 블렌드(0.6/0.2/0.2)의 corrector 적용 후 최종 예측 bias/slope 비교:

| | PRIMARY bias | PRIMARY slope | STRESS bias | STRESS slope |
|---|---:|---:|---:|---:|
| champion | -0.00069 | 1.0111 | -0.00132 | 1.2393 |
| 기각된 블렌드 | -0.00058 | 1.0060 | -0.00044 | 1.1592 |

두 폴드 모두 블렌드가 champion과 비슷하거나 오히려 더 잘 보정됨(slope가 1에 더 가까움, bias 더 작음). **calibration은 실패 원인이 아니다** — post-hoc compression 보정으로 되살릴 수 있는 문제가 아니고, LightGBM/XGBoost 자체의 2025 일반화가 약하다는 구조적 가설에 더 무게가 실림.
