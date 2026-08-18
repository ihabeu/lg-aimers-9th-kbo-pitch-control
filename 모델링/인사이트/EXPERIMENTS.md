# 실험 로그 (E-번호 순차 기록)

**시작일: 2026-08-12** (각 실험 항목에도 날짜 표기)

데이터 배경지식은 [`도메인.md`](도메인.md), [`전처리 및 인사이트.md`](전처리%20및%20인사이트.md) 참고. 진행 상태 요약은 [`HANDOFF.md`](HANDOFF.md).

형식: 각 실험은 변경점, rolling OOT(2022/2023/2024, 가중치 0.2/0.3/0.5) 또는 single-split(2019-23→24), dual-fold(primary 2023→2024/stress 2022→2023) 결과, 채택/기각 여부를 남긴다.

기준선: CatBoost baseline(raw 44피처, l2_leaf_reg=15) — 2019-23→24 single-split 734.49, rolling weighted 818.65. 실제 LB 789.23.

**검증 방법론 노트**: rolling 3폴드(2022/2023/2024)를 가중평균으로만 보면 판단을 그르칠 수 있다 — 2022→2023은 regime 전환 자체를 맞혀야 하는 별개의 어려운 문제라 secondary/참고용으로 두고, **2019-23→24(primary)를 기준으로 채택/기각을 판정**한다. 아래 실험들은 전부 이 기준으로 재판정돼 있다.

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

**라벨 복원**: 원본 데이터엔 투구별 R/M/O 라벨이 없어서, `asof_pitcher_reverse_rate`/`middle_rate`(이미 baseline 44개 피처에 있음)의 행간 변화량을 역산해서 복원(`modeling/rmo_labels.py`). row_id가 시간순이고 `asof_pitcher_n`이 행마다 정확히 +1씩 증가한다는(이 세션에서 확인) 전제를 이용. 복원율 99.89%, success 행에서 R/M 전부 0인 비율 100%로 정합성 확인.

**결과 (single-split 2019-23→24, standalone)**: score=575.52, brier=0.248369 (baseline 734.49 대비 -158.97). 동일 subset으로 학습한 단일 CatBoost(710.83)보다도 나쁨.

**상태**: standalone은 기각. 참고 자료도 standalone은 기각하고 "innovation"(베이스 모델 예측에 대한 보정치)으로만 채택했음 — 같은 방식으로 E003에서 재시도.

---

## E003 (2026-08-12) — R/M/O hazard를 innovation(보정치)으로

**변경점**: `p_final = p_catboost + beta * (p_rmo_hazard - p_catboost)`, beta 스윕(`modeling/rmo_innovation.py`).

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

## E004 (2026-08-12) — R-only 학습 (`modeling/r_only_training.py`)

2025 test가 전부 1군(R)이라는 사실에 근거해 F를 학습에서 완전히 빼는 게 나은지 재확인(평가는 항상 R만).

| | 2022 | 2023 | **2024(primary)** | 가중평균 |
|---|---:|---:|---:|---:|
| R+F 학습(baseline) | 500.11 | 470.74 | **742.14** | 612.31 |
| R만 학습 | 505.13 | 628.86 | **640.22** | 609.79 |

**기각.** primary에서 -101.92. 기존 "R/F 완전분리(623.31)"는 별도 모델 구조였는데, 이번엔 모델 구조는 그대로 두고 학습 행만 바꿔도 같은 결론 — CatBoost가 `game_type`을 피처로 이미 갖고 있어서 F 행을 손으로 빼는 게 표본만 줄이는 손해로 작용함(데이터 줄이기 계열 실패 사례 추가).

---

## E005 (2026-08-12) — F 레짐 필터, 단일 모델 (`modeling/f_regime_filtered_training.py`)

R/F를 별도 모델로 안 쪼개고, 학습 행에서 F만 2023년 이후(post-break)로 제한(모델 구조는 baseline과 동일).

| | 2022 | 2023 | **2024(primary)** | 가중평균 |
|---|---:|---:|---:|---:|
| baseline | 500.11 | 470.74 | **742.14** | 612.31 |
| R전체+F 2023~ | 504.84 | 627.44 | **698.58** | 638.49 |

**기각.** primary -43.56. R-only보다는 덜 심하게 지지만(F를 다 뺀 게 아니라 절반만 뺀 거라) 같은 패턴. F의 2023 regime break는 우리가 손댈 문제가 아니라 CatBoost가 `game_type`+`season` 조합으로 이미 흡수하고 있다는 결론 — **F 관련 조정 시도는 여기서 완전 종료.**

---

## E006 (2026-08-12) — 계층형 EB 피처 (`modeling/hierarchical_eb_features.py`)

다른 참가자 공개 노트북의 계층 구조(global→team→pitcher→pitcher×hand/count/pressure→pitcher×batter, 시즌 경계 분해로 "이번 시즌 상태"를 스냅샷-안전하게 복원)를 간소화 이식해 baseline 44피처 위에 추가.

| | 2022 | 2023 | **2024(primary)** | 가중평균 |
|---|---:|---:|---:|---:|
| baseline | 500.11 | 470.74 | **742.14** | 612.31 |
| +계층형 EB | 566.87 | 473.72 | **728.29** | 619.64 |

**기각.** 가중평균은 +7.33로 올랐지만 2022 폴드 하나가 다 끌고 간 것이고 primary는 -13.85. hand_matchup 때와 같은 "로컬 폴드 부호가 갈리면 못 믿는다" 원칙 재확인.

---

## E007 (2026-08-12) — 레벨시프트 calibration (`modeling/level_shift_calibration.py`)

기존 calibration(직전 시즌 OOF에 Platt/Isotonic fit)은 "인접 시즌에 fit한 보정기는 방향이 해마다 뒤집혀서 전이 안 된다"는 걸 다른 참가자 공개 EDA가 이미 보여준 바로 그 방식이었다. 대신 **여러 시즌에 걸친 선형 추세로 다음 시즌 수준을 외삽**하고, 예측 확률 모양은 안 건드리고 평균만 상수 하나로 이동시킨다: `shift = extrapolated_rate(target_year) - actual_rate(target_year-1)` — target_year 이전 라벨만 쓰므로 실제 제출에서도 test.csv를 전혀 안 보고 학습 데이터만으로 미리 계산되는 상수다.

| | 2022 | 2023 | **2024(primary)** | 가중평균 |
|---|---:|---:|---:|---:|
| baseline(보정 없음) | 500.11 | 470.74 | **742.14** | 612.31 |
| +레벨시프트 | 459.51 | 341.06 | **707.85** | 548.14 |

**기각. 이번엔 모든 폴드에서 진다(애매함 없음).** 원인: 선형추세가 실제 필요한 보정량보다 과하게 밀었다 — 2024는 보정 전 pred_mean 0.49677 vs 실제 0.48971로 차이가 겨우 -0.007인데, 추세선이 뽑아낸 shift는 -0.019로 거의 3배다. 2019→2020 급락기 기울기를 2022~2024의 완만해진 구간까지 그대로 끌고 가서 생긴 과대보정. CatBoost가 `season`을 피처로 이미 갖고 있어서 드리프트를 어느 정도 스스로 따라가고 있었는데 거기에 수동으로 더 얹으니 오히려 어긋났다 — F 케이스와 같은 이야기(모델이 이미 하는 걸 사람이 손대면 손해).

---

## E008 (2026-08-12) — baseline × 계층형EB residual 상관 (`modeling/residual_correlation_eb.py`)

"base 블렌드 + residual adapter" 2층 구조를 시도하기 전에, 애초에 피처 표현이 다른 두 모델(raw 44피처 CatBoost vs +계층형 EB CatBoost)의 오차가 실제로 덜 겹치는지부터 확인. 기존 `model_diversity.py`는 "같은 44피처, 다른 알고리즘"이었어서 이번엔 "다른 피처 표현, 같은 알고리즘(CatBoost)"으로 재검증.

2024 단일 폴드: pred correlation 0.95428, **residual correlation 0.99969** — `model_diversity.py`의 0.998보다도 높다. 피처 표현을 바꿔도 CatBoost가 찾는 오차 패턴은 사실상 동일. 다만 0.3/0.5/0.7 가중치로 단순 블렌드해보니 baseline(742.14)보다 소폭 높은 751점대가 나와서(2024만 보고 고른 값이라 아직 못 믿음) E009로 이어짐.

---

## E009 (2026-08-12) — baseline+EB 블렌드, discovery(2022+2023) 잠금 → confirmation(2024) 검증 (`modeling/eb_blend_discovery_confirm.py`)

E008의 블렌드 가중치를 confirmation 연도(2024)를 안 보고 discovery(2022+2023)의 worst-case 기준으로만 골라서 재검증 — "discovery에서 잠그고 confirmation은 순수 확인만" 원칙 적용.

```
discovery: w_eb를 0→1로 올릴수록 2022·2023 둘 다 단조 개선, worst-case 최댓값 = w=0.5
confirmation(2024, 가중치 선택에 전혀 안 쓴 값): baseline 742.14 → blend(w=0.5) 751.07  (+8.93)
```

**이 방향에서 유일하게 방법론을 통과한 양의 결과.** 다만 +8.93을 Brier 차이로 환산하면 약 0.0000223로, 기준으로 쓰는 노이즈 바닥선(경기 클러스터 부트스트랩 sd ≈0.000125)보다 작다 — 방향은 discovery 두 폴드와 confirmation 셋 다 일관되게 맞았지만, 절대 크기가 작아 "확실히 노이즈 이상"이라고 못 박기는 어렵다. residual 상관이 0.9997이었던 것과 일관된 결과(거의 안 겹치지만 완전히 겹치지도 않아서 아주 작은 분산 감소만 얻음).

**판정**: 채택 여부 보류. hand_matchup도 로컬(rolling OOT 2개 폴드)은 통과했다가 실제 LB에서 뒤집힌 전례가 있어서, 이번 결과(그것보다 더 엄격한 discovery/confirmation 분리까지 통과했지만 개선폭은 훨씬 작음)도 곧바로 제출 후보로 올리지 않고 기록만 해둔다.

---

## E010 (2026-08-12) — F 시즌 내 온도차(temporal decay) 피처 (`개발/v3_domain_experiments/f_temporal_decay.py`)

"F 성공률이 시즌 내에서도 계속 하락한다"는 관찰을 우리 자체 EDA로 재확인한 뒤 baseline 위에 독립적으로 구현.

**자체 재확인**: F 성공률이 2022년(4월 0.751→9월 0.691)과 2023년(4월 0.494→9월 0.462)엔 대체로 하락 추세지만, 2024년은 7월(0.429)까지 하락하다 8~10월(0.457→0.482→0.495)에 다시 반등 — 패턴이 매년 깨끗하게 반복되진 않음.

피처: `f_season_progress` = (game_month - 그 시즌 첫 달) × (game_type=='F'), R행은 0.

| | 2022 | 2023 | **2024(primary)** | 가중평균 |
|---|---:|---:|---:|---:|
| baseline | 2280.22 | 10.25 | **734.49** | 826.37 |
| +f_season_progress | 2262.43 | 2.52 | **721.69** | 814.09 |

**기각.** 세 폴드 전부 하락(primary -12.80). E001(`game_type×season` 명시적 교호작용, 기각)과 같은 패턴 — 패턴 자체는 실재해도 CatBoost가 `game_month`+`game_type`으로 이미 스스로 찾고 있어서 명시적 피처는 중복 정보+노이즈로만 작용.

---

## E011 (2026-08-12) — monotone_constraints (`개발/v3_domain_experiments/monotone_constraints.py`)

방향이 명확한 피처(성공률 계열 +, ball_rate -)에 CatBoost monotone_constraints 적용.

| | 2022 | 2023 | **2024(primary)** | 가중평균 |
|---|---:|---:|---:|---:|
| baseline | 2280.22 | 10.25 | **734.49** | 826.37 |
| +monotone | 2232.33 | 10.32 | **679.91** | 789.52 |

**기각.** primary -54.58.

---

## E012 (2026-08-13) — game_type='F'(2군) 완전 제거 학습 (재검증)

E004(R-only, single-split)와 별개로, rolling OOT(가중평균) 방법론으로 같은 가설을 다시 검증.

**배경**: test.csv 5행 전부 `game_type='R'`(1군)로 확인됨. "학습에서 F를 빼면 R 예측이 더 좋아지지 않을까" 가설 재검증.

**방법**: baseline CatBoost(44피처, depth=6, l2_leaf_reg=15) 그대로, rolling OOT(2022/23/24, 0.2/0.3/0.5) 각 폴드에서 (A) R+F 같이 학습 후 R행만 평가 vs (B) F를 학습에서 제거하고 R행만 평가. 조기종료도 R행 기준으로 함(평가 대상과 동일 분포).

**결과**:

| | 2022 | 2023 | 2024 | weighted |
| --- | --- | --- | --- | --- |
| A) R+F 학습 | 533.24 | 484.70 | 742.14 | **623.13** |
| B) R only 학습 | 506.56 | 561.77 | 625.68 | **582.68** |

**상태**: 기각. F를 빼면 오히려 -40.45로 더 나쁨 — 직관과 반대로 F 데이터가 R 예측에도(공유 트리 구조/정규화 효과로 추정) 도움이 됨. E004와 검증 방법론(single-split vs rolling OOT)이 다른데도 같은 결론에 도달 — 결론의 신뢰도를 높이는 교차확인. `모델링/v3_domain_experiments/exclude_f_domain.py`.

---

## E013 (2026-08-13) — champion + segment(core/hybrid/dev) residual corrector

base 모델(CatBoost champion, 789.23)은 그대로 두고, 그 오차만 segment별로 별도 학습해서 보정하는 2-stage 구조를 독립적으로 설계·구현. segment 기준(team 13이 F 참여율 이상치)은 이 세션 초반 우리 자체 EDA에서 발견.

로컬 primary 기준: base 734.49 → 3-way segment + ExtraTrees corrector **801.93**. corrector 모델 종류(ExtraTrees/RandomForest/XGBoost/LightGBM), segment 세분화(2/3/4-way), capacity, shrink, corrector 입력 피처(base_pred)까지 전부 로컬 dual-fold(2023→2024 primary, 2022→2023 stress)로 검증 — 상세 실험 로그는 `HANDOFF.md`의 세부 절 참고.

제출 패키지: `submit/v9_segment_corrector/submit.zip`. **상태**: 실제 LB **879.7995048079** — 새 유효 champion(789.23 대비 +90.57). 로컬(801.93)보다 실제가 더 높게 나옴.

---

## E014 (2026-08-13) — 멀티모델(CatBoost+LightGBM+XGBoost) 가중 블렌드 base + 기존 corrector — 실LB에서 기각

E013 champion(879.80) 이후, base 모델간 residual 상관관계(0.83~0.96, 같은 계열 모델끼리의 0.998+보다 낮음)를 근거로 base 단계 자체를 3-model 블렌드로 바꾸는 실험. 균등(1/3) 가중치로 먼저 검증: primary -4.57 / stress +85.61 — stress 대폭 개선인데 primary만 손해. CatBoost 비중을 높인 7개 가중치 스윕한 결과 (0.8~0.4, .1~.3, .1~.3) 전 구간이 두 폴드 모두 기존 champion(801.93/755.63)을 이김.

채택: weight(cat=0.6, lgb=0.2, xgb=0.2) — 로컬 primary 815.15(+13.22), stress 833.05(+77.42). LightGBM/XGBoost 하이퍼파라미터·가중치 그리드 전부 자체 설정.

배포 패키지 `submit/v10_multimodel_blend/submit.zip` 빌드, 로컬 sanity 확인 후 실제 제출.

**실제 제출 결과**: Public LB **869.7143690742** — champion(879.80) 미달, **-10.09**. 로컬은 두 폴드 다 이겼는데 실제는 오히려 낮음 — local/actual 재괴리 사례. **유효 champion은 계속 879.80(`submit/v9_segment_corrector/submit.zip`), 이 멀티모델 블렌드는 기각.**

---

## E015 (2026-08-13) — Diversity Lab: Ridge/Logistic/ElasticNet residual correlation 진단 — 기각

CatBoost/LightGBM/XGBoost 간 residual 상관관계(0.83~0.96, E014)보다 더 낮은 다양성을 기대하고, 완전히 다른 귀납적 편향(선형)인 Ridge(l2)/Logistic(규제없음)/ElasticNet을 `modeling/elastic_net.py`의 기존 전처리 파이프라인으로 테스트(`개발/v3_domain_experiments/diversity_lab_linear.py`).

PRIMARY(2023→2024): BSS≈336, corr(vs CatBoost)=0.718 — 트리 모델끼리보다 확실히 낮은 상관관계.
STRESS(2022→2023): **BSS=0.00**, corr=0.104 — 상관관계는 더 낮지만 신호 자체가 없음(노이즈 수준).

원인: `elastic_net.py`에 이미 기록된 game_type×season_regime 반전 때문에, stress fold 학습 데이터(<2023)엔 "post2023" 카테고리가 아예 없어서 선형모델이 이 구간을 완전 미지의 상태로 예측 — 낮은 상관관계가 "다양성"이 아니라 "신호 부재"에서 나온 것. **기각.** 블렌드 후보가 되려면 낮은 상관관계뿐 아니라 두 폴드 모두 최소한의 실질 신호(BSS)가 있어야 한다는 기준을 stress에서 탈락.

---

## E016 (2026-08-13) — Diversity Lab: MLP — 참고용 보류

`개발/v3_domain_experiments/diversity_lab_mlp.py`, 은닉층(64,32) MLP. PRIMARY: BSS=441.11, corr(vs CatBoost)=0.7509. STRESS: BSS=0.00, corr=0.2160.

재해석: CatBoost 단독도 STRESS에서 corrector 없이는 10.25로 사실상 붕괴 — stress 폴드는 corrector 없이는 모든 base 모델이 거의 무너지는 구간이라, MLP만의 결함이 아님. 다만 표준 정확도(441)가 트리(734)보다 크게 낮아 블렌드 후보로는 매력 낮음. **보류, 우선순위 낮음.**

---

## E017 (2026-08-13) — Diversity Lab: LSTM(투구 시퀀스, window=10) — 학습 미흡, 결론 보류

`개발/v3_domain_experiments/diversity_lab_lstm.py`. row_id 기준 투수별 과거 10개 투구 시퀀스로 many-to-one 예측. PRIMARY/STRESS 둘 다 BSS=0.00, corr(vs CatBoost) -0.08/0.48. loss가 3 epoch 동안 ln(2)=0.693 근처에서 거의 안 움직여 사실상 학습이 안 됨.

**신호가 없다는 결론이 아니라 "이 정도 투자(3 epoch, hidden=64)로는 학습이 안 됐다"는 것** — 시퀀스 자체에 신호가 없는지, 더 학습시켜야 나오는지 미확정. 추가 투자 대비 기대값 낮아 우선순위 낮게 보류.

**→ E020에서 원인(피처 정규화 누락) 발견 및 재시도, 최종 기각.**

---

## E018 (2026-08-13) — calibration 진단: E014 실LB 실패는 calibration 문제가 아님

E014(멀티모델 블렌드, 로컬 두 폴드 다 이겼는데 실제 LB -10.09)이 과신/과소신(calibration) 문제였는지 진단(`calibration_diagnostic.py`). champion(CatBoost단독)과 기각된 블렌드(0.6/0.2/0.2)의 corrector 적용 후 최종 예측 bias/slope 비교:

| | PRIMARY bias | PRIMARY slope | STRESS bias | STRESS slope |
|---|---:|---:|---:|---:|
| champion | -0.00069 | 1.0111 | -0.00132 | 1.2393 |
| 기각된 블렌드 | -0.00058 | 1.0060 | -0.00044 | 1.1592 |

두 폴드 모두 블렌드가 champion과 비슷하거나 오히려 더 잘 보정됨(slope가 1에 더 가까움, bias 더 작음). **calibration은 실패 원인이 아니다** — post-hoc compression 보정으로 되살릴 수 있는 문제가 아니고, LightGBM/XGBoost 자체의 2025 일반화가 약하다는 구조적 가설에 더 무게가 실림.

---

## E019 (2026-08-14) — CatBoost 멀티시드 배깅(base 모델 평균) — 기각

E014(멀티모델 블렌드)가 실LB에서 실패한 이유로 "다른 모델 패밀리 자체의 일반화가 약함"을 지목했었다. 이 가설이 맞다면, 모델 패밀리는 그대로 두고 같은 CatBoost를 시드만 바꿔 평균(bagging)하는 건 리스크가 낮을 것으로 예상하고 검증(`개발/v3_domain_experiments/catboost_seed_bagging.py`).

| | primary base→+corrector | stress base→+corrector |
|---|---|---|
| 단일시드(기존 champion, seed=42) | 734.49→**801.93** | 10.25→**755.63** |
| 3시드 평균(42,2026,314) | 719.79→786.08 | 10.31→752.05 |
| 5시드 평균(42,2026,314,7,123) | 721.91→790.73 | 5.03→757.38 |

**기각.** 두 시드 평균 조합 다 primary에서 champion보다 나쁘다(-11~-16). 원인: 개별 시드별 base BSS를 보면 seed=42(734.49)가 5개 중 압도적으로 가장 좋고 나머지(689~716)는 전부 눈에 띄게 낮다 — seed=42가 유독 좋은 시드라서, 다른 시드와 평균 내면 오히려 끌어내려진다. 이건 이 세션 이전에 이미 한 번 발견된 것과 정확히 같은 패턴이다(HANDOFF.md 초기 실험 로그: "5시드 앙상블 723.57, seed=42가 5개 중 최고") — 이번엔 dual-fold(primary/stress) 방법론으로 독립적으로 재확인됐다. **시드 배깅 방향은 이걸로 완전히 종료.**

---

## E020 (2026-08-14) — LSTM 재시도(정규화 수정 + 15 epoch) — 이번엔 제대로 학습됐지만 여전히 기각

E017의 loss 정체가 사실은 학습 실패였다는 걸 확인 — 입력 피처 스케일이 전혀 안 맞았음(0~100대의 `home_win_expectancy`, 0~1대의 rate, 수백 단위 라벨인코딩 정수가 그대로 섞여 있었음). train 통계로만 fit한 채널별 표준화(leak-safe)를 추가하고 epoch을 3→15로 늘려 재시도(`diversity_lab_lstm.py` v2).

| | PRIMARY | STRESS |
|---|---|---|
| loss 추이 | 0.6845→**0.6770**(15 epoch, 꾸준히 감소) | 0.6822→**0.6742** |
| BSS | 0.00 | 0.00 |
| corr(vs CatBoost) | -0.08 → **0.49** | 0.48 → **0.65** |

**이번엔 확실히 학습이 됐다** — loss가 꾸준히 감소했고(정체 아님), CatBoost와의 residual 상관관계도 거의 무의미한 수준에서 0.49~0.65로 뚜렷하게 올라갔다(모델이 진짜 신호를 찾아 CatBoost와 비슷한 방향으로 수렴하고 있다는 뜻). **그런데도 BSS는 여전히 0.00** — cross-entropy(loss)는 개선됐지만 제곱오차(Brier) 기준으로는 상수 예측(baseline) 수준을 못 넘었다. 즉 신호를 찾긴 했지만 실용적으로 쓸 만한 수준까지는 못 미쳤다.

**해석**: 6절(EDA.md 심층분석 1, ablation z-검정)에서 이미 `asof_*` 요약통계가 시간축 피처 다음으로 크게 기여한다는 게 확인됐다 — LSTM이 찾으려는 "원시 투구 순서의 순서-의존 패턴"이 `asof_*`가 이미 압축해서 제공하는 정보를 크게 못 넘어선다는 뜻으로 해석할 수 있다. 더 큰 capacity(hidden size, layer 수)나 더 많은 epoch으로 계속 투자하면 개선될 여지는 있지만, 트리 모델이 이 문제의 실질적 상한에 가깝다는 그동안의 반복된 결론(EXPERIMENTS.md 전체, 전처리 및 인사이트.md §11)과 일관되게, 투자 대비 기대값이 낮다고 판단해 **LSTM 트랙은 여기서 종료**한다.

---

## E021 (2026-08-14) — hand match를 corrector segment 라우팅 축으로 추가 — 로컬 개선폭은 있으나 통계적으로 유의하지 않음, 기각

오늘 밤 EDA 심층분석(eda/deep_dive.py)에서 투수x타자손 매치업 기준 분산 상한(939)이 현재 champion(879.80)보다 높고, 투수손x타자손 교호작용이 permutation 검정으로 강하게 유의(p<0.0001)하다는 게 확인됐다. hand_matchup을 "피처로 추가"하는 방식(6차 제출)은 이미 로컬 승·실LB 패로 기각된 전례가 있어서, 이번엔 다른 구조로 시도 — 피처가 아니라 이미 로컬/실제가 정합했던 성공 패턴(E013, segment residual corrector)과 같은 방식으로, hand match 여부를 corrector의 **라우팅 축**에 추가(core/hybrid/dev 각각을 same_hand/diff_hand로 나눠 6-way, `개발/v3_domain_experiments/segment_corrector_hand_routing.py`).

| | base | 3-way(기존 champion) | 6-way(+hand routing) | 차이 | pitcher-bootstrap z |
|---|---:|---:|---:|---:|---:|
| PRIMARY | 734.49 | 801.93 | 821.64 | +19.71 | **1.51** |
| STRESS | 10.25 | 755.63 | 781.96 | +26.34 | **1.13** |

세그먼트 표본 크기 확인(가장 작은 것도 11,447~12,496행으로 4-way F팀분리 실패 사례보다 훨씬 큼): `core_diff`~93k, `core_same`~85k, `hybrid_diff`~24k, `hybrid_same`~20k, `dev_same`~14-17k, `dev_diff`~11-12k.

**1차 판단(점수만)**: 로컬 두 폴드 다 개선, 원칙상 채택 기준 통과.

**2차 검증(유의성)**: 3-way 대비 6-way의 우위를 투수 단위 bootstrap(500회, EDA ablation z-검정과 같은 방법론)으로 재검정한 결과 **z=1.51(primary)/1.13(stress) — 둘 다 관례적 유의 기준(z≈1.96)에 못 미친다.** 점수 차이(+19.71/+26.34) 자체는 작지 않지만, 투수 단위로 리샘플링해보면 이 정도 개선폭이 우연히 나올 가능성이 무시할 수준이 아니라는 뜻이다.

**추가 경고**: 이 신호는 애초에 E023(residual bias 스캔)이 "hand_matchup과 동일 신호, 로컬은 좋았는데 실제 LB에서 하락한 전례가 있다"고 명시적으로 경고했던 바로 그 신호원이다. 인코딩 방식(피처 → segment 라우팅)만 바꿨을 뿐 근본 정보는 동일해서, 통계적으로 유의했더라도 신뢰도를 다른 신규 발견과 동일하게 볼 수는 없었을 것이다.

**결론**: 점수 개선은 있지만 통계적으로 뒷받침되지 않고, 같은 정보원의 실LB 실패 전례까지 있어 **기각**. `submit/v9_segment_corrector/submit.zip`(3-way, 879.80)을 그대로 유지한다. 코드는 재현 가능하게 남겨둔다.

---

## E022 (2026-08-14) — asof rate 상대값 3종(diff/mean/product) (`modeling/relative_rate_features.py`) — 기각

이전 세션에 작성만 되고 실행 로그가 없던 스크립트를 실제로 돌려서 확인. `pitcher_rate_diff = asof_pitcher_success_rate - asof_batter_success_rate`, `pitcher_rate_mean`, `pitcher_rate_product` 3개를 baseline 44피처에 추가.

| | baseline | +rate_diff/mean/product | 차이 |
|---|---:|---:|---:|
| single-split(2019-23→24) | 734.49 | 711.93 | **-22.56** |

**기각.** CatBoost는 트리라 pitcher/batter rate 두 컬럼의 차/평균/곱을 이미 스스로 분기로 찾아낼 수 있어서, 명시적으로 만들어 준 파생 컬럼은 다중공선성만 늘리고 정보는 안 늘린다 — E001(교호작용 피처), 6차 제출(hand_matchup 피처), monotone_constraints(E011)에서 반복 확인된 것과 같은 패턴.

## E023 (2026-08-14) — 이력 rate 고도화: EB 스무딩/불확실성/최근 드리프트 (`modeling/uncertainty_features.py`) — 기각

역시 작성만 되고 미실행 상태였던 스크립트. `pitcher_smoothed_rate/batter_smoothed_rate`(EB 스무딩, prior_strength=20), `pitcher_uncertainty/batter_uncertainty`(이항표준오차), `pitcher_recent_drift`(최근3경기-시즌평균)를 각각/전체 추가.

| | baseline | +pitcher uncertainty | +batter uncertainty | +recent drift | +all |
|---|---:|---:|---:|---:|---:|
| single-split(2019-23→24) | 734.49 | 698.66 | 702.52 | 715.71 | 714.92 |

**기각.** 넷 다 baseline 미달, 그중 pitcher uncertainty가 가장 크게 나쁨(-35.83). asof_pitcher_n/asof_batter_n(표본수)이 이미 44피처에 있어서 CatBoost가 "이 rate를 얼마나 믿을지"를 스스로 표본수와 함께 학습할 수 있는데, 굳이 표준오차를 명시적으로 계산해서 얹어주는 게 오히려 방해가 되는 것으로 보인다.

## Trackman 물리 데이터 재확인 (2026-08-14, 신규 실험 아님)

팀 깃헙(iamdbstjd/LGAIMERS)의 E18-T0/T1을 참고하다가 "anonymized pitcher_id ↔ Trackman ID를 직접 교집합으로 조인하면 0"이라는 이전 메모를 보고 재시도하려 했으나, `data/derived_trackman_pitcher_mapping.csv`(332명, rel_gap 유사도 매핑)를 이용한 동일한 접근이 이미 `modeling/trackman_features.py`로 구현되고 E024(HANDOFF.md)에서 residual 상관관계 거의 0(-0.0038~+0.0027)으로 **완전 종료 처리된 상태**였음을 확인. 중복 구현 방지를 위해 새로 만든 스크립트는 삭제. Trackman 물리 데이터 경로는 계속 닫힌 상태 유지.

## E024 (2026-08-14) — within-game pitch count (경기 내 누적 투구수) (`modeling/within_game_state.py`) — 기각

작성만 되고 미실행이던 스크립트. row_id가 사실상 완벽한 시간순 인덱스라는 걸 이용해, 투수별로 정렬 후 inning이 감소하는 지점을 게임 경계로 잡아 `pitch_count_before`(이번 경기에서 현재 투구 이전까지 이 투수가 던진 수, 경기 경계마다 리셋)를 복원. 지금까지의 asof_* 피처는 전부 커리어/시즌/최근 N경기 단위였고, "이번 경기 안에서의 피로도"라는 축은 이번이 처음이라 baseline과 겹치는 정보가 아니다.

| | baseline | +pitch_count_before | 차이 |
|---|---:|---:|---:|
| single-split(2019-23→24) | 734.49 | 711.78 | **-22.71** |

**기각.** 다만 게임 경계 탐지 자체가 완벽하지 않음(경기당 투구수 분포 max=767구로 실제 KBO 선발 최대치를 훨씬 초과 — 불펜 등판 등에서 이닝-감소 휴리스틱이 게임 경계를 놓치는 경우가 섞여 있음을 시사). 신호가 없다기보다 피처 자체의 노이즈가 컸을 가능성이 있지만, 지금까지 단일 신규 피처를 baseline에 얹어서 개선된 사례가 한 번도 없었다는 이 세션의 반복 패턴(E001, E010, E011, E022, E023, hand_matchup 등)에 비춰볼 때 게임 경계 정확도를 더 다듬어도 결과가 뒤집힐 가능성은 낮다고 판단, 추가 투자 안 함.

## E025 (2026-08-15) — corrector segment 라우팅 축으로 "중요도 높은" 변수 3종 시도 (`개발/v3_domain_experiments/segment_corrector_importance_routing.py`) — 명확히 기각(유의하게 악화)

E021(hand_matchup 라우팅)이 유의하지 않았던 이유가 "pitcher_hand/batter_hand 자체가 CatBoost importance 최하위권(0.0021, 44개 중 40위권 밖)이라 원래 신호가 약했기 때문"이라는 가설을 세우고, 반대로 **importance가 실제로 높은 변수**(eda/eda_outputs/train_feature_importance.csv 기준: `asof_pitcher_success_rate` 1위 0.1247, `asof_batter_success_rate` 5위 0.0671, `li` 15위 0.0138)를 중앙값 기준 2분할해 기존 3-way(core/hybrid/dev) 위에 6-way 라우팅으로 추가. 판정도 E021과 동일하게 pitcher-cluster bootstrap z-검정.

| 축 | PRIMARY 3-way→6-way | z | STRESS 3-way→6-way | z |
|---|---:|---:|---:|---:|
| asof_pitcher_success_rate | 801.93→759.04 (-42.89) | **-3.91** | 755.63→691.95 (-63.68) | **-2.91** |
| asof_batter_success_rate | 801.93→782.63 (-19.30) | **-2.28** | 755.63→730.44 (-25.19) | **-2.36** |
| li | 801.93→776.48 (-25.45) | **-2.76** | 755.63→717.20 (-38.43) | **-3.20** |

**명확히 기각.** E021과 다르게 이번엔 "유의하지 않다"가 아니라 **통계적으로 유의하게 나빠졌다**(모든 축·양쪽 폴드에서 |z|>1.96, 방향도 전부 악화). 가설과 정반대 결과: importance가 높을수록 오히려 라우팅 축으로 쓰면 더 나쁘다. 원인으로 추정되는 것 — hand/game_type처럼 원래 이산적인 변수와 달리 이 셋은 **연속형 고정보 변수**라서, 중앙값으로 잘라 세그먼트를 나누면 (1) 정보 손실(연속값 → 이분값)이 크고 (2) 각 segment corrector의 학습 표본이 반토막나 노이즈가 커지는데, 이 정보는 corrector가 이미 44피처 그대로 입력받아 연속형으로 잘 쓰고 있었던 것(라우팅 없이도 corrector 학습 데이터 안에 포함됨)이라 나눠봐야 얻는 게 없고 잃는 것만 있다. **결론: "중요도가 높은 변수 = 좋은 라우팅 축"이 아니다. 라우팅 축은 game_type/hand처럼 이산적이고, corrector가 이미 쓰고 있는 정보와 별개의 구조적 분기일 때만 시도할 가치가 있다.**

## E026 (2026-08-15) — LightGBM 처음부터 새로 튜닝 (다른 모델 패밀리 탐색 트랙 1단계) (`개발/lightgbm_family_exploration/`) — base 단독 성능이 CatBoost에 구조적으로 못 미침, 기각

사용자 요청으로 "CatBoost+corrector 아키텍처 위의 자연스러운 레버는 소진됐다"는 판단 이후 다른 모델 패밀리를 처음부터 탐색. E014(멀티모델 블렌드)의 LightGBM은 파라미터를 대충 하나 골라 썼던 것과 달리, CatBoost의 l2_leaf_reg 스윕과 동일한 수준으로 제대로 튜닝했다.

**1) 하이퍼파라미터 스윕**(`lightgbm_baseline_sweep.py`, num_leaves×min_child_samples×reg_lambda 18조합, single-split 2019-23→24): 최고 config(num_leaves=15, min_child_samples=1000, reg_lambda=1.0)가 616.43. **CatBoost baseline(734.49) 대비 -118.06 — 이 세션에서 나온 어떤 격차보다도 크다.**

**2) 진단(계산 실수인지 확인)**:
- AUC는 거의 동일(LightGBM 0.5452 vs CatBoost 0.548) — 랭킹 능력 자체는 큰 차이 없음.
- 그런데 예측 평균이 0.4959로 실제 성공률(0.4861)보다 체계적으로 높음(bias +0.0098). Brier-score 자체를 eval metric으로 직접 최적화(logloss 프록시가 아니라)해도 bias 그대로(616 근방에서 안 움직임) — early stopping 지표 문제가 아니라 모델 자체의 특성.
- 사후 평균보정(bias만 제거): 654.53. **진단용(valid에 직접 fit, 실전에선 불가능한 leak 있는 상한선) Platt scaling: 657.04.** 즉 완벽하게 보정해도 CatBoost보다 77점 이상 낮다 — calibration 문제가 30% 정도만 설명하고, 나머지 70%는 진짜 정보 추출 격차.

**3) dual-fold 재확인**(`lightgbm_dual_fold_check.py`): PRIMARY LightGBM=616.43 vs CatBoost=734.49(-118.06, 재확인). STRESS는 두 모델 다 점수 하한 근처(LightGBM=0.00, CatBoost=10.25, 차이 -10.25)라 이 폴드에서는 판단력이 별로 없음 — 원래 STRESS 폴드는 corrector 없이는 두 모델 다 거의 무의미한 값이 나온다는 게 champion 개발 초반부터 알려진 특성.

**4) residual 상관관계**: 0.9996 (E014가 측정했던 cross-family 범위 0.83~0.96보다 훨씬 높고, 오히려 "같은 계열 모델끼리의 0.998+"에 가까움). **핵심 발견: 잘 튜닝할수록 LightGBM은 CatBoost와 다른 실수를 하는 게 아니라 같은 실수를 더 못 걸러낸 채로 한다** — 즉 튜닝을 더 투자해도 다양성(블렌드 가치)이 생기는 게 아니라 오히려 CatBoost에 수렴하기만 한다. E014에서 관찰된 낮은 상관(0.83~0.96)은 그 스크립트가 LightGBM을 제대로 안 튜닝해서 나온 "노이즈로 인한 가짜 다양성"이었을 가능성이 높다.

**결론**: LightGBM은 이 데이터셋에서 튜닝을 아무리 투자해도 CatBoost를 단독으로 이기지 못하고(-118, 보정해도 -77), 잘 튜닝할수록 오히려 CatBoost와 겹치는 정보만 뽑아내서 앙상블 다양성 가치도 없다. base 위에 segment corrector를 얹는 4단계(계획했던 다음 단계)는, 118점 격차를 corrector 하나(champion 기준 +67점 수준)로 못 뒤집을 게 명백하고 잔차 패턴도 거의 동일해 corrector가 걷어낼 "새로운" 정보가 없을 것으로 판단, 추가 실행 없이 여기서 트랙 종료. **다른 모델 패밀리 탐색은 "CatBoost가 이 특정 44피처·leak-safe 이력 데이터 구조에서 구조적 우위가 있다"는 결론으로 마무리.**

## E027 (2026-08-15) — 대회 규정 기반 "혹시 빠뜨린 공식 피처가 있나" 감사 — `asof_pitcher_pitchmix_n` 발견, 테스트 후 기각

사용자가 "대회 규정을 참고하면서 신규 피처를 잘 생각해보라"고 요청. `대회 목적 및 규칙.md`/`data/data_description.md`에 명시된 원본 컬럼 49개(48 입력 + `control_success`) 전체를 `FEATURES`(44개)와 직접 대조하는 감사를 수행 — pitcher_id/batter_id(의도적 제외, corrector 라우팅에만 씀)를 빼면 정확히 하나가 비어 있었다: 공식 `asof_*` 19개 중 `asof_pitcher_pitchmix_n`(투수의 구종 이력 표본 수)이 FEATURES에 한 번도 들어간 적이 없었다.

바로 검증: `asof_pitcher_n`과의 상관관계 **0.99999997** — 사실상 동일 컬럼(둘 다 "이 투구 직전까지 이 투수의 누적 투구 수"를 서로 다른 서브시스템에서 센 것으로 추정). 추가해서 테스트한 결과 696.14(baseline 734.49 대비 **-38.35**) — 이미 있는 정보와 거의 완전히 겹치는 컬럼을 추가하면 손해라는 이 세션의 반복 패턴과 일치.

**결론**: 규정과 원본 컬럼 목록을 다시 감사해도 빠뜨린 게 딱 하나뿐이었고 그마저 중복이었다는 것 자체가, 44피처 구성이 이미 원본 데이터를 남김없이 다 썼다는 걸 보여주는 근거다. 또한 이 감사 과정에서 대회 규정(§9, data_description.md 5절)을 다시 확인한 결과 — 이 competition 특유의 강한 leak-safety 제약(`test.csv` 내부 행을 이용한 어떤 집계/rolling/target-encoding도 금지)이 전형적인 캐글류 대회에서 흔한 피처엔지니어링 클래스 전체(테스트셋 통계 기반 인코딩)를 원천 차단하고 있어서, 남은 합법적 재료가 정확히 이 44개 + trackman_history.csv(이미 시도, E024에서 종료)뿐이라는 걸 재확인. **신규 피처 방향은 이걸로 완전히 소진.**

## E028 (2026-08-15) — 파생변수를 "추가"가 아니라 "원본 대체"로 넣으면 다른가 (`개발/v3_domain_experiments/feature_replace_not_add.py`) — 기각(추가보다도 더 나쁨)

사용자 질문: E022/E023 등 지금까지의 모든 파생변수 실험이 `BASE_FEATURES + NEW_FEATURES`(원본 44개 유지 + 파생변수 얹기)였는데, 원본을 지우고 파생변수로 **대체**했으면 결과가 달랐을까? E022/E023이 쓰던 함수를 그대로 재사용해서(코드 안 새로 안 짬) FEATURES 조합 방식만 바꿔 재검증.

| | baseline(44) | 추가(기존 실험) | 교체 |
|---|---:|---:|---:|
| relative_rate: diff+mean(2개 원본→2개 파생, 순수 선형 재매개변수화) | 734.49 | 711.93(44+3, diff/mean/product) | **708.79**(44개, diff+mean만) |
| relative_rate: diff+mean+product | 734.49 | 711.93 | **696.57**(45개) |
| uncertainty: smoothed_rate로 raw rate 대체 | 734.49 | 710.93(44+2) | **701.73**(44개) |

**기각, 그것도 "추가"보다 더 나쁘게.** 특히 diff+mean 교체 케이스는 `asof_pitcher_success_rate`/`asof_batter_success_rate` 2차원을 산술적으로 정보 손실 없는 45도 회전(diff=p-b, mean=(p+b)/2 → p=mean+diff/2, b=mean-diff/2로 완전 복원 가능)으로만 바꾼 건데도 708.79로 원본 축(734.49)보다 -25.70 나빴다. 즉 "정보량은 그대로인데 축만 회전시켜도" 트리 모델에는 손해다 — CatBoost의 축-평행 분기가 원본 축(투수 rate, 타자 rate 각각의 독립적인 marginal 효과)에서 이미 효율적으로 신호를 찾고 있었는데, 회전된 축(차이/평균)으로는 그 각각의 marginal 효과를 다시 재구성해야 해서 오히려 더 비효율적이 된 것으로 해석된다. **결론: 원본을 지우는 것 자체가 항상 정보 손실이 없어도 손해 — CatBoost에게는 "원본 축을 그대로 준다"는 것 자체가 이미 최적에 가까운 선택이었다.**

## E029 (2026-08-15) — R/M/O hazard의 실패유형 log-ratio를 corrector 메타피처로 추가 (`개발/v3_domain_experiments/segment_corrector_rmo_logratio_feature.py`) — 이 세션 최고 성적이지만 기준 미달, 보류

팀 깃헙 E17-A의 아이디어(실패 유형 reverse/middle/outside를 hazard 서브모델로 예측한 뒤, 예측값 자체가 아니라 "실패 유형 간 비율"을 메타피처로 쓴다)를 코드는 그대로 안 가져오고 아이디어만 참고해서 독립 재구현(대회 규정 §11 원칙). R/M/O 라벨은 이미 있는 `rmo_labels.py`(E002/E003에서 leak-safe 검증됨) 그대로 재사용, hazard 서브모델은 champion과 같은 CatBoost(44피처, train_df로만 학습), `rmo_log_ratio_mr = log((qM+eps)/(qR+eps))`, `rmo_log_ratio_or = log((qO+eps)/(qR+eps))` 2개를 만들어 기존 3-way corrector의 입력 피처(44개)에 추가.

| | 기존 3-way(메타피처 없음) | +log-ratio 메타피처 2개 | 차이 | pitcher-bootstrap z |
|---|---:|---:|---:|---:|
| PRIMARY | 801.93 | 811.45 | +9.52 | 1.71 |
| STRESS | 755.63 | 768.56 | +12.93 | **2.01** |

**이 세션에서 나온 어떤 신규 피처/라우팅 실험보다도 좋은 결과다** — 두 폴드 다 개선했고, STRESS는 관례적 유의 기준(z≈1.96)을 실제로 넘었다(E021의 z=1.51/1.13, E025의 z=-2.28~-3.91과 비교해보면 확실히 다른 급). 다만 **PRIMARY가 z=1.71로 기준에 살짝 못 미친다** — 이 프로젝트가 지금까지 지켜온 "두 폴드 다 유의해야 채택" 기준(E021 기각 근거와 동일선상)을 엄격히 적용하면 아직 채택 기준 미달이다.

**보류(기각 아님) 처리한 이유**: (1) `segment_corrector_base_pred_feature.py`(보조 정보를 corrector 피처에 추가하는 같은 카테고리)가 이전에 stress -34.45로 실패한 전례가 있는데, 이번엔 오히려 두 폴드 다 개선이라 그 실패 패턴과는 다르다. (2) hand_matchup(E021)이나 importance 변수(E025)와 달리 "실패 유형의 구성비"라는, 이전에 corrector에 넣어본 적 없는 새로운 정보축이다. (3) z=1.71은 완전히 무의미한 수준(예: E025의 유의하게 나쁜 음수 z)과는 다르고, PRIMARY 쪽 표본이 커서(수십만 행) 재현성 자체는 상대적으로 안정적일 가능성이 있다. **다음 단계로 남겨둠**: eps 값 스윕, log_ratio_or만 단독으로 넣어보기, corrector capacity(min_samples_leaf 등) 재조정 후 재검증 등으로 primary z를 유의 기준까지 끌어올릴 수 있는지 확인 — 그래도 안 넘으면 이 세션의 다른 사례들과 마찬가지로 최종 기각.
