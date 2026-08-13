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

---

## E018 (2026-08-13) — calibration 진단: E014 실LB 실패는 calibration 문제가 아님

E014(멀티모델 블렌드, 로컬 두 폴드 다 이겼는데 실제 LB -10.09)이 과신/과소신(calibration) 문제였는지 진단(`calibration_diagnostic.py`). champion(CatBoost단독)과 기각된 블렌드(0.6/0.2/0.2)의 corrector 적용 후 최종 예측 bias/slope 비교:

| | PRIMARY bias | PRIMARY slope | STRESS bias | STRESS slope |
|---|---:|---:|---:|---:|
| champion | -0.00069 | 1.0111 | -0.00132 | 1.2393 |
| 기각된 블렌드 | -0.00058 | 1.0060 | -0.00044 | 1.1592 |

두 폴드 모두 블렌드가 champion과 비슷하거나 오히려 더 잘 보정됨(slope가 1에 더 가까움, bias 더 작음). **calibration은 실패 원인이 아니다** — post-hoc compression 보정으로 되살릴 수 있는 문제가 아니고, LightGBM/XGBoost 자체의 2025 일반화가 약하다는 구조적 가설에 더 무게가 실림.
