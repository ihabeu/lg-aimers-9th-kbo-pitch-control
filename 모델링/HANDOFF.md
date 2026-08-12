# 프로젝트 현황 핸드오프 문서

**마지막 업데이트: 2026-08-12**

이 세션(또는 다른 AI/사람)이 처음부터 다시 파악할 필요 없이, 이 문서 하나로 지금까지 상태를 이해할 수 있도록 정리함. 상세 내용은 각 절에서 링크한 파일 참고.

## 대회 개요

LG Aimers 9기 온라인 해커톤(Phase 2). 투구 단위로 `control_success`(제구 성공 확률, 0~1) 예측. 평가지표는 Brier Skill Score:

```
Score = max(0, 100000 × (1 - Brier / (r(1-r))))
Brier = mean((p_i - y_i)^2)
r = 전체 평가 데이터의 평균 성공률 (비공개)
```

Phase2 수료 기준: Public LB 549.51 이상. 데이터 설명서는 [`data/data_description.md`](../data/data_description.md).

## 지금까지 실제 제출 이력

| 제출 | 모델 | 학습 데이터 | Public LB |
| --- | --- | --- | --- |
| 1차 | CatBoost | 2019~2023 (2024 누락 버그) | 662.85 |
| 2차 | CatBoost | 2019~2024 전체 (버그 수정) | **789.23 (현재 최선)** |
| 3차 | Elastic Net(L1+L2) | 2019~2024 전체 | 342.74 |
| 4차 | NN(embedding MLP) | 2019~2023 (2024 누락 버그, 동일 실수 반복) | 361.43 |
| 5차 | CatBoost + matchup 3종 | 2019~2024 전체 | 711.05 (로컬 703.10과 거의 일치, baseline 789.23 미달로 폐기) |
| 7차 | CatBoost + recent_k_pitch_rate(스냅샷 고정) | 2019~2024 전체 | **252.51** (로컬 966.18과 완전히 다른 방향 — 원인 규명 후 폐기, 아래 상세 참고) |
| 6차 | CatBoost + hand_matchup | 2019~2024 전체 | **787.40** (로컬 rolling OOT는 baseline 대비 +18.40/+21.06으로 확실히 개선이었는데 실제 LB는 오히려 -1.83 낮음 — local/actual 재괴리 사례. 제출 파이프라인 자체는 검증 완료: 모델 feature 순서/cat_features가 script.py와 완전 일치, in-sample 2024 재현 시 score=879.78·예측확률 분포 정상·hand_matchup 카테고리 4종 정상 생성 → 파이프라인 버그 아니라 진짜 2025 일반화 문제로 판단) |

**hand_matchup 최종 판정**: 제출 모델로는 채택 안 함(789.23 유지). 다만 "선수별 조건부 이력 정보"라는 가설 자체를 검증하려고 STEP 1(platoon feature) 4개를 독립적으로 테스트함 — A: pitcher_vs_current_batter_hand_rate 724.39(-10.10), B: batter_vs_current_pitcher_hand_rate 678.90(-55.59), C: pitcher_platoon_advantage 734.33(-0.16), D: batter_platoon_advantage 703.55(-30.94). 전부 baseline 미달로 **platoon 방향도 종료**. hand_matchup의 로컬 개선이 실제 LB로 이어지지 않은 것과 별개로, platoon 자체도 로컬에서부터 신호가 없었음(코드: [`platoon_features.py`](modeling/platoon_features.py), [`platoon_ablation.py`](modeling/platoon_ablation.py)).

**⚠️ 반복된 함정**: 최종 제출 모델은 반드시 2019~2024 **전체**로 재학습해야 한다. 검증(2019~2023 학습→2024 검증)에서 찾은 하이퍼파라미터/라운드수는 유지하되, 실제 제출 아티팩트는 전체 데이터로 다시 학습해야 함. CatBoost에서 한 번, NN에서 또 한 번 이 실수를 반복함 — 새 모델 만들 때마다 반드시 확인할 것.

## 현재 최선 모델 (제출 완료, 789.23)

- 위치: [`modeling/baseline_catboost.py`](modeling/baseline_catboost.py) + [`modeling/baseline_catboost.ipynb`](modeling/baseline_catboost.ipynb)
- 레시피: CatBoost, raw 44피처(전처리 없음, 결측치 네이티브 처리), `depth=6, learning_rate=0.05, l2_leaf_reg=15`, iterations는 2019-23→24 검증에서 찾은 값(204)으로 2019~2024 전체 재학습
- 제출 패키지: [`submit/submit.zip`](submit/submit.zip) (model/, script.py, requirements.txt)
- 로컬 검증(2019-23→24, 참고용 — 실제 제출 모델과는 다른 데이터로 학습된 모델의 점수임): Brier 0.247972, Score 734.49

## 데이터 핵심 발견 (사실 vs 추측 구분 — 상세는 [`INSIGHTS.md`](INSIGHTS.md))

**확인된 사실**
- train은 2019~2024, test는 **2025년** (한 번도 못 본 시즌 예측)
- `game_type`(R/F)의 F 성공률이 2019~2022엔 0.59~0.71로 높다가 **2023년에 0.46~0.47로 정확히 반전**, R은 완만하게만 하락 (`eda/eda2.ipynb`)
- `pitcher_team_id` 13개 중 10개는 정상, 22/23/25는 `F`에만 100% 등장하는 극소표본(292~4,437건), 13번은 F 비율이 38%로 유독 높음(다른 정상팀 3~10%)
- `asof_*` 이력 피처가 변수중요도 압도적 1위권, 순간 상황변수는 거의 무신호
- CatBoost native `type="Interaction"` 기준 최강 교호작용은 `season × asof_pitcher_success_rate` 등 — `season`이 거의 모든 rate 피처와 얽힘. `game_type`은 개별 중요도는 1위지만 교호작용 순위엔 없음

**추측 (미확인, 팀 문서에도 추측으로 표기)**
- R=1군 정규리그, F=퓨처스리그일 가능성 (trackman_history의 팀 코드 패턴 근거, 공식 매핑 없음)
- 2023년 반전 원인: KBO ABS(로봇심판) 도입 관련 추측 — 웹검색으로 확인해봤지만 타이밍이 명확히 안 맞아서 근거 약함(`data_description.md`엔 이런 내용 없음, 우리가 자체 조사). **모델링에 필요한 정보 아님, 더 파지 말 것**

## 실험 로그 — CatBoost 기준 (상세: [`INSIGHTS.md`](INSIGHTS.md) 5절)

**전부 baseline(734.49) 대비 실패, 원본 44피처 그대로 쓰는 게 최선으로 확정됨:**

| 시도 | 결과 |
| --- | --- |
| Regime 플래그 4개(is_extra_inning 등) | 719.5 |
| Recency weighting(λ=0.5) | 694.91 |
| 완전 중복 변수 제거 | 709.35 |
| 하이퍼파라미터 스윕(depth/lr) | 687~703 |
| XGBoost | 502~617 |
| EB smoothing(cold-start) | 711.45 |
| R/F 완전 분리 모델 | 623.31 |
| R/F 분리 + F만 최근(2023) 윈도우 | 635.96 (분리 계열 중 최선이지만 여전히 baseline 미달) |
| 5시드 앙상블 | 723.57 (seed=42가 5개 중 최고였음) |
| 학습 윈도우 축소(1~3년) | 632~701 (데이터 많을수록 계속 좋아짐) |
| 월 단위 세분화 검증(2024 상반기 포함 학습, 하반기만 검증) | 570.57 vs 동일 검증셋 기존방식 575.21 — 차이 없음(노이즈 수준), 기존이 약간 우세 |
| matchup 3종(team_matchup/hand_matchup/count_state) 추가 | 703.10 — 로컬은 미달이지만 [`submit_catboost_matchup/submit.zip`](submit_catboost_matchup/submit.zip)으로 실 리더보드 테스트용 별도 제출(로컬/실제 괴리 전례 있어서) |
| +pitcher_id 단독 | 567.84 |
| +batter_id 단독 | 659.27 |
| +pitcher_id+batter_id 같이 | 516.96 (단독보다 더 나쁨 — asof_* 이력 피처가 이미 선수 정보를 압축 제공하는데 raw ID를 더하면 과적합만 유발) |
| team_matchup 단독 | 664.15 (rolling OOT fixed-iteration으로도 재확인, 동일 결론: 폐기) |
| hand_matchup 단독 | single-split 745.72(+11.23), **rolling OOT(fixed-iteration, 올바른 방법론) 재검증: 2024(primary) 752.89(+18.40), weighted 839.71(+21.06 vs baseline 818.65), 2022도 같은 방향 개선(2316.31 vs 2257.03)** → **채택**, [`submit_catboost_handmatchup/submit.zip`](submit_catboost_handmatchup/submit.zip) |
| count_state 단독 | 700.73 (rolling OOT fixed-iteration으로도 재확인, 동일 결론: 폐기) |
| season categorical (baseline 위에 단독) | 488.36 |
| asof_pitcher/batter 상대값 3개(diff/mean/product) | 711.93 |
| Trackman 과거 투수 profile(T1, 12개, HIGH 매핑 332명) | 691.75 |
| Trackman 과거 투수 profile(T1, 매핑 범위 넓힘 477명) | 703.39 (매핑 확대해도 여전히 baseline 대비 -31, **Trackman 폐기**) |

| pitcher_success_rate × li | 701.49 |
| pitcher_recent_success_rate × li | 725.14 (가장 근접했지만 여전히 미달) |
| pitcher_success_rate × inning | 705.21 |
| feature selection: bottom 5 제거(39개) | 719.83 |
| feature selection: bottom 10 제거(34개) | 710.14 |
| feature selection: bottom 15 제거(29개) | 704.33 |

**결론**: CatBoost는 raw 피처만으로 이미 대부분의 패턴(교호작용 포함)을 스스로 찾음. 사람이 얹는 개입은 거의 다 손해. pitcher_id/batter_id는 A/B/C 세 조합 다 명확히 마이너스라 이 방향(raw ID, ID×hand 교호작용, season-categorical 추가조합) 폐기. Trackman historical profile도 매핑 품질을 개선해도(332명→477명) 여전히 큰 폭으로 미달이라 폐기. 상황×선수상태 교호작용 3개도 전부 미달. **feature selection도 제거할수록 단조롭게 악화**(`top_bottom`/`runner_on_1b/2b/3b`는 단독 importance 0인데도 제거하면 -14.66 — 개별 중요도가 0이어도 interaction에 쓰이고 있다는 뜻) → **44개 전부 유지 확정**.

## Rolling 폴드의 올바른 해석 (regime 1 vs regime 2)

2019~2022와 2023~2024는 서로 다른 regime이다(`game_type=F` 성공률 반전 등 — 위 "데이터 핵심 발견" 참고).
이걸 "2023이 이상한 해"로 단순화하면 안 되고, 아래처럼 구분해야 한다.

```
2019 ─ 2020 ─ 2021 ─ 2022 │ 2023 ─ 2024 │ 2025
────────── Regime 1 ──────┼─ Regime 2 ──┼─ Test
```

- **2019-23→24 (primary)**: Regime 2를 이미 관측한 상태에서 Regime 2의 다음 시점을 예측 — 실제 2025 예측 상황과 구조가 가장 비슷함. **로컬 검증의 주 기준(734.49)으로 계속 사용.**
- **2019-22→23 (secondary/stress test)**: Regime 1만 보고 Regime 2로의 전환 자체를 예측하는, 훨씬 어려운 별개의 문제. 여기서 나쁜 점수(예: hand_matchup의 -6.01, baseline 자체도 10.25로 붕괴)가 나와도 "그 feature가 나쁘다"는 근거로 쓰면 안 됨 — 애초에 이 폴드는 다른 문제를 풀고 있음.

**적용 원칙**: 어떤 feature를 primary(2024) 기준으로만 판단하고, 후보가 살아남으면 secondary(2023)로 regime 전환 상황에서도 무너지지 않는지 안정성만 참고로 확인한다(합/불 기준으로 쓰지 않음). 이번 세션에서 살아남은 후보가 없어서 실제로 이 2차 확인까지 갈 필요는 없었음.

## 800점 목표 라운드 (2026-08-12, local 2024 primary 기준)

hand_matchup의 로컬 개선(+18.40)이 실제 LB에서 뒤집힌(-1.83) 사건 이후, "local 2024 단일 폴드 800 돌파"를 새 목표로 설정. feature 추가보다 먼저 CatBoost 자체의 목적함수/표현력 개선을 시도:

| 실험 | Score | baseline(734.49) 대비 |
| --- | --- | --- |
| loss_function="RMSE" (CatBoostRegressor, Logloss 대신 직접 Brier 최적화) | 709.97 | -24.52 |
| max_ctr_complexity=1 | 741.59 | +7.10 (노이즈 범위, 유일한 플러스) |
| max_ctr_complexity=2 | 736.46 | +1.97 |
| max_ctr_complexity=4(기본값) | 734.49 | — |
| max_ctr_complexity=6/8 | 729.78 | -4.71 |
| border_count=128 | 713.62 | -20.87 |
| border_count=254(기본값) | 734.49 | — |
| border_count=512 | 713.49 | -21.00 |

**결론**: 목적함수/하이퍼파라미터 축에서는 800 근처에도 못 감. CTR complexity는 낮출수록(=categorical 조합을 덜 쓸수록) 나아지는 패턴이라 "categorical 조합 정보가 오히려 해롭다"는 이 세션 전체 패턴(team_matchup/count_state 실패)과 일치.

### Residual Corrector (CatBoost baseline + LightGBM residual, alpha blend)

leak-safe OOF(2020~2023 각각 이전 시즌만으로 학습한 CatBoost의 예측 residual)로 LightGBM residual 모델을 학습하고, 2024(완전히 학습에서 제외된 진짜 holdout)에 `p_final = clip(p_base + alpha*r_hat, 0, 1)`로 적용:

- **corr(진짜 2024 residual, r_hat) = 0.0083** — 사실상 0. residual 모델이 2020~2023에서 배운 패턴이 2024엔 안 통함.
- alpha sweep: 0.05→746.95, 0.1→748.93(피크, 노이즈 수준), 0.2→721.46, 0.3→652.09, 0.5→387.63, 0.7 이상→0. 상관관계가 0에 가까운 것과 일치하게 작은 alpha에서만 우연한 소폭 개선, alpha 커지면 급격히 붕괴.
- game_type=F의 regime별 residual: regime1(2020-22) 평균 +0.019 vs regime2(2023) **-0.203** — regime shift 자체는 극적으로 확인되지만, 이건 이미 세션 초반에 테스트하고 실패한 "R/F 완전 분리 모델"(623.31)/"R/F 분리+F만 최근 윈도우"(635.96)가 왜 안 됐는지를 재확인하는 결과일 뿐, 새로운 해법은 아님.

**결론**: Residual Corrector(단순 LightGBM 버전) 폐기. correlation 자체가 없어서 alpha를 더 정교하게 튜닝해도 소용없음.

### Trackman 매핑 v2 + 물리량/구종선택 피처 (전부 폐기)

`trackman_mapping_v2.py`: 월별 투구수 히스토그램만으로(구종비율을 fingerprint에 추가했더니 월별-only 매칭과 2.3%만 일치해서 노이즈로 판단, 제외) 헝가리안 알고리즘 전역 1:1 매칭. hand는 매칭에 안 쓰고 사후검증에만 사용 — 전체 hand 일치율 84.97%(v1 greedy 81.1%보다 개선), cost 상위 50%(395명, cost<=0.1462)에서 93.7%. many-to-one 충돌 구조적으로 0건.

| 실험 | Score | baseline 대비 |
| --- | --- | --- |
| T4: rel_speed_drift(최근3개월-커리어) | 702.87 | -31.62 |
| T4: spin_rate_drift | 700.68 | -33.81 |
| T4: induced_vert_break_drift | 713.36 | -21.13 |
| T4: all drift | 713.24 | -21.25 |
| 투수×타자손 구종군 선택 비율(fastball/breaking/offspeed) | 714.48 | -20.01 |

**결론**: 매핑 알고리즘(v1 greedy→v2 헝가리안)을 개선해도, feature 표현을 static profile→physical drift→situational pitch selection으로 계속 바꿔봐도 전부 baseline 미달. **Trackman feature engineering 이 세션에서 완전 종료** (사용자 결정규칙: "마이너스면 Trackman feature engineering 종료" 적용).

## 모델 구조 전환: 단일 CatBoost 개선 → 모델 다양성 + stacking

Trackman 종료 이후 방향 전환. 지금까지 실패 패턴(raw ID/matchup/uncertainty/drift/RMSE/CTR/residual corrector/Trackman 전부 baseline 미달)을 "44피처+CatBoost 조합이 이미 강한 로컬 최적점"으로 해석하고, feature 하나 더 찾기보다 **서로 다른 inductive bias의 모델들을 만들어 OOF residual correlation을 확인 → 상관 낮은 모델만 stacking**하는 방향으로 전환.

1단계: CatBoost/LightGBM/XGBoost를 동일 44피처로 2024 holdout에서 비교 + residual correlation 확인 (`model_diversity.py`).

| 모델 | standalone score | corr(residual, CatBoost) |
| --- | --- | --- |
| CatBoost | 734.49 | — |
| LightGBM | 592.33 | 0.9994 |
| XGBoost | 572.15 | 0.9993 |
| NN | 491.78 | 0.9984 |
| Logistic(Elastic Net) | 336.24 | 0.9982 |
| RandomForest | 429.98 | 0.9992 |

**결론**: tree 3종은 물론, 아키텍처가 완전히 다른 NN/Logistic/RandomForest까지 전부 CatBoost와 0.998+ 상관관계. 즉 남은 오차가 모델 특성이 아니라 44피처로는 설명 안 되는 노이즈에 가깝다는 강한 증거 — **stacking/model diversity 방향 종료**.

### Regime weighting (2020~22 학습 가중치 축소)

새 피처 없이 CatBoost `sample_weight`만 조정(2023~24=1.0 고정, 2020~22를 0.25~1.0으로 축소):

| 2020~22 weight | Score |
| --- | --- |
| 1.0(기본) | 734.49 |
| 0.75 | 716.58 |
| 0.5 | 710.73 |
| 0.25 | 729.83 |

**결론**: 전부 기본(weight=1.0) 미달. 과거 regime 데이터를 깎는 것도 도움 안 됨 — CatBoost가 `season`을 이미 활용해서 스스로 최근 데이터에 더 의존하고 있을 가능성.

### Regime-aware as-of rate (2023 이후만 / R·F 분리 / 교집합)

`asof_pitcher_success_rate`는 커리어 전체 누적이라 2020~2022(regime 1) 정보가 계속 섞여 들어간다는 가설로, train.csv 자체를 self-referential하게 재집계(leak-safe, (season,month) 이전만)해서 검증:

| 실험 | Score | baseline 대비 |
| --- | --- | --- |
| ① post2023_rate/diff | 695.01 | -39.48 |
| ② R/F 분리 rate | 694.68 | -39.81 |
| ③ post2023×R/F 교집합 | 694.34 | -40.15 |
| ①+②+③ 전부 | 700.68 | -33.81 |

**결론**: 전부 폐기. subset으로 쪼갤수록(post2023 70% NaN, F 91% NaN, 교집합은 더 희소) 커버리지가 급감해서 기존 전체누적 rate보다 신호가 희석됨. regime shift가 실재하는 건 맞지만(residual 분석에서도 확인) as-of 통계를 subset으로 재계산하는 방식으로는 활용 불가.

### Trackman 매핑 v3 (team crosswalk 추가) + 전체 컬럼 통합 + Bayesian 계층모델

v2 신뢰 매핑(395명)의 구단 정보로 crosstab을 뽑아보니 train `pitcher_team_id` 13개 중 10개가 실제 KBO 10개 구단과 뚜렷하게 대응됨(예: 12=두산, 17=한화, 18=삼성). 이 구단 crosswalk를 헝가리안 매칭의 추가 하드제약으로 넣은 v3는 hand 일치율을 93.7%→96.0%로 개선했지만, 그 매핑으로 repertoire 재검증해도 결과는 비슷(700.12/708.33/711.51/699.64/695.45) — **매핑 품질 문제가 아님을 재확인**.

`extension`, `zone_speed`, `pitch_of_pa`/`balls_before`/`strikes_before`(trackman 자체 상황 컬럼) 등 이전에 안 쓴 컬럼까지 포함해 release point 일관성(712.68, -21.81), 압박상황 반응(693.80, -40.69), 전체 28개 파생피처 통합(701.15, -33.34) 모두 실패. 통합 모델의 전체 72피처(baseline 44+Trackman 28) importance 순위에서 **Trackman 최고 피처가 19위**(상위 18개 전부 baseline) — Trackman이 숨겨진 신호가 아니라 애초에 baseline 대비 정보량이 낮다는 결론.

**마지막 확인**: baseline 44개는 그대로 두고 importance 상위 Trackman 피처만 골라서(top 3/5/10/15) 재테스트 — 703.36/705.54/701.41/707.10, 전부 -27~-33점으로 거의 동일. 몇 개를 넣든 결과가 똑같다는 건 "노이즈가 섞여서"가 아니라 최상위 피처조차 정보량 자체가 부족하다는 뜻. **Trackman은 매핑 3단계(v1→v2→v3) × 표현 8가지 × importance 선별까지 다 해봤고, 이 세션에서 완전히 소진됨.**

STEP 5: Beta-Binomial 계층 스무딩(pitcher/batter/game_type/hand_matchup을 log-odds 가산, CatBoost와 독립적인 별도 확률모델)도 standalone score 0.00(전체평균보다 나쁨), CatBoost와 residual 상관 0.987로 블렌드 가치도 없음.

### row_id가 진짜 시간순임을 발견 (중요, 향후 세션 참고)

`train.csv`에 날짜/게임ID가 없어서 이 세션 내내 (season, game_month) 단위가 leak-safe cutoff의 한계라고 가정했었는데, 실제로는 **`row_id`가 거의 완벽한 글로벌 시간순 인덱스**임을 확인함:
- season이 row_id 순으로 100% non-decreasing, 심지어 season별로 row_id 구간이 완전히 분리됨(2019: 1~237413, 2020: 237414~481500, ...)
- 같은 season 내에서 game_month도 99.999% non-decreasing
- 투수별로 row_id 정렬 시 `asof_pitcher_n`이 100% 단조증가(직접 확인) — 투수 단위 시퀀스 순서도 완벽히 보존됨
- 이닝이 이전 행보다 감소하는 지점을 게임 경계로 잡으면 경기당 투구수 분포가 실제 KBO 선발 평균과 일치(표본 확인: 평균 93.6구, 범위 49~120)

**이게 의미하는 것**: 이 세션 초반에 "불가능"으로 결론낸 within-game state(경기 내 투구수 등)가 실제로는 복원 가능했음. 다만 실제 시도 결과:
- `pitch_count_before`(경기 내 누적 투구수, 이닝-감소 휴리스틱으로 복원): 711.78 (-22.71, 실패 — 불펜투수는 이닝-감소 하나로 게임 경계를 못 잡아서 노이즈 섞임, max 767구 같은 비현실적 값 존재)
- LSTM 시퀀스 모델(투수별 직전 20구 성공/실패 시퀀스 인코딩 + 수치형 44피처 일부): **731.89**(최고 epoch, 실행별 편차 있음 664~732) — 이 세션 NN 계열 중 압도적 최고(기존 tabular NN 491.78 대비 +240). 그래도 baseline(734.49) 근소 미달, corr(CatBoost, LSTM)=0.9976로 여전히 높아 블렌드 가치는 낮음.

**다음 세션에서 시도해볼 만한 것**: 게임 경계 탐지를 이닝-감소 외에 다른 신호(pitch_of_pa 부재를 대체할 balls/strikes/outs 패턴, 또는 top_bottom 전환)로 보강하면 불펜투수 케이스가 나아질 수 있음. LSTM에 카테고리 임베딩(현재는 수치형만 사용) 추가하면 baseline을 넘을 가능성 있음 — 미시도.

### 🔥 recent_k_pitch_rate — 이 세션 최대 성과 (row_id = 실제 시간순 인덱스 발견)

`train.csv`에 날짜/게임ID가 없어 (season, game_month)가 leak-safe cutoff의 한계라고 가정했었는데, **`row_id`가 사실상 완벽한 시간순 인덱스**임을 확인함(season이 row_id 순 100% non-decreasing, 심지어 season별 row_id 구간이 완전 분리; 투수별 `asof_pitcher_n`도 row_id 순 100% 단조증가). 이닝-감소 지점을 게임 경계로 잡으면 경기당 투구수 분포도 실제 KBO 평균과 일치(표본 확인).

이 발견으로 **투구 단위**(경기 단위가 아니라) 최근성 피처를 처음으로 leak-safe하게 만들 수 있었음: `recent_{5,10,20,50}_pitch_rate` (해당 투수 직전 K구의 성공률). 기존 `asof_pitcher_prev1/3/5_game_success_rate`는 경기 단위라 한 경기 내 컨디션 변화를 못 잡는데, 이건 그 빈틈을 메움.

**결과 (rolling OOT, fixed-iteration)**:

| 폴드 | baseline | +recent_k(4개) | 차이 |
| --- | --- | --- | --- |
| 2022(w=0.2) | 2257.03 | 2589.49 | **+332.46** |
| 2023(w=0.3, stress) | 0.0 | 0.0(brier은 0.253102→0.250541로 개선) | — |
| 2024(w=0.5, primary) | 734.49 | **966.18** | **+231.69** |
| weighted | 818.65 | **1000.99** | +182.34 |

이 세션에서 나온 유일하게 **두 폴드(2022, 2024) 모두 큰 폭으로 일관되게 개선**된 결과 — hand_matchup(+18.40, 실전 실패)의 10배 이상. AUC도 확실히 상승(0.5485→0.5561 primary).

**발견 과정**: LSTM으로 투수별 직전 20구 성공/실패 시퀀스를 인코딩해서 신호 유무부터 확인 → permutation importance에서 이 시퀀스가 기존 어떤 단일 피처보다 압도적으로 중요(2위의 4.6배) → LSTM 없이 단순 rolling 평균으로 단순화해도 거의 같은 효과 확인 → CatBoost에 직접 투입.

**규정 준수**: "test.csv 내부 행 순서 기반 rolling/expanding 금지" 조항 때문에, 실제 제출(`submit_catboost_recent_k/`)에서는 투수별로 2024 시즌 마지막 시점의 스냅샷(`recent_k_snapshot.csv`, 792명)을 고정해서 2025 test 전체에 동일 적용 — 로컬 검증(2024 안에서 계속 갱신)보다 실제 효과는 작을 수 있음. **7차로 제출함.**

#### ⚠️ 7차 제출 결과: 실제 LB 252.51 — 심각한 실패, 원인 규명 및 최종 폐기

실제 점수 252.51(baseline 789.23은커녕 수료 기준 549.51도 미달). 원인 조사 결과:

1. **직접 원인**: 학습 시 recent_k 결측률은 0.1~1.3%로 극히 낮았는데(2024 안에서 연속 갱신했으니 대부분 값이 채워짐), 실전은 스냅샷(792명) 밖 투수가 많아 결측률이 훨씬 높았을 것으로 추정. 검증 실험: 2024 holdout에 recent_k를 인위적으로 30% 결측시키면 score 243.87 — 실제 252.51과 거의 일치. **CatBoost가 학습 때 거의 못 봤던 결측 분기가 실전에서 대량 발생해 완전히 오작동**.

2. **근본 원인**: 애초에 검증(로컬 966.18)과 실전(스냅샷 고정)이 **서로 다른 메커니즘**을 테스트하고 있었음 — 로컬은 "투구마다 실시간 갱신"(train.csv 안에서만 벌어지는 일이라 규정 위반 아님), 실전은 "시즌 마지막 스냅샷 고정"(규정상 유일하게 허용되는 방식). 이 둘의 성능이 이렇게 다를 거라고 제출 전에 검증했어야 했는데 안 함 — **제출 전 반드시 실제 배포 조건(스냅샷 고정)으로 재검증했어야 하는데 누락한 프로세스 실수**.

3. **수정 시도**: `recent_k_season_frozen.py` — 처음부터 "직전 시즌 말까지의 스냅샷"을 학습에도 똑같이 적용(시즌 안에서 갱신 안 함, train/test 완전히 동일한 메커니즘) + 결측 시 `asof_pitcher_success_rate`로 fallback. 결측률이 fallback 전 34.8%로 실전 추정치(~30%)와 거의 일치해서 원인 진단이 재확인됨. fallback 후 정직하게(2023년 말 스냅샷 고정 → 2024 예측) 검증한 결과: **700.93 (-33.56), 여전히 baseline 미달**.

**최종 결론**: recent_k_pitch_rate는 "실시간 갱신" 형태로는 강력한 신호이지만, 규정상 유일하게 허용되는 "시즌 경계 스냅샷 고정" 형태로는 신호가 거의 사라짐(오프시즌을 거치며 "작년 말 컨디션"이 "지금 컨디션"을 설명 못 함). **규정을 지키면서 쓸 수 있는 형태로는 이 아이디어를 살릴 수 없음 — 완전 폐기**. `submit/submit.zip`(789.23)으로 최종 확정.

**교훈(중요, 다음 세션 필독)**: 새 피처가 "현재 시점 기준 계속 갱신되는 정보"를 담고 있다면, 그 피처를 실제 제출에 넣기 전에 반드시 **실제 배포 조건(예: 시즌/기간 경계에서 고정된 스냅샷)으로 다시 검증**해야 한다. train.csv 내부 rolling 검증에서 잘 나온 점수는 그 자체로 제출 가능 여부를 보장하지 않는다 — "test에서 쓸 수 있는 방식인가"를 먼저 확인하고, 그 방식으로 처음부터 다시 검증해야 함.

## 세션 최종 결론 (2026-08-12)

이 세션에서 시도한 거의 모든 축 — raw ID, matchup, calibration, uncertainty/count, RMSE/CTR/border_count 하이퍼파라미터, Residual Corrector, Trackman(매핑 3단계 고도화 v1→v2→v3 × static/drift/situational/repertoire/entropy/separation/release consistency/pressure response 8가지 표현), model diversity(tree 3종+NN+Logistic+RF stacking), regime weighting/모델분리, regime-aware as-of rate(post2023/R·F 분리), Bayesian 계층모델 — **전부 baseline(734.49 로컬 / 789.23 LB) 대비 실패**. 특히 model diversity 실험에서 아키텍처가 완전히 다른 모델들조차 residual이 0.998+ 상관관계를 보인 것은, 남은 오차가 특정 모델의 약점이 아니라 현재 제공된 44개 피처로는 설명 불가능한 노이즈에 가깝다는 상당히 강한 증거. `submit/submit.zip`(789.23)을 최종 제출로 유지.

## 최종 결론 (2026-08-12 기준, 이 세션의 실험 큐 종료)

CatBoost baseline(raw 44피처, l2_leaf_reg=15, 전체 데이터 재학습, 로컬 734.49 / **실제 LB 789.23**)을 이기는 변형을 이 세션에서 시도한 모든 방향(raw ID, matchup, season categorical, calibration, Trackman, 상황×선수상태 교호작용, feature selection)에서 하나도 찾지 못함. `submit/submit.zip`이 최종 제출 기준. 수료 기준(549.51)은 이미 여유 있게 통과. 추가로 시도할 만한 것은 [`INSIGHTS.md`](INSIGHTS.md) 실험 로그의 "다음에 할 만한 것"만 남아있고, 우선순위 높은 항목은 이 세션에서 소진됨.

## Trackman ID 매핑 (pitcher_id ↔ pitcher_trackman_id)

DACON 운영진 Q&A(2026-08-07, DACON.GM 답변): "1) 트랙맨 데이터는 학습데이터 기간에 대해서만 제공되므로 가능합니다. 2) 문제없습니다." — pitcher_id/pitcher_trackman_id 대응 관계 추정 및 그 대응으로 만든 투구 이전 시점 Trackman 통계치를 피처로 쓰는 것 모두 운영진이 명시적으로 허용.

- `train.csv`엔 `game_date`/`game_id`/투구순번이 없어 개별 투구 1:1 row-level join은 불가능. 대신 투수별 (season, game_month) 투구수 히스토그램을 지문으로 최근접 이웃 매칭([`modeling/trackman_mapping.py`](modeling/trackman_mapping.py)).
- 품질 검증: `pitcher_hand`(고정 속성, 두 파일 모두 존재하나 매칭 신호로는 안 씀)로 사후 검증 — 무작위 매칭 기준선 64.8%, 지문 최근접 매칭 전체 81.1%, 확신도(1위-2위 거리 gap) 상위 25%는 97.0% → 노이즈 아닌 실신호 확인.
- hand 불일치는 확정적 오매칭이라 하드필터, many-to-one 충돌은 확신도 높은 쪽만 채택 → HIGH 332명(41.9%, train 행 기준 커버리지 68.4%).
- Trackman feature(구속/회전/무브먼트/릴리스/구종비율 12개, leak-safe cutoff: 현재 행의 (season,month) **이전** 데이터만 누적) 적용 결과가 매핑을 넓혀도(HIGH 332명→477명) baseline 대비 -31~-43으로 뚜렷이 낮아 **Trackman 트랙 종료**. 매핑/피처 코드는 [`modeling/trackman_mapping.py`](modeling/trackman_mapping.py), [`modeling/trackman_features.py`](modeling/trackman_features.py)에 남겨둠(재시도 대비).

## 다른 모델 (블렌드 후보, 단독으로는 CatBoost 미달)

| 모델 | 위치 | 로컬 검증 | 특이사항 |
| --- | --- | --- | --- |
| Elastic Net(L1+L2) | [`modeling/elastic_net.py`](modeling/elastic_net.py) + `.ipynb`, [`submit_nn/`](submit_nn/) | `game_type` 그대로: 0점(베이스라인 미달) → 교호작용 추가: 336점 → `game_type` 완전 제외: **385점(최선)** | 다중공선성 제거 필요(선형모델이라 CatBoost와 반대), 디트렌딩 실험 진행 중 |
| NN(entity embedding MLP) | [`modeling/nn_baseline.py`](modeling/nn_baseline.py), [`submit_nn/`](submit_nn/) | lr 스윕 435~612, dropout/weight_decay 스윕 227~616(최선: dropout=0.5, wd=1e-5 → 616.36) | StandardScaler vs MinMax 차이 없음(591.77 vs 588.22). 정규화 튜닝으로도 CatBoost 못 따라잡음 — L1 명시적 구현은 미완 |

## 완료된 추가 실험

- 월 단위 세분화 검증: 570.57 vs 기존방식 575.21(동일 검증셋) — 효과 없음
- R/F 비대칭 윈도우(R 전체+F만 2023): 635.96 — 여전히 baseline 미달
- **Elastic Net 디트렌딩**(`asof_pitcher_success_rate` 등 6개 rate에서 연도별 선형추세를 train으로만 학습해 제거): 337.55 — `game_type` 교호작용 버전(336.24)과 거의 동일(노이즈 수준), 여전히 `game_type` 완전 제외 버전(385.29, Elastic Net 최선)보다 낮음
- NN rolling OOT 검증(3폴드, CatBoost와 동일 가중치 0.2/0.3/0.5, epoch=3 고정): 2022=2038.77(비정상 고득점, CatBoost와 동일 패턴) / 2023=0.0(CatBoost처럼 베이스라인보다도 못함 — 2023 붕괴가 모델 무관하게 데이터 자체 특성임을 재확인) / 2024=534.85(주 지표, CatBoost 734.49에 여전히 못 미침) / 가중평균 675.18(2022 이상치 때문에 부풀려진 숫자, 참고만)

## 폴더 구조

```
lg aimers/
  data/                          # train.csv, test.csv, trackman_history.csv, data_description.md
  모델링/
    HANDOFF.md                   # 이 문서
    README.md                    # 데이터셋 개요 (초반 작성)
    INSIGHTS.md                  # 팀 공유용 상세 인사이트 + 실험 실패 로그
    eda/
      eda.py, eda.ipynb           # 컬럼 분류(식별자/수치형/시계열형/이진형/범주형), 결측치, 상관관계, 변수중요도
      eda2.py, eda2.ipynb          # 2023년 반전 현상 집중 분석
      EDA.md, COLUMNS.md
    modeling/
      baseline_catboost.py/.ipynb  # 현재 최선 (789.23)
      elastic_net.py/.ipynb        # 선형모델, 블렌드 후보
      nn_baseline.py                # NN, 블렌드 후보
      recency_weighted_catboost.py  # recency weighting 실험(실패)
      xgb_quick.py                  # XGBoost 실험(실패)
      models/                        # 학습된 모델 아티팩트
    submit/          # CatBoost 제출 패키지 (현재 최선, 789.23)
    submit_elastic_net/  # Elastic Net 제출 패키지
    submit_nn/            # NN 제출 패키지
  ANALYSIS_NOTES 등은 참고한 다른 팀(다른참가자/LGAIMERS) 레포 자체 파일이 아니라
  이 프로젝트에서 그 레포를 참고해 얻은 인사이트가 위 INSIGHTS.md/HANDOFF.md에 녹아있음

참고 레포: https://github.com/다른참가자/LGAIMERS (E0~E18 실험, 이 프로젝트가 방법론적으로 여러 번 참고함 —
regime 교호작용, EB smoothing, rolling OOT 가중치 0.2/0.3/0.5 등)
```

## 코드 작성 원칙

로직은 `.py`에 함수로, `.ipynb`는 그 `.py`를 import해서 셀 단위로 실행 + 결과/의도를 마크다운으로 남김 (jupyter nbconvert --execute로 미리 실행해서 출력 포함해서 저장). 제출 패키지는 `submit_문자열/` 형태 폴더에 `model/`, `script.py`, `requirements.txt` — 압축 해제 후 재실행까지 검증하고 전달.

## 다음에 할 만한 것

이 세션에서 계획했던 실험 큐(ID 계열 → matchup → calibration → Trackman → 상황×선수상태 → feature selection)를
전부 소진했고 전부 baseline(734.49 로컬 / 789.23 LB) 대비 실패해서 **`submit/submit.zip`을 최종 제출로 확정**.
Elastic Net/NN 블렌딩도 단독 성능 격차가 너무 커서(385, 616) 시도하지 않기로 결정함(위 실험 로그 참고).

추가로 시도할 게 남았다면(우선순위 낮음, 확정된 다음 계획 아님):
- Trackman 매핑을 fingerprint 외 다른 신호(구종 패턴, 활동 시기)로 보강해서 재시도
- hand_matchup을 2020~2022→2023(secondary/stress) 폴드에서도 재확인 후 재검토 (위 "Rolling 폴드의 올바른 해석" 참고)
- 수료 기준(549.51)은 이미 여유 있게 통과했으므로, 남은 시간은 Phase 3 대비 코드/PPT 정리에 쓰는 것도 고려 가능
