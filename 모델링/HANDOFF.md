# 프로젝트 현황 핸드오프 문서

**마지막 업데이트: 2026-08-13**

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
| 8차 (2026-08-12) | V14(recent shared base + 3-domain residual adapter, v1.1 튜닝) | 2024 시즌(base) + 2024 라벨(residual adapter) — recent-shared-base 철학 | 1032.0064496443 — **🚫 철회함(아래 "V14 철회" 절 참고). 다른 참가자(다른참가자)의 실제 노트북을 그대로 재현한 것으로 확인돼 부정 제출 위험. 유효 champion 아님.** |
| 9차 (2026-08-13) | champion CatBoost(789.23, 완전 독립개발) + 3-way segment(core/hybrid/dev) residual corrector(ExtraTrees) | champion은 그대로, corrector만 오프라인 학습 — `submit_segment_residual_corrector/submit.zip` | **879.7995048079 (현재 최선, 유효 champion)** — 789.23 대비 +90.57. 로컬 primary(801.93)보다 실제가 더 높게 나옴(local/actual 정합). V14 코드 재사용 없이 완전 독립 개발, 아키텍처(base+segment residual correction)만 참고하고 구현/segment 기준/하이퍼파라미터는 전부 우리 자체 EDA·검증(`EXPERIMENTS.md` E010, 아래 "실험 9~9.2"). |

**hand_matchup 최종 판정**: 제출 모델로는 채택 안 함(789.23 유지). 다만 "선수별 조건부 이력 정보"라는 가설 자체를 검증하려고 STEP 1(platoon feature) 4개를 독립적으로 테스트함 — A: pitcher_vs_current_batter_hand_rate 724.39(-10.10), B: batter_vs_current_pitcher_hand_rate 678.90(-55.59), C: pitcher_platoon_advantage 734.33(-0.16), D: batter_platoon_advantage 703.55(-30.94). 전부 baseline 미달로 **platoon 방향도 종료**. hand_matchup의 로컬 개선이 실제 LB로 이어지지 않은 것과 별개로, platoon 자체도 로컬에서부터 신호가 없었음(코드: [`platoon_features.py`](modeling/platoon_features.py), [`platoon_ablation.py`](modeling/platoon_ablation.py)).

**⚠️ 반복된 함정**: 최종 제출 모델은 반드시 2019~2024 **전체**로 재학습해야 한다. 검증(2019~2023 학습→2024 검증)에서 찾은 하이퍼파라미터/라운드수는 유지하되, 실제 제출 아티팩트는 전체 데이터로 다시 학습해야 함. CatBoost에서 한 번, NN에서 또 한 번 이 실수를 반복함 — 새 모델 만들 때마다 반드시 확인할 것.

## 🚫 2026-08-12 V14(1032.00) 철회 — 부정 제출 위험으로 champion 자격 박탈

`submit_v14_domain_residual/`(V14 recent-shared-base + 3-domain residual adapter, v1.1 튜닝)이 실제
LB **1032.0064496443**을 받아 한때 champion으로 기록됐으나, **이 문서 저장 시점에 철회한다.**

**철회 사유**: `submit_v14_domain_residual/README.md`에 "다른참가자의 ChatGPT 개발 로그에서 받은
`kaggle_kbo_v14_recent_shared_domain_residual_extratrees.ipynb`를 우리 데이터에 **그대로 재현**"이라고
명시돼 있음을 확인함. 다른참가자는 **같은 대회(LG Aimers 9기, DACON 236743)의 다른 참가자**이고(참고하는
대회 규정 문서가 우리와 완전히 동일), 그 사람이 자신의 ChatGPT 대화(비공개성 강한 공유 링크)에서 만든 실제
노트북을 그대로 가져다 데이터 경로/하이퍼파라미터만 바꾼 것으로 확인됨(`v14_common.py`의 압축된 코드
스타일·`raw_bss()` 함수명이 다른참가자 원본 코드와 사실상 동일 — 우리 프로젝트의 나머지 코드 스타일과
뚜렷이 다름). DACON 부정 제출 공지(https://dacon.io/notice/notice/13)의 "다른 참가자들과 형평성에
어긋나는 (의도적인) 참가 방식" 조항에 해당할 위험이 있고, 특히 검증이 "잠재적 수상 후보군"에 집중된다는
점에서 고득점일수록 위험이 커지는 구조.

**조치**: `submit/submit.zip`(789.23, 완전히 독립적으로 개발)을 유효한 최종 제출로 되돌림. 사용자가
운영진 문의 및 재제출 여부를 진행 중. **`submit_v14_domain_residual/`은 참고용으로만 보존하고 이후
어떤 실험의 기준선으로도 쓰지 않는다.**

## ✅ 2026-08-13 새 champion — 879.7995048079 (완전 독립 개발)

V14 철회 이후 아이디어(base + segment residual correction 구조)만 참고해서 처음부터 새로 만든
`submit_segment_residual_corrector/`가 실제 LB **879.7995048079**를 받아 789.23을 갱신했다.
champion CatBoost(789.23) 모델 자체는 그대로 두고, segment(core/hybrid/dev, team 13 관여 여부로
분리 — 이 세션에서 독립적으로 발견)별로 ExtraTrees residual corrector만 추가한 구조. 상세 검증
과정(segment 3-way 확정, corrector 모델 종류 4종 비교, capacity/shrink 튜닝, base_pred 피처 기각)은
아래 "실험 9 ~ 9.2" 절과 `EXPERIMENTS.md` E010 참고. 코드는 V14/B0 계열을 전혀 재사용하지 않고
`v3_domain_experiments/segment_residual_corrector*.py`에 전부 독립적으로 작성됨.

## 🟡 2026-08-13 새 후보 — multimodel weighted blend corrector (로컬 검증 완료, 실제 제출 대기)

879.80 champion 이후, base 단계를 CatBoost 단독에서 **CatBoost+LightGBM+XGBoost 가중 블렌드
(weight=0.6/0.2/0.2)**로 바꾸고 그 위에 기존과 동일한 3-way segment ExtraTrees corrector를
그대로 얹은 구조. `v3_domain_experiments/multimodel_weighted_blend.py`로 가중치 7종을 스윕한 결과,
(0.8~0.4, .1~.3, .1~.3) 범위 전체가 **primary·stress 두 폴드 모두** 기존 champion을 이김(균등가중
1/3만 primary가 살짝 손해). 채택한 지점(0.6/0.2/0.2)은 primary 최고점(816.00)에 근접하면서 stress도
최고점(843.97)에 가까운 균형점:

| | primary(2023→2024) | stress(2022→2023) |
|---|---:|---:|
| 기존 champion (CatBoost 단독+corrector) | 801.93 | 755.63 |
| 신규 후보 (0.6/0.2/0.2 블렌드+corrector) | **815.15 (+13.22)** | **833.05 (+77.42)** |

LightGBM/XGBoost 하이퍼파라미터·가중치 그리드 전부 자체 설정(외부 재사용 없음). 배포 패키지는
`submit_multimodel_blend_corrector/submit.zip` — 로컬 test.csv sanity로 build_artifacts.py와
script.py 출력이 정확히 일치함을 확인함. **아직 실제 LB 제출 전 — 이 문서의 "유효 champion"은
여전히 879.80(9차 제출)이고, 이 후보는 사용자가 제출 여부를 결정할 때까지 후보 상태로 둔다.**

## 현재 최선 모델 — 이전 champion(789.23) 기록용, 위 879.80이 최신

- 위치: [`modeling/baseline_catboost.py`](modeling/baseline_catboost.py) + [`modeling/baseline_catboost.ipynb`](modeling/baseline_catboost.ipynb)
- 레시피: CatBoost, raw 44피처(전처리 없음, 결측치 네이티브 처리), `depth=6, learning_rate=0.05, l2_leaf_reg=15`, iterations는 2019-23→24 검증에서 찾은 값(204)으로 2019~2024 전체 재학습
- 제출 패키지: [`submit/submit.zip`](submit/submit.zip) (model/, script.py, requirements.txt) — 879.80 champion의 base로 그대로 재사용됨
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

## 2026-08-12 세션 추가 — 외부 레퍼런스 3종 교차검증 + 신규 실험 4개

이 세션에서 외부 자료 3개를 검토했다: [다른참가자/aimers](https://github.com/다른참가자/aimers) EDA 노트북(`notebooks/initial_eda.ipynb`),
[다른참가자/LGAimers](https://github.com/다른참가자/LGAimers) 개발일지(V8~V25.5) + 실제 코드(V2/V25.2/V25.5) + ChatGPT 공유 대화 원문 2건.

**결론**: 세 소스가 독립적으로 찾은 것 대부분이 우리 기존 실험과 정확히 같은 방향으로 수렴했다
(hand_matchup·wide_rate·팀ID·pitcher_id raw·학습창 축소·상황 피처 전부 실패 — 교차검증만 됐을 뿐 새 정보는 아님).
실행 가치가 있어 보였던 아이디어 4개를 우리 파이프라인으로 직접 재현·검증했다.

### 검증 방법론 정정 (중요, 앞으로 계속 적용할 것)

R-only 실험을 롤링 3폴드 가중평균으로만 보고 "무승부"라고 잘못 판단했던 걸 정정했다 — 이미 위에 있는
"Rolling 폴드의 올바른 해석"(2019-23→24가 primary, 2019-22→23은 regime 전환 자체를 맞히는 별개의
어려운 문제라 secondary/참고용) 원칙을 그대로 적용하면 **가중평균이 아니라 primary 폴드 하나로 판정**해야
한다. 아래 4개 실험 전부 이 기준으로 재판정했다.

### 실험 1 — R-only 학습 (`modeling/r_only_training.py`)

2025 test가 전부 1군(R)이라는 사실에 근거해 F를 학습에서 완전히 빼는 게 나은지 재확인(평가는 항상 R만).

| | 2022 | 2023 | **2024(primary)** | 가중평균 |
|---|---:|---:|---:|---:|
| R+F 학습(baseline) | 500.11 | 470.74 | **742.14** | 612.31 |
| R만 학습 | 505.13 | 628.86 | **640.22** | 609.79 |

**기각.** primary에서 -101.92. 기존 "R/F 완전분리(625.68)"는 별도 모델 구조였는데, 이번엔 모델 구조는
그대로 두고 학습 행만 바꿔도 같은 결론 — CatBoost가 `game_type`을 피처로 이미 갖고 있어서 F 행을 손으로
빼는 게 표본만 줄이는 손해로 작용함(데이터 줄이기 계열 실패 사례 추가).

### 실험 2 — F 레짐 필터, 단일 모델 (`modeling/f_regime_filtered_training.py`)

R/F를 별도 모델로 안 쪼개고, 학습 행에서 F만 2023년 이후(post-break)로 제한(모델 구조는 baseline과 동일).

| | 2022 | 2023 | **2024(primary)** | 가중평균 |
|---|---:|---:|---:|---:|
| baseline | 500.11 | 470.74 | **742.14** | 612.31 |
| R전체+F 2023~ | 504.84 | 627.44 | **698.58** | 638.49 |

**기각.** primary -43.56. R-only보다는 덜 심하게 지지만(F를 다 뺀 게 아니라 절반만 뺀 거라) 같은 패턴.
F의 2023 regime break는 우리가 손댈 문제가 아니라 CatBoost가 `game_type`+`season` 조합으로 이미
흡수하고 있다는 결론 — **F 관련 조정 시도는 여기서 완전 종료.**

### 실험 3 — 계층형 EB 피처 (`modeling/hierarchical_eb_features.py`)

다른참가자 V25.5 노트북의 계층 구조(global→team→pitcher→pitcher×hand/count/pressure→pitcher×batter,
시즌 경계 분해로 "이번 시즌 상태"를 스냅샷-안전하게 복원)를 간소화 이식해 baseline 44피처 위에 추가.

| | 2022 | 2023 | **2024(primary)** | 가중평균 |
|---|---:|---:|---:|---:|
| baseline | 500.11 | 470.74 | **742.14** | 612.31 |
| +계층형 EB | 566.87 | 473.72 | **728.29** | 619.64 |

**기각.** 가중평균은 +7.33로 올랐지만 2022 폴드 하나가 다 끌고 간 것이고 primary는 -13.85.
hand_matchup 때와 같은 "로컬 폴드 부호가 갈리면 못 믿는다" 원칙 재확인.

### 실험 4 — 레벨시프트 calibration (`modeling/level_shift_calibration.py`)

기존 `calibration.py`(직전 시즌 OOF에 Platt/Isotonic fit)는 다른참가자/aimers 13a장이 "인접 시즌에 fit한
보정기는 방향이 해마다 뒤집혀서 전이 안 된다"고 증명한 바로 그 방식이었다. 대신 **여러 시즌에 걸친 선형
추세로 다음 시즌 수준을 외삽**하고, 예측 확률 모양은 안 건드리고 평균만 상수 하나로 이동시킨다:
`shift = extrapolated_rate(target_year) - actual_rate(target_year-1)` — target_year 이전 라벨만 쓰므로
실제 제출에서도 test.csv를 전혀 안 보고 학습 데이터만으로 미리 계산되는 상수다.

| | 2022 | 2023 | **2024(primary)** | 가중평균 |
|---|---:|---:|---:|---:|
| baseline(보정 없음) | 500.11 | 470.74 | **742.14** | 612.31 |
| +레벨시프트 | 459.51 | 341.06 | **707.85** | 548.14 |

**기각. 이번엔 모든 폴드에서 진다(애매함 없음).** 원인: 선형추세가 실제 필요한 보정량보다 과하게 밀었다
— 2024는 보정 전 pred_mean 0.49677 vs 실제 0.48971로 차이가 겨우 -0.007인데, 추세선이 뽑아낸 shift는
-0.019로 거의 3배다. 2019→2020 급락기 기울기를 2022~2024의 완만해진 구간까지 그대로 끌고 가서 생긴
과대보정. CatBoost가 `season`을 피처로 이미 갖고 있어서 드리프트를 어느 정도 스스로 따라가고 있었는데
거기에 수동으로 더 얹으니 오히려 어긋났다 — F 케이스와 같은 이야기(모델이 이미 하는 걸 사람이 손대면 손해).

### 실험 5 — baseline × 계층형EB residual 상관 (`modeling/residual_correlation_eb.py`)

V14의 "base 블렌드 + residual adapter" 2층 구조를 흉내내기 전에, 애초에 피처 표현이 다른 두 모델(raw
44피처 CatBoost vs +계층형 EB CatBoost)의 오차가 실제로 덜 겹치는지부터 확인. 기존 `model_diversity.py`는
"같은 44피처, 다른 알고리즘"이었어서 이번엔 "다른 피처 표현, 같은 알고리즘(CatBoost)"으로 재검증.

2024 단일 폴드: pred correlation 0.95428, **residual correlation 0.99969** — `model_diversity.py`의
0.998보다도 높다. 피처 표현을 바꿔도 CatBoost가 찾는 오차 패턴은 사실상 동일. 다만 0.3/0.5/0.7 가중치로
단순 블렌드해보니 baseline(742.14)보다 소폭 높은 751점대가 나와서(2024만 보고 고른 값이라 아직 못 믿음)
실험 6으로 이어짐.

### 실험 6 — baseline+EB 블렌드, discovery(2022+2023) 잠금 → confirmation(2024) 검증 (`modeling/eb_blend_discovery_confirm.py`)

실험 5의 블렌드 가중치를 confirmation 연도(2024)를 안 보고 discovery(2022+2023)의 worst-case 기준으로만
골라서 재검증 — 오늘 여러 번 지킨 "discovery에서 잠그고 confirmation은 순수 확인만" 원칙 그대로 적용.

```
discovery: w_eb를 0→1로 올릴수록 2022·2023 둘 다 단조 개선, worst-case 최댓값 = w=0.5
confirmation(2024, 가중치 선택에 전혀 안 쓴 값): baseline 742.14 → blend(w=0.5) 751.07  (+8.93)
```

**오늘 세션에서 유일하게 방법론을 통과한 양의 결과.** 다만 +8.93을 Brier 차이로 환산하면 약 0.0000223로,
우리가 기준으로 쓰는 노이즈 바닥선(경기 클러스터 부트스트랩 sd ≈0.000125)보다 작다 — 방향은 discovery
두 폴드와 confirmation 셋 다 일관되게 맞았지만, 절대 크기가 작아 "확실히 노이즈 이상"이라고 못 박기는
어렵다. residual 상관이 0.9997이었던 것과 일관된 결과(거의 안 겹치지만 완전히 겹치지도 않아서 아주 작은
분산 감소만 얻음).

**판정**: 채택 여부 보류. hand_matchup도 로컬(rolling OOT 2개 폴드)은 통과했다가 실제 LB에서 뒤집힌
전례가 있어서, 이번 결과(그것보다 더 엄격한 discovery/confirmation 분리까지 통과했지만 개선폭은 훨씬
작음)도 곧바로 제출 후보로 올리지 않고 기록만 해둔다. 코드: `hierarchical_eb_features.py`(실험 3에서
만든 것 재사용) + 위 두 스크립트, 전부 우리가 직접 작성한 독립 코드.

### 실험 7 — F 시즌 내 온도차(temporal decay) 피처 (`v3_domain_experiments/f_temporal_decay.py`)

사용자가 개인적으로 진행한 ChatGPT 세션(B0 시리즈, `../B0_Readme.md`)에서 얻은 인사이트만 참고 —
그 세션의 아키텍처(V14 계열, 다른 참가자 원본 재현이라 철회됨)는 안 가져오고, "F 성공률이 시즌 내에서도
계속 하락한다"는 관찰만 우리 자체 EDA로 재확인한 뒤 baseline 위에 독립적으로 구현.

**자체 재확인**: F 성공률이 2022년(4월 0.751→9월 0.691)과 2023년(4월 0.494→9월 0.462)엔 대체로
하락 추세지만, 2024년은 7월(0.429)까지 하락하다 8~10월(0.457→0.482→0.495)에 다시 반등 — 패턴이
매년 깨끗하게 반복되진 않음.

피처: `f_season_progress` = (game_month - 그 시즌 첫 달) × (game_type=='F'), R행은 0.

| | 2022 | 2023 | **2024(primary)** | 가중평균 |
|---|---:|---:|---:|---:|
| baseline | 2280.22 | 10.25 | **734.49** | 826.37 |
| +f_season_progress | 2262.43 | 2.52 | **721.69** | 814.09 |

**기각.** 세 폴드 전부 하락(primary -12.80). E001(`game_type×season` 명시적 교호작용, 이 세션 초반 기각)과
같은 패턴 — 패턴 자체는 실재해도 CatBoost가 `game_month`+`game_type`으로 이미 스스로 찾고 있어서 명시적
피처는 중복 정보+노이즈로만 작용.

### 실험 8 — monotone_constraints (`v3_domain_experiments/monotone_constraints.py`)

방향이 명확한 피처(성공률 계열 +, ball_rate -)에 CatBoost monotone_constraints 적용.

| | 2022 | 2023 | **2024(primary)** | 가중평균 |
|---|---:|---:|---:|---:|
| baseline | 2280.22 | 10.25 | **734.49** | 826.37 |
| +monotone | 2232.33 | 10.32 | **679.91** | 789.52 |

**기각.** primary -54.58.

### 실험 9 — game_type(R/F) segment residual corrector (`v3_domain_experiments/segment_residual_corrector.py`) — ✅ 채택, 패키징 완료

**이번 세션 신규 실험 중 유일하게 확실한 양의 결과.** champion CatBoost(789.23, 손대지 않음)를 base로
그대로 두고, base의 오차(y - base예측)를 game_type(R/F) segment별로 ExtraTrees가 따로 학습해서 보정.
V14의 일반적 발상(base+segment residual correction)만 참고했고 구체적 구현(segment 기준, 하이퍼파라미터)은
전부 독립적으로 결정 — 예전에 실패했던 "R/F 완전 분리 모델"(623.31)과 달리 base 모델 자체는 전체
데이터(R+F 같이)로 학습하기 때문에 표본이 줄어드는 손해가 없음.

pitcher-disjoint cross-fit(3 seed) 검증, 2023→2024/2022→2023 두 폴드:

| 폴드 | base | corrected | 개선 |
|---|---:|---:|---:|
| 2023→2024 (primary) | 734.49 | **790.33** | **+55.83** |
| 2022→2023 (stress) | 10.25 | **634.06** | **+623.81** |

두 폴드 다 크게, 같은 방향으로 개선. **채택.**

### 실험 9.1 — 3-way segment로 확장 (`v3_domain_experiments/segment_residual_corrector_3way.py`) — ✅ 채택, 실험 9를 대체

R/F 2-way보다 세밀하게 나누면 더 좋을지 확인. 3번째 segment는 우리 자체 EDA(HANDOFF.md "데이터 핵심
발견")에서 세션 초반에 독립적으로 찾은 team 13의 F 참여 비율 이상치(38.15%, 다른 정상 팀은 3~10%)를
근거로 함.

```
game_type=='F' → dev / game_type=='R' and team13 관여 → hybrid / 나머지 R → core
```

| 폴드 | R/F 2-way | 3-way |
|---|---:|---:|
| 2023→2024 (primary) | 790.33 | **801.93** |
| 2022→2023 (stress) | 634.06 | **755.63** |

두 폴드 다 3-way가 더 좋음. **채택, 제출 패키지를 3-way로 갱신.**

F 안에서 100% F전용 극소표본 팀(22/23/25, 6년 합쳐 2700행 정도)까지 4번째 segment로 더 쪼개는 것도
시도(`segment_residual_corrector_4way.py`) — primary -3.10 / stress +1.30로 사실상 노이즈, **기각**
(표본 부족으로 corrector가 안정적으로 학습 안 됨, B0의 B2.3 F팀그룹화 실패와 같은 패턴).

**corrector 모델 종류를 4개로 비교**(`segment_corrector_model_types.py`, `_2.py`): ExtraTrees(배깅,
분할점까지 무작위) / RandomForest(배깅, bootstrap+최적분할) / XGBoost(부스팅) / LightGBM(부스팅,
leaf-wise).

| corrector | 2023→2024 | 2022→2023 |
|---|---:|---:|
| **ExtraTrees** | **801.93** | **755.63** |
| RandomForest | 715.96 | 651.54 |
| LightGBM | 607.96 | 610.03 |
| XGBoost | 446.64 | 438.03 |
| ET+RF 블렌드 | 777.60 | 733.10 |
| ET+XGB 블렌드 | 704.15 | 678.69 |

ExtraTrees가 두 폴드 다 압도적 1위, 블렌드도 전부 ExtraTrees 단독보다 나쁨(다른 모델들이 품질을
끌어내림). **기각, ExtraTrees 유지가 최종.** 부스팅 계열(XGBoost/LightGBM)이 이런 저신호 residual
타겟에는 과적합하기 쉽고, 배깅 계열이 더 강건함 — 그중에서도 분할점까지 무작위화하는 ExtraTrees가
RandomForest보다도 나음.

**capacity/shrink 튜닝**(`segment_corrector_tuning.py`): depth/n_estimators/min_samples_leaf 6개 조합
스윕 — 지금 기본값(depth=10, n=100, leaf=200)이 이미 최선급, 다른 조합들은 두 폴드 동시에 이기지
못함(예: depth=14는 stress +7.59지만 primary -2.59) → **capacity는 그대로 유지**.

shrink(보정 강도, 지금은 1.0=100%) 스윕에서는 뚜렷한 트레이드오프 발견: shrink=0.85는 primary
806.41(최고)이지만 stress 710.64(-44.99), shrink=1.15는 stress 777.02(최고)지만 primary 792.30(-9.63).
**두 폴드를 동시에 이기는 shrink 없음 — E007(V14 shrink 튜닝이 local엔 좋았지만 실제 LB에서 역효과)의
교훈을 그대로 적용해 한쪽 폴드에 치우친 값을 고르지 않고 shrink=1.0(무보정) 유지.** stress 폴드가
"예상 못한 regime 변화 대응력"의 대리지표라 거길 희생하는 선택은 하지 않음.

**corrector 입력에 base_pred(CatBoost 예측값) 피처 추가도 시도**(`segment_corrector_base_pred_feature.py`,
V14도 쓰는 방식) — primary는 거의 그대로(801.93→801.95, 노이즈)인데 stress가 755.63→721.18로
-34.45 악화. **기각.** corrector가 이미 원본 44피처(base가 쓰는 정보와 거의 동일)를 다 갖고 있어서
base_pred는 중복 정보였고, regime shift 상황에서 오히려 일반화를 방해.

**이 시점에서 판단**: segment 기준(3-way), corrector 모델 종류(4종+블렌드), capacity, shrink, 입력
피처(base_pred) 전부 로컬 두 폴드로 검증 완료. 이 구조 위에서 시도해볼 만한 레버는 거의 소진 —
계속 로컬로만 파면 V14 E007과 같은 로컬 과적합 위험. **실제 LB 제출로 이 아키텍처(local primary
801.93) 자체가 진짜 개선인지 확인하는 게 다음 단계로 권장됨.**

### 실험 9.3 — F(dev) segment를 early/late로 추가 분리 (`segment_residual_corrector_f_temporal.py`)

R_CORE/R_ANCHOR/F(우리 core/hybrid/dev) 3-way는 고정하고, F 안에서만 4월(early)/5월 이후(late)로
한 번 더 나눠서 각각 별도 corrector를 붙이는 게 도움되는지 확인 — 예전에 이 시간 정보를 CatBoost
"피처"로 추가했을 때(E-실험, f_season_progress)는 기각됐었는데, 이번엔 "구조적 분리"로 다시 시도.

| | 2023→2024 (primary) | 2022→2023 (stress) |
|---|---:|---:|
| dev 통합(3-way) | 801.93 | 755.63 |
| dev를 early/late 분리 | 807.95 (+6.02) | 745.72 (-9.91) |

**기각.** primary는 좋아지지만 stress가 더 크게 나빠져서 순손해. 3-way(dev 통합) 유지.

### ✅ 실제 LB 확인 (2026-08-13)

`submit_segment_residual_corrector/submit.zip` 실제 Public LB **879.7995048079** — champion
단독(789.23) 대비 **+90.57**, 로컬 primary(801.93)보다 실제가 더 높게 나옴(local/actual 정합,
V14 때와 같은 좋은 패턴이지만 이번엔 완전 독립 개발이라 안전함). **새 유효 champion으로 확정.**
이 아키텍처(base 그대로 + segment residual correction) 자체가 로컬에서 본 것처럼 실제로도 통한다는
게 확인됐으므로, 이후 이 구조를 더 다듬는 방향(F temporal/team profile 등)은 계속 시도해볼 가치가
있음 — 다만 로컬 두 폴드에만 과적합되지 않도록 매번 실제 제출로 재확인 권장.

제출 패키지: `submit_segment_residual_corrector/submit.zip` — champion `.cbm` 그대로 재사용 +
`build_correctors.py`가 오프라인으로 학습한 corrector(segment 3개×seed 3개=9개 ExtraTrees + 고정
카테고리 인코딩 맵)를 `model/correctors.joblib`로 저장, `script.py`가 test.csv에 적용. 로컬 test.csv
재현값 소수점까지 일치 확인, 245,789행 스트레스 테스트 약 2.3초(제한 10분). local 대표 점수(primary)
**801.93**, 실제 LB **879.7995048079**.

### 실험 10 — 능력×상황 조건부 피처 (hand/count/month), 879.80 이후 신규 탐색

corrector importance가 상황 정보(game_month/batter_hand/strikes_before) 위주라는 관찰에서 출발.
투수별 EB-smoothed 조건부 성공률(pitcher×batter_hand, pitcher×count, pitcher×month, leak-safe)을
corrector 입력에 추가.

| 조합 | primary | stress |
|---|---:|---:|
| 없음(현재) | 801.93 | 755.63 |
| hand | 793.06(-8.87) | 790.34(+34.71) |
| count | 796.33(-5.60) | 768.77(+13.15) |
| month | 794.35(-7.58) | 760.17(+4.54) |
| hand+month | 793.37(-8.56) | 796.52(+40.89) |
| hand+count+month | 792.14(-9.79) | 798.14(+42.52) |

전부 같은 방향(primary↓/stress↑)의 트레이드오프. **residual correlation(현재 vs hand+month) =
0.9998~1.0000** — 사실상 같은 예측이라 블렌드로 "둘 다 개선"되는 지점 없음(alpha 스윕으로 확인,
완전히 매끄러운 트레이드오프 곡선). **기각(둘 다 이겨야 채택 원칙 유지), 참고용으로만 기록.**

### 실험 11 — residual bias 스캔 (pitcher-disjoint cross-fit)

현재 champion의 2024 cross-fit residual을 segment/pitcher_hand/batter_hand/month/dayofweek/
base_state/inning 및 쌍별 조합으로 스캔. 표본 충분(n≥300)한 것 중 가장 뚜렷한 건
`pitcher_hand×batter_hand`(L×L -0.0118, L×R +0.0096, R×L +0.0048, R×R -0.0031) — **이미 기각된
hand_matchup과 동일 신호**(로컬은 좋았는데 실제 LB 하락 전례). 나머지 큰 bias(10월 +0.029, 12회
연장 +0.043, 일요일 -0.020)는 전부 n<3000의 소표본 노이즈. **새로운 미탐색 신호 없음 확인.**

### 실험 12 — Trackman 과거 투수 프로필 12개, residual correlation 진단

`modeling/trackman_features.py`(우리 자체 매핑, 신뢰 332명/커버리지 61~66%)의 12개 피처 전부
현재 champion residual과 상관관계 **거의 0**(-0.0038~+0.0027, n=163,649). 세션 초반 baseline
위에서의 Trackman 실패(-31~-43)가 이번 구조에서도 독립 재확인됨. **Trackman historical profile
계열은 완전 종료.**

### 실험 13 — 3-stage 구조(ability/situation 강제 분리) — 기각

Stage1(선수 이력만) → Stage2(상황만으로 Stage1 잔차 설명) → Stage3(기존 segment corrector)로
분리. 전체피처를 한 모델에 다 주는 기존 2-stage보다 두 폴드 다 나쁨(primary 801.93→780.00,
stress 755.63→713.96). CatBoost가 스스로 찾는 능력×상황 교호작용을 인위적 분리가 오히려 방해.

### 실험 14 — recency weighting(season 지수감쇠) λ 9개 정밀 스윕 — 기각

세션 초반 λ=0.5 단일 테스트(694.91, 기각)를 λ=0.05~1.0 9개 값으로 재확인. **모든 λ에서 λ=0(가중치
없음)보다 나쁨** — primary 최선 대안(λ=0.5)도 -30.77, stress는 대부분 λ에서 0.00까지 하락.
"오래된 데이터를 버리지 말 것"이 이번에도 강하게 재확인됨. recency weighting 트랙 완전 종료.

### 종합 (2026-08-13, 실험 10~14 이후)

879.80 champion 이후 시도한 5개 방향(조건부 능력 피처, residual bias 스캔, Trackman, 3-stage 구조,
recency weighting) 중 확실한 개선은 없음. 조건부 능력 피처(hand+month)만 유일하게 "트레이드오프
후보"로 남아있고 나머지 4개는 명확히 기각. `submit_segment_residual_corrector/submit.zip`(879.80)이
계속 유효 champion.

**실험 13 범위 명확화**: 기각된 건 "CatBoost가 스스로 찾는 능력×상황 교호작용을 인위적으로
분해하는 3-stage"(Ability→Situation→Residual)만이다. "서로 다른 모델/정보원을 단계적으로 결합"하는
구조 자체가 기각된 게 아님 — 다만 (a) 여러 tree/boosting 모델 블렌드는 이미 residual 상관관계
0.998+(세션 초반 model_diversity, 이 대화의 corrector 모델 종류 비교 둘 다)로 다양성이 거의 없었고,
(b) Trackman을 새 정보원으로 쓰는 것도 실험 12에서 residual 상관관계 거의 0으로 막힘 — 그래서
"다른 정보원 기반 3-stage" 자체가 유효하려면 먼저 residual diversity가 있는 모델/정보원을 찾아야
한다는 게 현재까지의 결론.

### 실험 15 — 멀티모델(CatBoost+LightGBM+XGBoost) 가중 블렌드 base + 기존 corrector — ✅ 로컬 신기록, 실제 제출 대기

실험 5(model_diversity)에서 CatBoost/LightGBM/XGBoost 단독 residual 상관관계가 0.83~0.96으로
(같은 계열 모델끼리의 0.998+보다) 낮게 나온 걸 근거로, "base 블렌드 자체 + 기존 corrector"를
정확히 조합해서 처음 테스트(`multimodel_base_ensemble.py`, 균등 1/3 가중치):

| | primary | stress |
|---|---:|---:|
| 기존 champion (CatBoost 단독+corrector) | 801.93 | 755.63 |
| 3-model 균등 블렌드+corrector | 797.36(-4.57) | 841.23(+85.61) |

stress는 크게 개선인데 primary만 살짝 손해 — CatBoost 비중을 높인 가중 블렌드로 스윕
(`multimodel_weighted_blend.py`, 7개 가중치):

| weight(cat,lgb,xgb) | primary | stress |
|---|---:|---:|
| (1.0,0,0) 기존 champion | 801.93 | 755.63 |
| (0.8,.1,.1) | **816.00** | 799.16 |
| (0.7,.15,.15) | 814.23 | 821.55 |
| (0.6,.2,.2) ✅ 채택 | 815.15 | 833.05 |
| (0.5,.25,.25) | 812.32 | 834.15 |
| (0.4,.3,.3) | 806.68 | **843.97** |
| (1/3,1/3,1/3) | 797.07 | 841.12 |

**(0.8,.1,.1)부터 (0.4,.3,.3)까지 전 구간이 두 폴드 모두 기존 champion을 이김.** (0.6,.2,.2)를
채택 — primary 최고점(816.00)에 거의 근접하면서 stress도 최고점(843.97)에 가까운 균형점.
LightGBM/XGBoost 하이퍼파라미터·가중치 그리드 전부 자체 설정(외부 재사용 없음). 배포 패키지
`submit_multimodel_blend_corrector/submit.zip` 빌드 완료, 로컬 test.csv sanity 확인
(script.py 출력이 build_artifacts.py의 최종 sanity 값과 정확히 일치). **실제 LB 제출은 아직 —
사용자 결정 대기.**

**2026-08-13 추가 — 첫 제출 시도 InstallError**: `xgboost==3.3.0`이 DACON 설치 환경 미러에 없어서
설치 실패. `xgboost==2.1.4`/`lightgbm==4.5.0`(정착된 구버전)로 다운그레이드 후 그 버전으로 모델을
재학습·재직렬화(joblib pickle은 xgboost/lightgbm 메이저 버전 간 호환 보장 안 됨 — 학습 버전과 설치
버전을 맞춰야 함), submit.zip 재빌드 완료. 245,789행 실제 규모 추론 시간도 실측: **3.7초**(제한
10분). 상세는 `submit_multimodel_blend_corrector/README.md`.

### 종합 (2026-08-12 업데이트: 실험 9.1로 결론 갱신)

신규 실험 9개 중 7개(R-only, F 레짐필터, 계층형 EB 단독, 레벨시프트 calibration, model_diversity
반복확인 성격의 실험 5, F 시즌내 온도차, monotone_constraints)는 primary 기준 기각, 실험 6(baseline+EB
블렌드, +8.93)은 노이즈 바닥선 근처라 보류. **raw 44피처 자체를 건드리는 방향(피처 추가/제거/제약)은
이 시점에서 사실상 소진됐다고 판단** — 세 개의 독립 소스(우리 자체 실험, 다른참가자/aimers,
다른참가자)가 "이 44피처 안에서는 CatBoost+전체데이터가 실질적 상한"이라는 결론에 수렴했다.

**다만 실험 9/9.1(champion + segment residual corrector)로 이 상한을 실제로 넘었다.** 핵심은
피처를 더 추가하는 게 아니라 **base 모델은 그대로 두고 그 오차를 segment별로 별도 학습해서 보정**하는
구조 변경이었다 — "feature 조정으로는 못 번다"는 위 결론과 모순되지 않는다(feature가 아니라 2단계
구조를 바꾼 것). **`submit_segment_residual_corrector/submit.zip`(3-way, 로컬 primary 801.93,
+67.43)이 다음 제출 후보.** `submit/submit.zip`(789.23)은 그대로 안전망으로 유지.
