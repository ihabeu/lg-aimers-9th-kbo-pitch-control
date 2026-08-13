# EDA 설명 (train.csv 기준)

**마지막 업데이트: 2026-08-09**

함수·스키마 정의는 [eda.py](eda.py), 실행 결과와 설명은 [eda.ipynb](eda.ipynb) (eda.py를 import해서 셀별로 돎, 이미 실행된 결과 그대로 남아있음). 심층 분석(ablation z-검정, 분산 상한, 교호작용 permutation 검정) 코드는 [deep_dive.py](deep_dive.py). 컬럼별 상세는 [COLUMNS.md](COLUMNS.md), 원본 컬럼 정의는 [../../data/data_description.md](../../data/data_description.md), 데이터셋 전체 개요는 [../인사이트/README.md](../인사이트/README.md) 참고.

전체 실행 시간은 로컬에서 약 17초.

## 1. 컬럼 분류 기준

dtype 자동분류 대신 도메인 지식으로 5갈래로 나눴다.

| 분류 | 개수 | 컬럼 |
| --- | --- | --- |
| 식별자 | 3 | `row_id`, `pitcher_id`, `batter_id` |
| 시계열형 | 4 | `season`, `game_month`, `game_dayofweek`, `inning` |
| 수치형 | 30 | 볼카운트, 점수차, 기대승률, `li`, `asof_*` 이력 피처 |
| 이진형 | 7 | `top_bottom`, `game_type`, `pitcher_hand`, `batter_hand`, `runner_on_1b/2b/3b` |
| 범주형 (명목형) | 3 | `base_state`, `pitcher_team_id`, `batter_team_id` |
| 타겟 | 1 | `control_success` |

합계 3+4+30+7+3+1=48 (원본 49컬럼 - 완전 중복 1개 제거).

**시계열형을 수치형에서 따로 뺀 이유**: `season`은 순환 없이 계속 증가해서 그대로 수치로 써도 되지만, `game_month`/`game_dayofweek`는 순환한다(12월 다음이 1월). 전처리 방식이 서로 다를 수 있어서 하나로 묶었다.

**`asof_pitcher_pitchmix_n`은 스키마에서 제외했다.** `asof_pitcher_n`과 값이 100% 완전히 동일한 중복 컬럼이라는 걸 검증했다 (4절 참고).

## 2. 결측치

| 컬럼군 | 결측 비율 | 원인 |
| --- | --- | --- |
| `asof_pitcher_prev1/3/5_game_*` | 1.98% (29,185건) | 직전 N경기 등판 이력 자체가 없음 |
| `asof_pitcher_*`(누적), `asof_batter_*` | 0.05~0.06% (792/830건) | `asof_pitcher_n==0`인 완전 첫 투구 |

두 그룹은 원인이 다르니 같은 방식으로 채우면 안 된다. 처리 방향은 부록 C 참고.

## 3. 타겟 분포 및 상황변수 대비 성공률

전 컬럼을 다 보지 않고 변수 중요도가 높거나(5절) 도메인상 의심스러운 변수 위주로 봤다.

**연속형 상위 3개(`asof_pitcher_success_rate`, `asof_pitcher_reverse_rate`, `asof_batter_success_rate`)를 10구간으로 나눠보면 셋 다 단조증가/감소한다.** 특히 `asof_pitcher_success_rate`는 최하위 구간 0.457 → 최상위 구간 0.612로 거의 선형 — 변수가 타겟과 잘 정렬돼 있다는 뜻.

| 변수 | 값/구간 | 성공률 |
| --- | --- | --- |
| `season` | 2019 → 2024 | 0.565 → 0.486 (계속 하락, drift) |
| `game_type` | R vs F | 0.514 vs 0.603 |
| `pitcher_team_id` | 22, 23번 | 0.692, 0.610 — 단, 표본 676/4,437건뿐(다른 팀 13만~21만), 소표본 노이즈 가능성 큼 |

## 4. 변수 간 상관관계

수치형+시계열형+이진형(0/1 인코딩)만 대상 (범주형은 순서 없어 피어슨 상관 자체가 의미 없어 제외).

**완전 중복/파생 (실제 값으로 검증 완료)**

| 관계 | 검증 결과 |
| --- | --- |
| `asof_pitcher_n` = `asof_pitcher_pitchmix_n` | 100% 동일 → 스키마에서 후자 제거 |
| `run_total_before` = `run_top_before` + `run_bot_before` | 완전한 합 |
| `num_runners_on` = `runner_on_1b`+`runner_on_2b`+`runner_on_3b` | 완전한 합 |
| `home_win_expectancy` + `away_win_expectancy` ≈ 100 | 반올림 오차만 (평균 100.0001, 표준편차 0.02) |

**구조적으로 묶이는 그룹 (상관 0.5~0.9대, 완전 중복은 아님)**

- `asof_pitcher_success_rate`/`reverse_rate`/`ball_rate`/`strike_rate` (상관 -0.7~-0.8) — 투구 결과를 상호배타적으로 나눈 비율로 추정
- `asof_pitcher_prev1/3/5_game_*`끼리 (0.57~0.88) — 겹치는 rolling window라 당연함
- `asof_pitcher_breaking_rate`/`offspeed_rate` (-0.58) — 구종 비율 제로섬 구조로 추정

트리 모델은 다중공선성에 크게 영향받지 않아 그대로 둬도 되지만, 상호배타적 분해 그룹은 나중에 계층적 분해 모델링 후보로 남겨둔다.

## 5. 변수 중요도 (RandomForest, 20만행 서브샘플)

| 순위 | 컬럼 | importance |
| --- | --- | --- |
| 1 | `asof_pitcher_success_rate` | 0.12 내외 |
| 2 | `asof_pitcher_reverse_rate` | |
| 3 | `asof_pitcher_prev5_game_success_rate` | |
| 4 | `asof_pitcher_prev3_game_success_rate` | |
| 5 | `asof_batter_success_rate` | |
| 6 | `game_type` (이진형) | |
| 7 | `season` (시계열형) | |

`asof_*` 투수/타자 이력이 상위를 싹 쓸었고 순간 상황변수(볼카운트, 주자 등)는 하위권. 정확한 수치는 [eda.ipynb](eda.ipynb) 5절 또는 `eda_outputs/train_feature_importance.csv` 참고 (RandomForest 서브샘플 기반이라 재실행 시마다 소수점 단위로 조금씩 바뀔 수 있음, `random_state=42`라 큰 순위는 안정적).

## 6. 심층 분석 1 — Ablation z-검정: 어떤 피처 그룹이 진짜 필요한가

5절의 RandomForest 변수중요도는 "단독으로 얼마나 강한가"만 본다. 컬럼끼리 서로 상관돼 있으면(4절) "혼자 얼마나 강한가"와 "이미 다른 피처를 다 가진 모델에 이 그룹을 더/덜 넣으면 얼마나 달라지는가"는 다른 질문이다. 후자를 재려고 실제 CatBoost 모델로 그룹을 하나씩 빼고 primary 폴드(2019-23→24)에서 다시 학습해 손해를 쟀다. 유의성은 검증셋을 투수 단위로 리샘플링하는 bootstrap(500회)으로 z-score를 계산했다 — 투수 단위로 묶은 이유는 같은 투수의 여러 투구가 서로 독립이 아니기 때문(pitcher-disjoint cross-fit과 같은 원칙).

코드: [deep_dive.py](deep_dive.py) `run_ablation()`. 결과: [eda_outputs/deep_dive_ablation.csv](eda_outputs/deep_dive_ablation.csv).

| 뺀 그룹 | 제거 후 score | 손해 | z |
| --- | ---: | ---: | ---: |
| 전체 44피처 (기준) | 734.49 | — | — |
| 시간(season/month/dayofweek/inning/top_bottom) | 208.39 | **+526.11** | **9.51** |
| 투수/타자 손, 팀 ID | 571.11 | +163.39 | 9.10 |
| 투수 통산 성공/반대/가운데(career core) | 597.06 | +137.43 | 5.45 |
| game_type | 634.03 | +100.46 | 6.11 |
| 투수 최근 폼(1/3/5경기) | 658.67 | +75.82 | 5.31 |
| 카운트/아웃/주자(순간 상황) | 685.12 | +49.37 | 5.19 |
| 투수 구종 성향(fastball/breaking/offspeed) | 681.13 | +53.37 | 6.82 |
| 표본수(asof_pitcher_n) | 696.81 | +37.68 | 4.81 |
| 타자 이력 | 713.16 | +21.34 | 2.13 |
| 점수/승부 중요도(score+leverage) | 722.65 | +11.85 | 1.81 |

**시간 피처(`season` 포함)를 통째로 빼면 압도적으로 가장 크게 무너진다(z=9.51, score 734→208).** 2절의 시즌 drift(-7.86%p)가 이 문제의 가장 큰 단일 신호원이라는 뜻 — `season` 하나가 아니라 `season×asof_*` 교호작용까지 CatBoost가 이 그룹 전체에 의존하고 있어서, 빼면 그 교호작용 전부가 같이 사라진다.

**두 번째로 큰 손해는 손/팀 ID(z=9.10)다.** 5절 RandomForest 기준으로는 팀 ID가 하위권이었는데 CatBoost ablation에서는 2위로 올라온다 — RandomForest는 범주형을 임의 정수로 인코딩해서 팀 정보를 거의 못 쓰지만, CatBoost의 ordered target statistics 인코딩은 팀별 성공률 차이를 효율적으로 활용한다는 뜻이다. **인코딩 방식이 바뀌면 같은 컬럼의 "중요도"가 완전히 달라질 수 있다** — 5절 RandomForest 결과만 보고 팀 ID를 버리면 안 된다는 근거.

**`game_type` 단독도 상당히 크다(z=6.11).** 3절에서 F/R 반전을 봤을 때 "CatBoost가 season과 조합해서 이미 흡수하고 있다"고 결론냈는데, 그 조합에 참여하는 `game_type` 자체를 완전히 빼면 여전히 큰 손해라는 것도 같이 확인된다 — season과 game_type은 서로 대체 관계가 아니라 상호보완 관계다.

**점수/승부 중요도(score+leverage) 그룹은 유일하게 z<2로 통계적으로 약하다.** 완전히 무신호는 아니지만(z=1.81), 이 그룹만 따로 빼는 피처 엔지니어링은 기대값이 낮다는 뜻 — 이미 여러 실험(EXPERIMENTS.md)에서 순간 상황 피처 추가/제거 시도가 전부 기각된 것과 일관된다.

## 7. 심층 분석 2 — 분산 상한: 이론적으로 얼마나 더 좋아질 수 있는가

"완벽한 모델의 Brier"는 0이 아니라 `r(1-r) - Var(그룹별 진짜 성공률)`이다 — 아무리 좋은 모델도 같은 그룹(예: 같은 투수) 안에서의 무작위성(그날 컨디션, 공 하나하나의 우연)까지는 못 없앤다. 그룹을 세밀하게 쪼갤수록(더 많은 정보를 안다고 가정할수록) 이 상한은 높아진다. 그룹별 "진짜" 분산은 표본이 작을수록 관측 분산이 과대추정되므로(표본 1개면 성공률이 무조건 0 또는 1), one-way random-effects ANOVA의 method-of-moments 추정량으로 표본 노이즈를 제거해서 계산했다(`sigma2_between = max(0, (MSB-MSW)/n0)`).

코드: [deep_dive.py](deep_dive.py) `run_variance_ceiling()`. 결과: [eda_outputs/deep_dive_variance_ceiling.csv](eda_outputs/deep_dive_variance_ceiling.csv). 1군(R) 행 1,314,088개, 전체 분산 r(1-r)=0.249803 기준.

| 그룹 기준 | 그룹 수 | 진짜 분산(추정) | 상한 Brier | 상한 score |
| --- | ---: | ---: | ---: | ---: |
| 투수(통산) | 612 | 0.001998 | 0.247805 | **799.98** |
| 투수 × 타자손 매치업 | 1,216 | 0.002346 | 0.247458 | **939.05** |
| 투수 × 타자 | 78,043 | 0.004778 | 0.245025 | **1912.64** |

**"투수가 누구인지만 알 때"의 이론적 상한은 score 800 정도다.** 현재 champion(실제 LB 879.80)이 이미 이 상한을 넘는다 — 상황 정보(카운트, 주자, 시즌 등)가 "같은 투수 안에서도 얼마나 잘 던질지"를 추가로 설명해주기 때문에, 순수 투수-그룹 상한보다 높은 점수가 나오는 게 자연스럽다(그룹 상한은 "투수 정체성만" 안다고 가정했을 때의 한계지, 상황 정보까지 아는 모델의 한계가 아니다).

**투수×타자손(좌우 매치업) 기준 상한은 939.** 손 매치업까지 세밀하게 안다면 이론적으로 940점 선까지 여지가 있다는 뜻이지만, `hand_matchup`을 실제로 피처화한 시도(EXPERIMENTS.md)는 로컬에서는 통했다가 실제 리더보드에서 뒤집힌 전례가 있다 — **상한이 있다는 것과 "그 상한까지 실제로 안전하게 도달할 수 있다"는 것은 다른 문제**라는 걸 보여주는 좋은 사례다.

**투수×타자(개별 매치업) 기준 상한은 무려 1913.** 이게 이론적으로는 1300점 목표를 훨씬 넘어서는 여지지만, 그룹이 78,043개나 되고 그룹당 평균 표본이 매우 작아서(1,314,088/78,043 ≈ 17행) 실제로 이 상한에 접근하려면 개별 매치업 조합을 안정적으로 추정할 방법이 필요하다 — 지금까지 시도한 pitcher×batter 관련 피처(hand_matchup, matchup 3종 등)가 전부 소표본 과적합/로컬-실제 괴리로 막혔던 이유가 바로 이 "그룹은 세밀한데 표본은 얕다"는 구조 때문으로 해석할 수 있다. **1300점 목표에 실제로 다가가려면 이 매치업 축을 얼마나 안전하게(과적합 없이) 활용하느냐가 관건**이라는 정량적 근거.

## 8. 심층 분석 3 — 교호작용 permutation 유의성 검정

가법 모형(각 변수의 주효과 합)으로 설명 안 되는 "진짜 교호작용"이 있는지, 우연(null)보다 유의하게 큰지를 permutation(무작위 재배열 200회)으로 검정했다. 두 변수를 각각 몇 구간으로 나누고, 각 셀의 실제 평균과 "행 평균+열 평균-전체평균"(가법 예측)의 차이를 제곱해 표본크기로 가중합한 것을 교호작용 강도로 쓴다.

코드: [deep_dive.py](deep_dive.py) `run_interaction_tests()`. 결과: [eda_outputs/deep_dive_interactions.csv](eda_outputs/deep_dive_interactions.csv).

| 교호작용 | 관측 강도 | null 평균±sd | p-value |
| --- | ---: | ---: | ---: |
| 투수손 × 타자손 | 0.000124 | 0.000000±0.000000 | **<0.0001** |
| season × 투수 통산성공률(5분위) | 0.000122 | 0.000004±0.000001 | **<0.0001** |
| game_type × season | 0.000000 | 0.000000±0.000000 | 0.3550 |

**투수손×타자손, season×투수성공률 둘 다 강하게 유의하다(p<0.0001).** 전자는 7절의 "상이 코드가 동일 코드보다 매 시즌 높다"는 관찰, 후자는 3절의 시즌 drift가 `asof_*` rate의 해석 자체를 바꾼다는 관찰과 각각 독립적인 방법으로 재확인된다.

**game_type×season은 이 검정에서는 유의하지 않았다(p=0.355) — 3절의 직접 관찰(F가 2019~2022엔 R보다 높다가 2023부터 낮아짐)과 겉보기에 모순된다.** 원인으로 추정되는 것: 이 검정은 6개 시즌 × 2개 game_type = 12개 셀 전체에 걸친 "평균적인" 비가법성을 재는데, F는 전체의 10.9%뿐이라 셀별 표본이 작고, 반전이 2023년 전후로만 날카롭게 나타나는 데 비해 나머지 시즌들은 완만해서 전체 평균 강도가 희석된다. **이 검정의 null 결과가 "F/R 반전이 없다"는 뜻이 아니라, "이 특정 coarse한 검정 방식이 좁고 날카로운 regime shift를 탐지하기엔 둔감하다"는 뜻으로 해석해야 한다** — 3절의 직접 관찰(연도별 F/R 성공률 표)이 더 신뢰할 만한 증거다.
## 부록 A. 도메인 규칙 검증

- 콜드스타트: `asof_pitcher_n==0`인 792건에서 관련 rate 컬럼이 100% 결측 (설명서 문구와 일치)
- `base_state` 문자열이 `runner_on_1b/2b/3b`와 100% 일치 (불일치 0건) — 즉 `base_state`는 저 세 이진 컬럼의 압축 표현일 뿐, 정보량은 중복

## 부록 B. trackman_history.csv

train/test와 1:1로 안 붙는 별도 참고 데이터. 지금 베이스라인에서는 안 쓰기로 결정함 (ID 매핑 비용 대비 검증된 효과 없음).

## 부록 C. 다음 단계: 전처리 방향 (제안, 아직 미실행)

- **수치형 스케일링**: LightGBM 같은 트리 모델이면 불필요 (분할이 단조변환에 불변). 선형/신경망 계열을 섞는다면 StandardScaler 추천 (`li`, `score_diff_*`에 극단값이 있어 MinMax보다 안전)
- **범주형 인코딩**: 3개뿐, 최대 13개 값이라 원핫도 부담 없지만 LightGBM 쓰면 네이티브 카테고리 지정이 더 간단
- **시계열형**: `season`은 수치 그대로, `game_month`/`game_dayofweek`는 순환형이라 범주형(원핫/네이티브)으로 — 트리 모델이면 sin/cos 인코딩보다 이걸로 충분. `inning`은 수치 유지 + `is_extra_inning`(10회 이상) 플래그 추가를 피처 엔지니어링 후보로 제안

### 구간 기반(threshold) 파생 피처

모든 수치형이 매끄러운 선형 관계는 아니다. `asof_pitcher_success_rate`처럼 구간을 나눠봐도 쭉 단조증가하는 변수는 raw 값 그대로가 최선이지만(3절에서 검증함), 특정 값을 기준으로 상황 자체가 바뀌는(regime change) 변수는 raw 값 대신/추가로 구간 플래그를 만드는 게 나을 수 있다. 트리 모델은 이론상 이런 임계값을 스스로 찾아낼 수 있지만, 도메인 지식으로 미리 임계값을 지정해주면 적은 데이터로도 더 안정적으로 그 분할을 학습한다. `season≥2023 × game_type=F` regime 플래그가 이후 실험에서 실제로 유효했다 — 이게 정확히 이 방식의 성공 사례.

후보:

| 피처 | 임계값 | 근거 |
| --- | --- | --- |
| `is_extra_inning` | `inning >= 10` | 연장전은 마무리 투수 소진, 평소 안 던지는 투수 등판 등으로 상황 자체가 다름 |
| `season_regime` | `season >= 2023` | 2023년부터 성공률 추세가 꺾이는 걸 EDA로 확인한 구간 |
| `is_full_count` | `balls_before==3 & strikes_before==2` | 풀카운트는 투수가 반드시 승부를 봐야 하는 특수 상황 |
| `pitcher_experience_tier` | `asof_pitcher_n` 구간(예: <50 / 50~500 / 500+) | cold-start 구간(n≤50)은 EB smoothing으로 따로 처리 |

원본 수치형(`inning`, `season`, `asof_pitcher_n` 등)은 그대로 두고 이 플래그들을 **추가**하는 방향 — raw 값의 세밀한 정보를 잃지 않으면서 regime 경계를 명시적으로 알려주는 것.
- **결측치**: LightGBM이면 NaN 그대로 전달이 최선 (모델이 분기 방향을 데이터로부터 학습, 사람이 정한 고정 규칙보다 나음). 결측 안 받는 모델과 비교할 거면 중앙값/KNN 대신 Empirical Bayes smoothing 추천. KNN은 구조적 결측(이력 자체가 없음)이라 안 맞고, 행 삭제는 test에서 자주 나올 유형(첫 등판)을 학습에서 빼는 셈이라 제일 위험

다음 결정할 것: 최종 모델을 LightGBM 단일로 갈지 여러 모델 블렌드로 갈지 — 이게 정해지면 `preprocess.py`로 실제 전처리 스크립트를 만들고 `data/processed/`에 결과를 저장하는 단계로 넘어간다.

