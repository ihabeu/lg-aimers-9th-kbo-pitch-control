# EDA 설명 (train.csv 기준)

**마지막 업데이트: 2026-08-09**

함수·스키마 정의는 [eda.py](eda.py), 실행 결과와 설명은 [eda.ipynb](eda.ipynb) (eda.py를 import해서 셀별로 돎, 이미 실행된 결과 그대로 남아있음). 컬럼별 상세는 [COLUMNS.md](COLUMNS.md), 원본 컬럼 정의는 [../../data/data_description.md](../../data/data_description.md), 데이터셋 전체 개요는 [../README.md](../README.md) 참고.

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

트리 모델은 다중공선성에 크게 영향받지 않아 그대로 둬도 되지만, 상호배타적 분해 그룹은 나중에 계층적 분해 모델링(7절 이전 실험의 E16-H1 방식) 후보로 남겨둔다.

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

## 부록 A. 도메인 규칙 검증

- 콜드스타트: `asof_pitcher_n==0`인 792건에서 관련 rate 컬럼이 100% 결측 (설명서 문구와 일치)
- `base_state` 문자열이 `runner_on_1b/2b/3b`와 100% 일치 (불일치 0건) — 즉 `base_state`는 저 세 이진 컬럼의 압축 표현일 뿐, 정보량은 중복

## 부록 B. trackman_history.csv

train/test와 1:1로 안 붙는 별도 참고 데이터. 지금 베이스라인에서는 안 쓰기로 결정함 (ID 매핑 비용 대비 검증된 효과 없음 — 다른 팀이 이미 시도해서 확인).

## 부록 C. 다음 단계: 전처리 방향 (제안, 아직 미실행)

- **수치형 스케일링**: LightGBM 같은 트리 모델이면 불필요 (분할이 단조변환에 불변). 선형/신경망 계열을 섞는다면 StandardScaler 추천 (`li`, `score_diff_*`에 극단값이 있어 MinMax보다 안전)
- **범주형 인코딩**: 3개뿐, 최대 13개 값이라 원핫도 부담 없지만 LightGBM 쓰면 네이티브 카테고리 지정이 더 간단
- **시계열형**: `season`은 수치 그대로, `game_month`/`game_dayofweek`는 순환형이라 범주형(원핫/네이티브)으로 — 트리 모델이면 sin/cos 인코딩보다 이걸로 충분. `inning`은 수치 유지 + `is_extra_inning`(10회 이상) 플래그 추가를 피처 엔지니어링 후보로 제안

### 구간 기반(threshold) 파생 피처

모든 수치형이 매끄러운 선형 관계는 아니다. `asof_pitcher_success_rate`처럼 구간을 나눠봐도 쭉 단조증가하는 변수는 raw 값 그대로가 최선이지만(3절에서 검증함), 특정 값을 기준으로 상황 자체가 바뀌는(regime change) 변수는 raw 값 대신/추가로 구간 플래그를 만드는 게 나을 수 있다. 트리 모델은 이론상 이런 임계값을 스스로 찾아낼 수 있지만, 도메인 지식으로 미리 임계값을 지정해주면 적은 데이터로도 더 안정적으로 그 분할을 학습한다. 실제로 이전 실험의 실험에서 **`season≥2023 × game_type=F` regime 플래그가 전체 실험 중 단일 개선폭이 가장 컸다** — 이게 정확히 이 방식의 성공 사례.

후보:

| 피처 | 임계값 | 근거 |
| --- | --- | --- |
| `is_extra_inning` | `inning >= 10` | 연장전은 마무리 투수 소진, 평소 안 던지는 투수 등판 등으로 상황 자체가 다름 |
| `season_regime` | `season >= 2023` | 이전 실험이 검증한 가장 큰 단일 개선 요인 (2023년부터 성공률 추세가 꺾임) |
| `is_full_count` | `balls_before==3 & strikes_before==2` | 풀카운트는 투수가 반드시 승부를 봐야 하는 특수 상황 |
| `pitcher_experience_tier` | `asof_pitcher_n` 구간(예: <50 / 50~500 / 500+) | 이전 실험이 cold-start 구간(n≤50)에서 EB smoothing으로 따로 처리해 효과를 봄 |

원본 수치형(`inning`, `season`, `asof_pitcher_n` 등)은 그대로 두고 이 플래그들을 **추가**하는 방향 — raw 값의 세밀한 정보를 잃지 않으면서 regime 경계를 명시적으로 알려주는 것.
- **결측치**: LightGBM이면 NaN 그대로 전달이 최선 (모델이 분기 방향을 데이터로부터 학습, 사람이 정한 고정 규칙보다 나음). 결측 안 받는 모델과 비교할 거면 중앙값/KNN 대신 Empirical Bayes smoothing 추천 (지난번 이전 실험이 검증한 방법). KNN은 구조적 결측(이력 자체가 없음)이라 안 맞고, 행 삭제는 test에서 자주 나올 유형(첫 등판)을 학습에서 빼는 셈이라 제일 위험

다음 결정할 것: 최종 모델을 LightGBM 단일로 갈지 여러 모델 블렌드로 갈지 — 이게 정해지면 `preprocess.py`로 실제 전처리 스크립트를 만들고 `data/processed/`에 결과를 저장하는 단계로 넘어간다.

## 7. 참고: 다른 팀 실험 로그와 비교

다른 참가자 공개 레포 팀의 `model_test.md`를 보면 상황변수만 쓴 모델(E1)이 AUC 0.498로 사실상 랜덤이었고, 투수 이력을 넣은 순간(E2) 개선이 시작됐다고 함 — 여기서 나온 변수 중요도 결과와 같은 이야기다. 그 팀은 추가로 `season>=2023 & game_type=F` 교호작용(regime shift)에서 가장 큰 단일 개선을 얻었는데, 우리 EDA에서도 `game_type=F`가 성공률 0.603으로 튀고 중요도 상위권인 걸 보면 같은 신호를 잡고 있는 것으로 보인다.
