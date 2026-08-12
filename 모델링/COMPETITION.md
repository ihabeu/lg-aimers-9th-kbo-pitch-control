# 대회 개요

**마지막 업데이트: 2026-08-12**

## 배경

야구에서 투수의 제구력은 실점 억제, 볼카운트 운영, 타자 대응 전략에 직접 영향을 주는 핵심 요소다. 기존엔 평균자책점, 볼넷 수, 스트라이크 비율 같은 경기 후 집계 지표로 제구력을 평가했지만, 실제 경기에서는 투구 직전의 볼카운트·주자 상황·타자/투수 특성·과거 투구 이력 등이 복합적으로 작용한다. 이 대회는 결과 통계가 아니라 **투구가 이루어지기 전까지 확인 가능한 정보만으로 제구 성공 가능성을 예측**하는 문제를 다룬다.

## 대회 단계

- **Phase 1**: 이수 조건(교육과정)
- **Phase 2**(현재 진행 중, 온라인 해커톤): 이 문서가 다루는 대회. 여기서 상위 약 100명이 Phase 3 진출
- **Phase 3**: 1박 2일 오프라인 해커톤, 세부 과제 추후 안내(온라인과 동일하게 야구 데이터 기반 AI 문제로 진행 예정)

## 주제 / 문제 정의

투구 단위의 **제구 성공 확률 예측**. 각 투구 직전까지 확인 가능한 경기 상황·선수 정보·주자 상황·과거 이력을 바탕으로, 그 투구가 제구에 성공할 확률(`control_success`, 0~1)을 예측하는 모델을 만든다.

`control_success`는 공의 실제 위치를 기준으로 정의되며, 아래 3가지는 제구 **실패**, 그 외 유효한 투구는 제구 **성공**이다.

1. 스트라이크존 가운데 부근으로 들어간 공
2. 스트라이크존에서 크게 벗어난 공
3. 포수의 요구 방향과 반대로 들어간 공

**핵심 제약**: 투구 이전 시점에서 확인 가능한 정보만 입력으로 써야 한다. 현재 투구 이후 확정되는 정보(실제 코스/판정/구종/Trackman 실측값 등)는 전부 사용 금지.

## 데이터셋

`data/` 폴더 (원본 설명서: [`data_description.md`](../data/data_description.md))

| 파일 | 크기 | 역할 |
| --- | --- | --- |
| `train.csv` | 1,475,092행 × 49컬럼 | 학습용. 각 행은 고유 투구 1개. `control_success`(정답) 포함 |
| `test.csv` | 배포본 5행(형식 확인용) × 48컬럼 | 평가용. 실제 채점 시 서버가 245,789행 실데이터로 교체. 정답 컬럼 없음 |
| `sample_submission.csv` | 5행 × 2컬럼 | 제출 양식 (`row_id`, `control_success` 확률) |
| `trackman_history.csv` | 1,793,078행 × 30컬럼 | 2019~2024년 Trackman 물리 특성 로그. train/test와 1:1로 안 붙는 별도 ID 체계, 참고용 |

train/test 공통 입력 피처는 5개 그룹으로 구성된다 (컬럼별 상세는 [`eda/COLUMNS.md`](eda/COLUMNS.md)).

- 기본 식별자/경기 정보: `row_id`, `season`, `game_month`, `game_dayofweek`, `inning`, `top_bottom`, `game_type`
- 투구 직전 카운트/점수: `balls_before`, `strikes_before`, `outs_before`, `run_top_before`, `run_bot_before`, `run_total_before`, `score_diff_home`, `score_diff_pitcher_team`
- 주자/상황 중요도: `runner_on_1b/2b/3b`, `num_runners_on`, `base_state`, `home/away_win_expectancy`, `li`
- 선수/팀: `pitcher_id`, `batter_id`, `pitcher_hand`, `batter_hand`, `pitcher_team_id`, `batter_team_id`
- 투구 직전까지의 과거 이력(`asof_*`): 투수/타자의 누적·최근경기·구종별 성공률 등 19개 사전계산 피처

train은 2019~2024시즌, **test는 2025시즌**(학습 데이터에 없는 미래 시즌)이다. `trackman_history.csv`는 train/test와 별도 ID 체계를 쓰고 직접 결합되는 테이블이 아니며, 투수 단위 요약 피처를 만드는 데 참고용으로만 쓸 수 있다.

## 평가 지표

**Brier Skill Score**

```
Score = max(0, 100000 × (1 - Brier / 평균제구율_Brier))
Brier = mean((p_i - y_i)^2)
r = mean(y_i)                     # 전체 평가 데이터의 평균 제구 성공률, 비공개
평균제구율_Brier = r × (1 - r)     # "그냥 평균으로 찍기" 베이스라인
```

- `p_i`: i번째 샘플의 제구 성공 예측 확률
- `y_i`: i번째 샘플의 실제 정답(0 또는 1)
- Public Score: 전체 테스트 데이터 100% 기준. Private Score: 대회 종료 시점의 Public Score와 동일

## 목표 / 평가 방식

- **Phase 2 수료 조건**: Phase1 이수 + Phase2 Public Score **549.51점 이상** (운영진 베이스라인 코드를 운영진 평가 환경에서 실행한 기준 점수)
- 1차 평가: 리더보드 Private Score 100%
- 2차 평가: Phase3 진출 희망 팀은 코드+PPT 제출 후 검증. Private 상위 약 100명이 코드/PPT 검증까지 통과하면 Phase3 진출

## 제출 형식 (코드 제출 대회)

`submit.zip` 구조 (최상위에 불필요한 폴더 없이 정확히 이 구조여야 함):

```
submit.zip
├── model/              # 모델 가중치
├── script.py           # 추론 코드 (제출 시 자동 실행)
└── requirements.txt    # 필요 패키지
```

`script.py`는 `data/test.csv`(평가 서버가 자동 마운트, 읽기전용)를 읽어 `output/submission.csv`(컬럼: `row_id`, `control_success`)를 생성해야 한다.

**제약 조건**

- 전체 추론 시간 ≤ 10분 (245,789개 샘플), 패키지 설치 시간 ≤ 10분
- 제출 파일 ≤ 10GB (압축 해제 후 ≤ 32GB)
- 오프라인 환경 (패키지 설치 외 인터넷 연결 불가)
- 서버 사양: 6 vCPU, 28GB RAM, L4 GPU 22.4GiB VRAM, Python 3.11.15, CUDA 12.8
- 서버에 사전 설치된 주요 패키지(버전 다르게 쓰면 충돌 위험이라 `requirements.txt`에 안 넣는 걸 권장): `torch==2.7.1+cu128`, `pandas==2.0.3`, `numpy==1.26.4`, `scipy==1.15.3`, `scikit-learn==1.8.0`, `transformers==4.46.3` 등
- 오류 종류 2가지: 설치 오류(구조 불일치 등, 일일 제출 횟수 미반영) vs 제출 오류(`script.py` 실행 중 발생하는 모든 오류, 일일 제출 횟수 반영됨)

## 이 프로젝트에서 실제로 확인한 것

- 위 규정 문서에 데이터 경로가 구조도에는 `data/`, 유의사항 텍스트에는 `open/`으로 다르게 적혀 있어(오타로 추정) `script.py`가 두 경로를 다 시도하도록 만들어둠
- 채점 서버 사전 설치 패키지 목록에 `catboost`/`xgboost`/`joblib`(버전 다름) 등은 없어서 `requirements.txt`에 명시 필요
- 실제 진행 상황, 실험 로그, 핵심 인사이트는 [`HANDOFF.md`](HANDOFF.md)와 [`INSIGHTS.md`](INSIGHTS.md) 참고
