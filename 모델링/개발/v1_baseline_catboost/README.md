# v1 — CatBoost baseline

**작성일: 2026-08-12** · **실제 리더보드: 789.23** (현재까지 최종 제출, 수료 기준 549.51 통과)

## 요약

원본 44개 입력 피처를 그대로 CatBoost에 넣은 모델. 전처리를 거의 안 하는 게 핵심 — CatBoost의 native categorical/결측 처리를 그대로 활용한다. 자세한 이유는 [`../preprocessing/README.md`](../preprocessing/README.md) 참고.

## 사용한 피처 (44개)

`train_baseline_catboost.py`의 `FEATURES`/`CAT_FEATURES` 참고. 5개 그룹:

- **수치형(30개)**: 카운트/주자/점수 상황 + `asof_*` 이력 피처 19개(투수/타자 누적·최근경기·구종별 성공률)
- **시계열-수치형(2개)**: `season`, `inning` (순서가 실제로 의미 있어서 수치형 유지)
- **이진형(7개)**: `top_bottom`, `game_type`, `pitcher_hand`, `batter_hand`, `runner_on_{1,2,3}b`
- **범주형(3개)**: `base_state`, `pitcher_team_id`, `batter_team_id`
- **시계열-범주형(2개)**: `game_month`, `game_dayofweek` (순환성 때문에 범주형 처리)

범주형 12개는 전부 CatBoost native categorical로 전달(원-핫/타깃인코딩 없음).

## 하이퍼파라미터

```
depth=6, learning_rate=0.05, l2_leaf_reg=15.0 (기본값 3.0 대비 스윕해서 채택)
loss_function=Logloss, eval_metric=BrierScore(모니터링용)
```

`l2_leaf_reg`가 유일하게 튜닝한 값. CatBoost는 리프값에 L2만 지원(L1/L0 옵션 자체가 없음).

## 학습 절차 (leak-safe, "2024 제외 버그" 재발 방지)

1. `train.csv`를 season<2024(학습) / season==2024(검증)로 나눠 early stopping으로 최적 iteration 수(204)를 찾음.
2. 그 iteration 수를 고정하고 `train.csv` **전체(2019~2024)**로 재학습 — 이게 실제 제출 모델.
3. 검증에 쓴 연도를 최종 모델 학습에서 빼먹는 실수를 이 프로젝트에서 두 번(CatBoost 1차, NN) 저질렀던 적이 있어서, 매 모델마다 반드시 재확인.

## 검증

Rolling out-of-time 3폴드(2019-21→22, 2019-22→23, 2019-23→24, 가중치 0.2/0.3/0.5)와 단일 2024 검증 둘 다 확인. 자세한 실험 로그는 [`HANDOFF.md`](../../인사이트/HANDOFF.md) 참고. 44개 이상의 개선 시도(raw ID, matchup, Trackman, model diversity, calibration, recent_k_pitch_rate 등)가 전부 이 baseline을 못 넘었음 — 그 실패 기록이 오히려 이 baseline이 얼마나 강한 로컬 최적점인지 보여줌.

## 파일 구성

```
v1_baseline_catboost/
  train_baseline_catboost.py   # 학습 코드 (원본: modeling/baseline_catboost.py)
  model/catboost_baseline.cbm  # 최종 학습된 모델 (전체 데이터)
  script.py                    # 제출용 추론 스크립트
  requirements.txt
```
