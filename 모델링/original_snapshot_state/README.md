# Snapshot State Ensemble

V14의 최근 레짐 관점은 참고하되 코드를 재현하지 않은 독립 실험이다.

- 각 행의 `asof_*` 누적값과 직전 시즌 마지막 스냅샷의 차이로 해당 시즌 상태를 복원한다.
- 장기 이력, 당해 시즌 EB-smoothed 상태, 직전 1/3/5경기 상태의 차이를 피처로 쓴다.
- CatBoost 없이 LightGBM과 HistGradientBoosting 두 모델을 블렌딩한다.
- 2023 학습 → 2024 검증만으로 평가한다. test 행 간 집계나 rolling은 사용하지 않는다.

실행:

```bash
cd ".../모델링/original_snapshot_state"
python experiment.py
```
