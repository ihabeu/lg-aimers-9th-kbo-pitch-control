"""
Trackman 매핑 v2: 전역 1:1 assignment(헝가리안) + unmatched 옵션.

v1(trackman_mapping.py, greedy 최근접 + 사후 dedup)과의 차이 및 시행착오:
- fingerprint에 구종비율(fastball/breaking/offspeed rate)을 추가해봤으나, 월별-only 매칭과 구종비율-only
  매칭이 겨우 2.3%만 일치(무작위 기준선 대비는 높지만 절대적으로 낮음) — 구종비율은 3개 숫자뿐이라
  906명 중 한 명을 특정하기엔 차원이 부족해 노이즈에 가깝다. 그래서 fingerprint는 v1과 동일하게
  월별(season, game_month) 투구수 히스토그램만 사용.
- hand를 매칭 단계의 하드 제약으로 쓰면 hand 일치율이 정의상 100%가 되어 독립적인 사후 검증 수단을
  잃는다. 그래서 hand는 매칭에 전혀 안 쓰고 순수 월별 신호로만 전역 1:1(헝가리안)을 수행한 뒤,
  hand 일치율로 사후 검증한다 (v1과 동일한 원칙 유지, 매칭 알고리즘만 greedy->헝가리안으로 교체).
- 결과: 전체 hand 일치율 84.97%(v1 81.1%보다 개선), many-to-one 충돌은 구조적으로 0건
  (헝가리안은 전역 최적 1:1이라 v1처럼 여러 명이 같은 trackman_id를 두고 경쟁할 일이 없음).
- cost(매칭 거리) 상위 50% 구간의 hand 일치율이 93.7%로 가장 높아 이 지점을 신뢰 임계값으로 채택.
  그 밖(cost가 threshold보다 큰) 투수는 "매칭 안 함(unmatched)"으로 제외 — 커버리지를 늘리는 것보다
  정확한 매칭만 확보하는 쪽을 우선.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

TRAIN_HAND_MAP = {2: "Right", 1: "Left"}
COST_THRESHOLD = 0.1462  # cost 상위 50% 구간 상한 (hand 일치율 93.7%로 가장 높았던 지점)


def build_month_fingerprints(train: pd.DataFrame, tm: pd.DataFrame):
    train_fp = train.groupby(["pitcher_id", "season", "game_month"]).size().unstack(["season", "game_month"], fill_value=0)
    tm_fp = tm.groupby(["pitcher_trackman_id", "season", "game_month"]).size().unstack(["season", "game_month"], fill_value=0)
    all_cols = sorted(set(train_fp.columns) | set(tm_fp.columns))
    train_fp = train_fp.reindex(columns=all_cols, fill_value=0)
    tm_fp = tm_fp.reindex(columns=all_cols, fill_value=0)
    # 총 투구수가 선수마다 크게 달라서(수십~수천) row sum으로 정규화 안 하면 활동량 차이가
    # 분포 모양 차이보다 거리를 지배해버린다.
    train_fp = train_fp.div(train_fp.sum(axis=1).replace(0, 1), axis=0)
    tm_fp = tm_fp.div(tm_fp.sum(axis=1).replace(0, 1), axis=0)
    return train_fp, tm_fp


def build_mapping_v2(cost_threshold: float = COST_THRESHOLD) -> pd.DataFrame:
    train = load("train.csv")[["pitcher_id", "season", "game_month", "pitcher_hand"]]
    tm = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "data" / "trackman_history.csv",
                      usecols=["pitcher_trackman_id", "season", "game_month", "pitcher_hand"])

    train_fp, tm_fp = build_month_fingerprints(train, tm)
    train_ids = train_fp.index.to_numpy()
    tm_ids = tm_fp.index.to_numpy()

    D = cdist(train_fp.to_numpy(), tm_fp.to_numpy(), metric="euclidean")
    row_idx, col_idx = linear_sum_assignment(D)  # 전역 최적 1:1, hand 제약 없음(사후 검증용으로 남겨둠)
    cost = D[row_idx, col_idx]

    mapping = pd.DataFrame({
        "pitcher_id": train_ids[row_idx],
        "pitcher_trackman_id": tm_ids[col_idx],
        "cost": cost,
    })
    confident = mapping[mapping["cost"] <= cost_threshold].reset_index(drop=True)
    print(f"헝가리안 1:1 매칭: 전체 {len(mapping)}명 중 cost<={cost_threshold} 신뢰 매핑 {len(confident)}명 "
          f"({len(confident) / len(mapping):.1%}), many-to-one 충돌 0건(구조적)")
    return confident


if __name__ == "__main__":
    m = build_mapping_v2()
    print(m.head(10))
