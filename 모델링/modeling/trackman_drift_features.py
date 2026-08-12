"""
Trackman T4: 최근(trailing) vs 커리어(cumulative) 물리량 drift.

trackman_mapping_v2로 복원한 pitcher_id <-> pitcher_trackman_id 매핑(396명, cost<=threshold 신뢰
구간)을 이용해, 각 train 행의 (season, game_month) "이전"까지의 Trackman 데이터로
- career_avg_*: 전체 누적 평균
- recent_avg_*: 최근 3개 활동월(populated period) 평균
- *_drift = recent_avg_* - career_avg_*
를 rel_speed(구속)/spin_rate(회전)/induced_vert_break(수직무브먼트)에 대해 만든다.

전부 과거(이전 시점) Trackman 데이터로만 계산 -> DACON Q&A(2026-08-07)가 허용한 "투구 시점 이전까지의
Trackman 통계치를 투수 단위 요약 피처로 사용" 범위 안. 현재 투구 자체의 측정값은 전혀 사용하지 않음.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trackman_mapping_v2 import build_mapping_v2  # noqa: E402
from trackman_features import build_monthly_agg, _period, TM_NUMERIC  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DRIFT_STATS = ["rel_speed", "spin_rate", "induced_vert_break"]
RECENT_WINDOW = 3  # 최근 활동 월 수 (populated period 기준)

DRIFT_FEATURES = [f"{s}_drift" for s in DRIFT_STATS]


def add_drift_features(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv")
    monthly = build_monthly_agg(tm).sort_values(["pitcher_trackman_id", "period"])

    for stat in DRIFT_STATS:
        sum_col, n_col = f"{stat}_sum", f"{stat}_count"
        g = monthly.groupby("pitcher_trackman_id")
        # 둘 다 현재 행(period) 자신은 제외 -> shift(1) 먼저: "그 이전 period까지"만 누적/윈도우 집계
        monthly[f"career_sum_{stat}"] = g[sum_col].transform(lambda s: s.shift(1).cumsum())
        monthly[f"career_n_{stat}"] = g[n_col].transform(lambda s: s.shift(1).cumsum())
        monthly[f"recent_sum_{stat}"] = g[sum_col].transform(lambda s: s.shift(1).rolling(RECENT_WINDOW, min_periods=1).sum())
        monthly[f"recent_n_{stat}"] = g[n_col].transform(lambda s: s.shift(1).rolling(RECENT_WINDOW, min_periods=1).sum())

    feat_cols = ["pitcher_trackman_id", "period"]
    for stat in DRIFT_STATS:
        feat_cols += [f"career_sum_{stat}", f"career_n_{stat}", f"recent_sum_{stat}", f"recent_n_{stat}"]
    feat = monthly[feat_cols].sort_values("period")

    df = df.merge(mapping, on="pitcher_id", how="left")
    df["period"] = _period(df["season"], df["game_month"])
    df = df.sort_values("period")
    feat["pitcher_trackman_id"] = feat["pitcher_trackman_id"].astype("float64")

    merged = pd.merge_asof(df, feat, on="period", by="pitcher_trackman_id", direction="backward", allow_exact_matches=True)

    for stat in DRIFT_STATS:
        career = merged[f"career_sum_{stat}"] / merged[f"career_n_{stat}"]
        recent = merged[f"recent_sum_{stat}"] / merged[f"recent_n_{stat}"]
        merged[f"{stat}_drift"] = recent - career

    return merged.sort_index()


if __name__ == "__main__":
    mapping = build_mapping_v2()
    df = add_drift_features(load("train.csv"), mapping)
    print(df[DRIFT_FEATURES].describe())
    print("\nNaN 비율:")
    print(df[DRIFT_FEATURES].isna().mean())
