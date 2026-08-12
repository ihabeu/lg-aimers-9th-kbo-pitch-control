"""
"심리적/압박 상황 반응" 피처 — trackman_history 자체의 상황 컬럼(balls_before, strikes_before)을
이용해, 이 투수가 과거에 압박 상황(3볼 또는 2스트라이크)에서 던질 때 구속/spin이 평소(0-0 카운트)와
얼마나 달랐는지를 leak-safe 과거 요약으로 만든다. 현재 투구의 카운트가 아니라 "과거에 압박 상황에서
어땠는가"의 요약이라 현재 투구 정보를 쓰는 게 아님 — DACON Q&A가 허용한 범위(투구 이전까지의 투수
단위 요약 피처) 안에 있다.

- pitcher_pressure_velocity_drop = (0-0 카운트 평균 구속) - (3볼 또는 2스트라이크 카운트 평균 구속)
  양수면 압박 상황에서 구속이 떨어지는 투수, 음수면 오히려 더 세게 던지는 투수.
- pitcher_pressure_spin_drop: 동일 로직을 spin_rate에.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from trackman_mapping_v3 import get_confident_mapping  # noqa: E402
from trackman_features import _period, DATA_DIR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

PRESSURE_FEATURES = ["pitcher_pressure_velocity_drop", "pitcher_pressure_spin_drop"]


def _situational_monthly(tm: pd.DataFrame, situation_mask: pd.Series, label: str) -> pd.DataFrame:
    sub = tm[situation_mask]
    g = sub.groupby(["pitcher_trackman_id", "season", "game_month"])
    agg = g.agg(n=("rel_speed", "size"), velo_sum=("rel_speed", "sum"), spin_sum=("spin_rate", "sum")).reset_index()
    agg["period"] = _period(agg["season"], agg["game_month"])
    agg = agg.sort_values(["pitcher_trackman_id", "period"])
    gg = agg.groupby("pitcher_trackman_id")
    agg[f"cum_n_{label}"] = gg["n"].transform(lambda s: s.shift(1).cumsum())
    agg[f"cum_velo_{label}"] = gg["velo_sum"].transform(lambda s: s.shift(1).cumsum())
    agg[f"cum_spin_{label}"] = gg["spin_sum"].transform(lambda s: s.shift(1).cumsum())
    return agg[["pitcher_trackman_id", "period", f"cum_n_{label}", f"cum_velo_{label}", f"cum_spin_{label}"]]


def add_pressure_response_features(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv")

    relaxed = _situational_monthly(tm, (tm["balls_before"] == 0) & (tm["strikes_before"] == 0), "relaxed")
    pressure = _situational_monthly(tm, (tm["balls_before"] == 3) | (tm["strikes_before"] == 2), "pressure")

    df = df.merge(mapping, on="pitcher_id", how="left")
    df["period"] = _period(df["season"], df["game_month"])
    df = df.sort_values("period")
    relaxed["pitcher_trackman_id"] = relaxed["pitcher_trackman_id"].astype("float64")
    pressure["pitcher_trackman_id"] = pressure["pitcher_trackman_id"].astype("float64")

    merged = pd.merge_asof(df, relaxed.sort_values("period"), on="period", by="pitcher_trackman_id",
                            direction="backward", allow_exact_matches=True)
    merged = pd.merge_asof(merged, pressure.sort_values("period"), on="period", by="pitcher_trackman_id",
                            direction="backward", allow_exact_matches=True)

    relaxed_velo = merged["cum_velo_relaxed"] / merged["cum_n_relaxed"]
    pressure_velo = merged["cum_velo_pressure"] / merged["cum_n_pressure"]
    relaxed_spin = merged["cum_spin_relaxed"] / merged["cum_n_relaxed"]
    pressure_spin = merged["cum_spin_pressure"] / merged["cum_n_pressure"]

    merged["pitcher_pressure_velocity_drop"] = relaxed_velo - pressure_velo
    merged["pitcher_pressure_spin_drop"] = relaxed_spin - pressure_spin
    return merged.sort_index()


def main():
    mapping = get_confident_mapping()
    df = add_pressure_response_features(load("train.csv"), mapping)
    print(df[PRESSURE_FEATURES].describe())
    print("\nNaN 비율:", df[PRESSURE_FEATURES].isna().mean().to_dict())

    bc.FEATURES = list(bc.FEATURES) + PRESSURE_FEATURES
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"\n+ pressure_velocity_drop + pressure_spin_drop: score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")
    print("(비교 기준: baseline 734.49)")


if __name__ == "__main__":
    main()
