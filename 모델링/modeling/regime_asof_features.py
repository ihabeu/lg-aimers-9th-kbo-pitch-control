"""
STEP 1~3: 2023 regime-aware as-of rate + R/F(1군/2군 추정) 분리 as-of rate.

기존 asof_pitcher_success_rate는 커리어 전체 누적이라, 2020~2022(regime 1) 정보가 2024 예측에
계속 섞여 들어간다. train.csv 자체를 self-referential하게 써서(leak-safe, (season,month) 이전
데이터만) 아래를 새로 만든다:

- pitcher_post2023_rate: season>=2023인 행만으로 누적한 성공률 (2023 이전 시즌 데이터는 아예 제외)
- pitcher_post2023_diff: post2023_rate - asof_pitcher_success_rate(기존 커리어 전체)
- pitcher_R_rate / pitcher_F_rate: game_type별로 나눠서 누적한 성공률 (전체 기간)
- pitcher_RF_diff: R_rate - F_rate
- pitcher_post2023_R_rate / pitcher_post2023_F_rate: 위 둘의 교집합 (2023 이후 & game_type별)

타자는 asof_batter_prev*_game_success_rate 자체가 원래 없어서(이전에 확인됨) 우선 투수만.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402


def _period(season: pd.Series, month: pd.Series) -> pd.Series:
    return season.astype(int) * 12 + month.astype(int)


def _cumulative_rate(df: pd.DataFrame, group_cols: list[str], out_col: str) -> pd.Series:
    """group_cols(예: [pitcher_id] 또는 [pitcher_id, game_type]) 기준으로 (season,month) 이전까지의
    누적 성공률. df는 이미 원하는 subset(예: season>=2023)으로 필터링된 상태여야 한다."""
    monthly = (
        df.groupby(group_cols + ["season", "game_month"])["control_success"]
        .agg(n="size", success="sum").reset_index()
    )
    monthly["period"] = _period(monthly["season"], monthly["game_month"])
    monthly = monthly.sort_values(group_cols + ["period"])
    g = monthly.groupby(group_cols)
    monthly["cum_n"] = g["n"].transform(lambda s: s.shift(1).cumsum())
    monthly["cum_success"] = g["success"].transform(lambda s: s.shift(1).cumsum())
    monthly[out_col] = monthly["cum_success"] / monthly["cum_n"]

    df = df.copy()
    df["period"] = _period(df["season"], df["game_month"])
    df_sorted = df.sort_values("period")
    monthly_sorted = monthly.sort_values("period")
    merged = pd.merge_asof(
        df_sorted, monthly_sorted[group_cols + ["period", out_col]],
        on="period", by=group_cols, direction="backward", allow_exact_matches=True,
    )
    return merged.sort_index()[out_col]


def add_regime_asof_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ① post-2023 (2023 이전 시즌 행은 아예 제외하고 누적)
    post2023_source = df[df["season"] >= 2023]
    df["pitcher_post2023_rate"] = _cumulative_rate(post2023_source, ["pitcher_id"], "pitcher_post2023_rate").reindex(df.index)
    df["pitcher_post2023_diff"] = df["pitcher_post2023_rate"] - df["asof_pitcher_success_rate"]

    # ② R/F 분리 (전체 기간, game_type별)
    for gt in ["R", "F"]:
        sub = df[df["game_type"] == gt]
        col = f"pitcher_{gt}_rate"
        df[col] = _cumulative_rate(sub, ["pitcher_id"], col).reindex(df.index)
    df["pitcher_RF_diff"] = df["pitcher_R_rate"] - df["pitcher_F_rate"]

    # ③ post-2023 x R/F 교집합
    post2023_df = df[df["season"] >= 2023]
    for gt in ["R", "F"]:
        sub = post2023_df[post2023_df["game_type"] == gt]
        col = f"pitcher_post2023_{gt}_rate"
        df[col] = _cumulative_rate(sub, ["pitcher_id"], col).reindex(df.index)

    return df


REGIME_FEATURES_1 = ["pitcher_post2023_rate", "pitcher_post2023_diff"]
REGIME_FEATURES_2 = ["pitcher_R_rate", "pitcher_F_rate", "pitcher_RF_diff"]
REGIME_FEATURES_3 = ["pitcher_post2023_R_rate", "pitcher_post2023_F_rate"]

if __name__ == "__main__":
    df = add_regime_asof_features(load("train.csv"))
    cols = REGIME_FEATURES_1 + REGIME_FEATURES_2 + REGIME_FEATURES_3
    print(df[cols].describe())
    print("\nNaN 비율:")
    print(df[cols].isna().mean())
