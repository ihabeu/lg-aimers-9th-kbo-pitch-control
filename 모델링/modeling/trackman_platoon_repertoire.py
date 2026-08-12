"""
Trackman 마지막 실험: 투수 x 타자손별 과거 구종군 선택 비율 (count 분할 없음, 최대한 단순하게).

pitcher_vs_LHB_fastball_rate / breaking_rate / offspeed_rate
pitcher_vs_RHB_fastball_rate / breaking_rate / offspeed_rate

leak-safe: 현재 행의 (season, game_month) 이전 Trackman 기록만 사용. v2 매핑(395명, cost<=0.1462)만 대상,
매핑 안 된 투수는 NaN -> CatBoost 네이티브 결측 처리.

이 실험 결과가 마이너스면 Trackman feature engineering을 여기서 종료하기로 함(HANDOFF.md 결정 규칙 참고).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trackman_mapping_v2 import build_mapping_v2  # noqa: E402
from trackman_features import _period, DATA_DIR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

TRAIN_BATTER_HAND_MAP = {2: "Right", 1: "Left"}  # trackman_mapping_v2.TRAIN_HAND_MAP과 동일 방향(다수값 기준)

PLATOON_REPERTOIRE_FEATURES = [
    "pitcher_vs_current_hand_fastball_rate",
    "pitcher_vs_current_hand_breaking_rate",
    "pitcher_vs_current_hand_offspeed_rate",
]


def add_platoon_repertoire_features(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv",
                      usecols=["pitcher_trackman_id", "season", "game_month", "batter_hand", "pitch_type_group"])

    monthly = (
        tm.groupby(["pitcher_trackman_id", "batter_hand", "season", "game_month", "pitch_type_group"])
        .size().rename("n").reset_index()
        .pivot_table(index=["pitcher_trackman_id", "batter_hand", "season", "game_month"],
                     columns="pitch_type_group", values="n", fill_value=0)
        .reset_index()
    )
    monthly["period"] = _period(monthly["season"], monthly["game_month"])
    monthly = monthly.sort_values(["pitcher_trackman_id", "batter_hand", "period"])

    for col in ["fastball", "breaking", "offspeed"]:
        if col not in monthly.columns:
            monthly[col] = 0
    cum_cols = ["fastball", "breaking", "offspeed"]
    g = monthly.groupby(["pitcher_trackman_id", "batter_hand"])
    for col in cum_cols:
        monthly[f"cum_{col}"] = g[col].transform(lambda s: s.shift(1).cumsum())
    monthly["cum_total"] = monthly[[f"cum_{c}" for c in cum_cols]].sum(axis=1)

    df = df.merge(mapping, on="pitcher_id", how="left")
    df["period"] = _period(df["season"], df["game_month"])
    df["batter_hand_str"] = df["batter_hand"].map(TRAIN_BATTER_HAND_MAP)
    df = df.sort_values("period")

    monthly["pitcher_trackman_id"] = monthly["pitcher_trackman_id"].astype("float64")
    out_parts = []
    for hand in ["Left", "Right"]:
        sub_df = df[df["batter_hand_str"] == hand].copy()
        sub_feat = monthly[monthly["batter_hand"] == hand].sort_values("period")
        merged = pd.merge_asof(
            sub_df, sub_feat[["pitcher_trackman_id", "period", "cum_fastball", "cum_breaking", "cum_offspeed", "cum_total"]],
            on="period", by="pitcher_trackman_id", direction="backward", allow_exact_matches=True,
        )
        out_parts.append(merged)
    result = pd.concat(out_parts).sort_index()

    result["pitcher_vs_current_hand_fastball_rate"] = result["cum_fastball"] / result["cum_total"]
    result["pitcher_vs_current_hand_breaking_rate"] = result["cum_breaking"] / result["cum_total"]
    result["pitcher_vs_current_hand_offspeed_rate"] = result["cum_offspeed"] / result["cum_total"]
    return result


if __name__ == "__main__":
    mapping = build_mapping_v2()
    df = add_platoon_repertoire_features(load("train.csv"), mapping)
    print(df[PLATOON_REPERTOIRE_FEATURES].describe())
    print("\nNaN 비율:")
    print(df[PLATOON_REPERTOIRE_FEATURES].isna().mean())
