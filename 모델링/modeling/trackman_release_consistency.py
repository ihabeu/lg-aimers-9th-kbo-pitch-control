"""
Release point 일관성(std of rel_height, rel_side) — 제구는 릴리스포인트 일관성과 직결되는 물리적
특성이라 구속/spin/movement보다 더 직접적인 신호일 수 있다는 가설. 지금까지 avg만 썼고 std는 단독으로
테스트한 적 없음. v3 매핑(hand 일치율 96%) 사용, leak-safe cutoff은 기존과 동일.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
from trackman_mapping_v3 import get_confident_mapping  # noqa: E402
from trackman_features import build_monthly_agg, cumulative_profile, DATA_DIR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402
import pandas as pd
import numpy as np

RELEASE_CONSISTENCY_FEATURES = ["release_height_std", "release_side_std"]


def add_release_consistency_features(df, mapping):
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv")
    monthly = build_monthly_agg(tm)
    cum = cumulative_profile(monthly)

    for col in ["rel_height", "rel_side"]:
        n = cum[f"{col}_count"].replace(0, np.nan)
        mean = cum[f"{col}_sum"] / n
        var = cum[f"{col}_sumsq"] / n - mean ** 2
        cum[f"{col}_std"] = np.sqrt(var.clip(lower=0))

    feat = cum[["pitcher_trackman_id", "period", "rel_height_std", "rel_side_std"]].rename(
        columns={"rel_height_std": "release_height_std", "rel_side_std": "release_side_std"}
    ).sort_values("period")

    df = df.merge(mapping, on="pitcher_id", how="left")
    from trackman_features import _period
    df["period"] = _period(df["season"], df["game_month"])
    df = df.sort_values("period")
    feat["pitcher_trackman_id"] = feat["pitcher_trackman_id"].astype("float64")

    merged = pd.merge_asof(df, feat, on="period", by="pitcher_trackman_id", direction="backward", allow_exact_matches=False)
    return merged.sort_index()


def main():
    mapping = get_confident_mapping()
    df = add_release_consistency_features(load("train.csv"), mapping)
    print(df[RELEASE_CONSISTENCY_FEATURES].describe())
    print("\nNaN 비율:", df[RELEASE_CONSISTENCY_FEATURES].isna().mean().to_dict())

    bc.FEATURES = list(bc.FEATURES) + RELEASE_CONSISTENCY_FEATURES
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"\n+ release_height_std + release_side_std: score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")
    print("(비교 기준: baseline 734.49)")


if __name__ == "__main__":
    main()
