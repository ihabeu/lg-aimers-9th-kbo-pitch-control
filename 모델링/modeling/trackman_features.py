"""
Trackman 과거 투수 프로필 피처 (1차: historical profile, ~10개).

trackman_mapping.py로 복원한 pitcher_id <-> pitcher_trackman_id 매핑을 이용해,
각 train 행의 (season, game_month) 시점 "이전"에 해당 투수가 던진 Trackman 투구만으로
평균/표준편차/구종비율을 계산한다. train.csv엔 game_date가 없어 season+game_month가
확보 가능한 가장 세밀한 cutoff 단위다 (같은 달 안의 순서는 구분 불가 -> 같은 달 데이터는 전부 제외).

- hist_avg_rel_speed, hist_std_rel_speed
- hist_avg_spin_rate, hist_std_spin_rate
- hist_avg_induced_vert_break, hist_avg_horz_break
- hist_avg_extension, hist_avg_rel_height, hist_avg_rel_side
- hist_fastball_ratio, hist_breaking_ratio, hist_offspeed_ratio

매핑 안 된(비신뢰) pitcher_id는 전부 NaN -> CatBoost 네이티브 결측 처리.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trackman_mapping import build_mapping  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

TM_NUMERIC = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side"]
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

HIST_FEATURES = [
    "hist_avg_rel_speed", "hist_std_rel_speed",
    "hist_avg_spin_rate", "hist_std_spin_rate",
    "hist_avg_induced_vert_break", "hist_avg_horz_break",
    "hist_avg_extension", "hist_avg_rel_height", "hist_avg_rel_side",
    "hist_fastball_ratio", "hist_breaking_ratio", "hist_offspeed_ratio",
]


def _period(season: pd.Series, month: pd.Series) -> pd.Series:
    return season.astype(int) * 12 + month.astype(int)


def build_monthly_agg(tm: pd.DataFrame) -> pd.DataFrame:
    """pitcher_trackman_id x period(season,month) 단위 월간 집계 (합/제곱합/표본수/구종군 카운트)."""
    g = tm.groupby(["pitcher_trackman_id", "season", "game_month"])
    agg = g[TM_NUMERIC].agg(["sum", lambda s: (s ** 2).sum(), "count"])
    agg.columns = [f"{c}_{f if f != '<lambda_0>' else 'sumsq'}" for c, f in agg.columns]
    pitch_counts = g["pitch_type_group"].value_counts().unstack(fill_value=0)
    pitch_counts.columns = [f"n_{c}" for c in pitch_counts.columns]
    out = agg.join(pitch_counts).reset_index()
    out["period"] = _period(out["season"], out["game_month"])
    return out.sort_values(["pitcher_trackman_id", "period"])


def cumulative_profile(monthly: pd.DataFrame) -> pd.DataFrame:
    """투수별로 period 기준 누적(cumsum) -> 해당 시점까지의 전체 이력. merge_asof에서 strictly-before로 join."""
    monthly = monthly.sort_values(["pitcher_trackman_id", "period"]).copy()
    cum_cols = [c for c in monthly.columns if c.endswith(("_sum", "_sumsq", "_count")) or c.startswith("n_")]
    monthly[cum_cols] = monthly.groupby("pitcher_trackman_id")[cum_cols].cumsum()
    return monthly


def profile_to_features(cum: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=cum.index)
    n_total = cum["n_fastball"] + cum["n_breaking"] + cum["n_offspeed"] + cum.get("n_other", 0)
    for col in TM_NUMERIC:
        n = cum[f"{col}_count"].replace(0, np.nan)
        mean = cum[f"{col}_sum"] / n
        var = cum[f"{col}_sumsq"] / n - mean ** 2
        if col == "rel_speed":
            out["hist_avg_rel_speed"], out["hist_std_rel_speed"] = mean, np.sqrt(var.clip(lower=0))
        elif col == "spin_rate":
            out["hist_avg_spin_rate"], out["hist_std_spin_rate"] = mean, np.sqrt(var.clip(lower=0))
        elif col == "induced_vert_break":
            out["hist_avg_induced_vert_break"] = mean
        elif col == "horz_break":
            out["hist_avg_horz_break"] = mean
        elif col == "extension":
            out["hist_avg_extension"] = mean
        elif col == "rel_height":
            out["hist_avg_rel_height"] = mean
        elif col == "rel_side":
            out["hist_avg_rel_side"] = mean
    out["hist_fastball_ratio"] = cum["n_fastball"] / n_total
    out["hist_breaking_ratio"] = cum["n_breaking"] / n_total
    out["hist_offspeed_ratio"] = cum["n_offspeed"] / n_total
    out["pitcher_trackman_id"] = cum["pitcher_trackman_id"]
    out["period"] = cum["period"]
    return out


def add_trackman_history_features(df: pd.DataFrame, mapping: pd.DataFrame | None = None) -> pd.DataFrame:
    if mapping is None:
        mapping = build_mapping()
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv")

    monthly = build_monthly_agg(tm)
    cum = cumulative_profile(monthly)
    feat = profile_to_features(cum).sort_values("period")

    df = df.merge(mapping, on="pitcher_id", how="left")
    df["period"] = _period(df["season"], df["game_month"])
    df = df.sort_values("period")
    # 매핑 안 된 행은 NaN이라 pitcher_trackman_id가 float64가 됨 -> feat 쪽도 float64로 맞춰야 merge_asof가 통과함.
    feat["pitcher_trackman_id"] = feat["pitcher_trackman_id"].astype("float64")

    merged = pd.merge_asof(
        df, feat, on="period", by="pitcher_trackman_id",
        direction="backward", allow_exact_matches=False,
    )
    return merged.sort_index()


if __name__ == "__main__":
    df = add_trackman_history_features(load("train.csv"))
    coverage = df["hist_avg_rel_speed"].notna().mean()
    print(f"train 행 기준 Trackman 이력 커버리지: {coverage:.1%}")
    print(df[HIST_FEATURES].describe())
