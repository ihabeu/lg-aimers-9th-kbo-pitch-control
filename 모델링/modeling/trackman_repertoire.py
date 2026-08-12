"""
STEP 2: Trackman repertoire (사용비율 + 구종별 physical + separation + entropy).

기존 실패한 static profile/drift는 "전체 평균 하나"였다. 이번엔 구종군(fastball/breaking/offspeed)별로
쪼개서 "투수가 어떤 무기를 어떤 비율로, 어떤 물리적 특성으로 던지는가"를 표현한다. v2 매핑(395명,
헝가리안) 사용, leak-safe cutoff은 기존과 동일((season,month) 이전만).

- {fb,brk,off}_usage: 구종군별 사용비율
- {fb,brk,off}_velocity / spin: 구종군별 평균 구속/회전
- fb_brk_velocity_sep = fb_velocity - brk_velocity (구종 간 물리적 분리도)
- repertoire_entropy: Shannon entropy of (fb_usage, brk_usage, off_usage) — 무기가 다양할수록 큼
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trackman_mapping_v2 import build_mapping_v2  # noqa: E402
from trackman_features import _period, DATA_DIR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

PITCH_GROUPS = ["fastball", "breaking", "offspeed"]
PITCH_ABBR = {"fastball": "fb", "breaking": "brk", "offspeed": "off"}

REPERTOIRE_FEATURES = (
    [f"{PITCH_ABBR[g]}_usage" for g in PITCH_GROUPS]
    + [f"{PITCH_ABBR[g]}_velocity" for g in PITCH_GROUPS]
    + [f"{PITCH_ABBR[g]}_spin" for g in PITCH_GROUPS]
    + ["fb_brk_velocity_sep", "fb_off_velocity_sep", "fb_brk_spin_sep", "repertoire_entropy"]
)


def build_pitch_type_monthly(tm: pd.DataFrame) -> pd.DataFrame:
    g = tm.groupby(["pitcher_trackman_id", "pitch_type_group", "season", "game_month"])
    agg = g.agg(n=("rel_speed", "size"), velo_sum=("rel_speed", "sum"), spin_sum=("spin_rate", "sum"))
    agg = agg.reset_index()
    agg["period"] = _period(agg["season"], agg["game_month"])
    return agg.sort_values(["pitcher_trackman_id", "pitch_type_group", "period"])


def add_repertoire_features(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv",
                      usecols=["pitcher_trackman_id", "season", "game_month", "pitch_type_group", "rel_speed", "spin_rate"])
    monthly = build_pitch_type_monthly(tm)

    g = monthly.groupby(["pitcher_trackman_id", "pitch_type_group"])
    monthly["cum_n"] = g["n"].transform(lambda s: s.shift(1).cumsum())
    monthly["cum_velo"] = g["velo_sum"].transform(lambda s: s.shift(1).cumsum())
    monthly["cum_spin"] = g["spin_sum"].transform(lambda s: s.shift(1).cumsum())

    df = df.merge(mapping, on="pitcher_id", how="left")
    df["period"] = _period(df["season"], df["game_month"])
    df = df.sort_values("period")
    monthly["pitcher_trackman_id"] = monthly["pitcher_trackman_id"].astype("float64")

    parts = []
    for grp in PITCH_GROUPS:
        sub_feat = monthly[monthly["pitch_type_group"] == grp].sort_values("period")
        merged = pd.merge_asof(
            df, sub_feat[["pitcher_trackman_id", "period", "cum_n", "cum_velo", "cum_spin"]],
            on="period", by="pitcher_trackman_id", direction="backward", allow_exact_matches=True,
        )
        abbr = PITCH_ABBR[grp]
        merged[f"{abbr}_n"] = merged["cum_n"]
        merged[f"{abbr}_velocity"] = merged["cum_velo"] / merged["cum_n"]
        merged[f"{abbr}_spin"] = merged["cum_spin"] / merged["cum_n"]
        parts.append(merged[[f"{abbr}_n", f"{abbr}_velocity", f"{abbr}_spin"]])

    result = pd.concat([df.reset_index(drop=True)] + [p.reset_index(drop=True) for p in parts], axis=1)
    result.index = df.index

    total_n = result[["fb_n", "brk_n", "off_n"]].sum(axis=1)
    for grp in PITCH_GROUPS:
        abbr = PITCH_ABBR[grp]
        result[f"{abbr}_usage"] = result[f"{abbr}_n"] / total_n

    result["fb_brk_velocity_sep"] = result["fb_velocity"] - result["brk_velocity"]
    result["fb_off_velocity_sep"] = result["fb_velocity"] - result["off_velocity"]
    result["fb_brk_spin_sep"] = result["fb_spin"] - result["brk_spin"]

    usage = result[["fb_usage", "brk_usage", "off_usage"]].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(usage > 0, usage * np.log(usage), 0.0)
    result["repertoire_entropy"] = -np.nansum(entropy_terms, axis=1)
    result.loc[total_n.isna() | (total_n == 0), "repertoire_entropy"] = np.nan

    return result


if __name__ == "__main__":
    mapping = build_mapping_v2()
    df = add_repertoire_features(load("train.csv"), mapping)
    print(df[REPERTOIRE_FEATURES].describe())
    print("\nNaN 비율:")
    print(df[REPERTOIRE_FEATURES].isna().mean())
