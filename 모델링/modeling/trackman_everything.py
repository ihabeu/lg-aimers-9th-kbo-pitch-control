"""
이 세션에서 만든 Trackman 파생 피처 전부 + 지금까지 안 쓴 컬럼(extension, zone_speed)까지 합쳐서
한 번에 테스트. v3 매핑(hand 일치율 96%) 사용. "부분적으로는 다 실패했으니 전부 합치면 다를까?"를
마지막으로 확인하는 실험 — 지금까지 패턴(피처 늘릴수록 나빠짐)을 감안하면 기대치는 낮지만 확인함.
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

ALL_NUMERIC = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]
PITCH_GROUPS = ["fastball", "breaking", "offspeed"]
ABBR = {"fastball": "fb", "breaking": "brk", "offspeed": "off"}


def build_everything(df: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv")

    # 1) 전체(구종 안 가리고) avg+std, 8개 컬럼 전부
    g = tm.groupby(["pitcher_trackman_id", "season", "game_month"])
    overall = g[ALL_NUMERIC].agg(["sum", lambda s: (s ** 2).sum(), "count"])
    overall.columns = [f"{c}_{f if f != '<lambda_0>' else 'sumsq'}" for c, f in overall.columns]
    overall = overall.reset_index()
    overall["period"] = _period(overall["season"], overall["game_month"])
    overall = overall.sort_values(["pitcher_trackman_id", "period"])
    go = overall.groupby("pitcher_trackman_id")
    for col in ALL_NUMERIC:
        overall[f"cs_{col}"] = go[f"{col}_sum"].transform(lambda s: s.shift(1).cumsum())
        overall[f"cc_{col}"] = go[f"{col}_count"].transform(lambda s: s.shift(1).cumsum())
        overall[f"csq_{col}"] = go[f"{col}_sumsq"].transform(lambda s: s.shift(1).cumsum())

    # 2) 구종군별 velocity/spin (repertoire)
    gp = tm.groupby(["pitcher_trackman_id", "pitch_type_group", "season", "game_month"])
    pitch = gp.agg(n=("rel_speed", "size"), velo_sum=("rel_speed", "sum"), spin_sum=("spin_rate", "sum")).reset_index()
    pitch["period"] = _period(pitch["season"], pitch["game_month"])
    pitch = pitch.sort_values(["pitcher_trackman_id", "pitch_type_group", "period"])
    gpp = pitch.groupby(["pitcher_trackman_id", "pitch_type_group"])
    pitch["cn"] = gpp["n"].transform(lambda s: s.shift(1).cumsum())
    pitch["cvelo"] = gpp["velo_sum"].transform(lambda s: s.shift(1).cumsum())
    pitch["cspin"] = gpp["spin_sum"].transform(lambda s: s.shift(1).cumsum())

    df = df.merge(mapping, on="pitcher_id", how="left")
    df["period"] = _period(df["season"], df["game_month"])
    df = df.sort_values("period")
    overall["pitcher_trackman_id"] = overall["pitcher_trackman_id"].astype("float64")
    pitch["pitcher_trackman_id"] = pitch["pitcher_trackman_id"].astype("float64")

    cs_cols = [c for c in overall.columns if c.startswith(("cs_", "cc_", "csq_"))]
    overall_sorted = overall[["pitcher_trackman_id", "period"] + cs_cols].sort_values("period")
    merged = pd.merge_asof(df, overall_sorted,
                            on="period", by="pitcher_trackman_id", direction="backward", allow_exact_matches=True)

    feature_cols = []
    for col in ALL_NUMERIC:
        n = merged[f"cc_{col}"].replace(0, np.nan)
        mean = merged[f"cs_{col}"] / n
        var = merged[f"csq_{col}"] / n - mean ** 2
        merged[f"tm_avg_{col}"] = mean
        merged[f"tm_std_{col}"] = np.sqrt(var.clip(lower=0))
        feature_cols += [f"tm_avg_{col}", f"tm_std_{col}"]

    total_n = None
    for grp in PITCH_GROUPS:
        sub = pitch[pitch["pitch_type_group"] == grp].sort_values("period")
        merged = pd.merge_asof(merged, sub[["pitcher_trackman_id", "period", "cn", "cvelo", "cspin"]],
                                on="period", by="pitcher_trackman_id", direction="backward", allow_exact_matches=True,
                                suffixes=("", f"_{grp}"))
        abbr = ABBR[grp]
        merged[f"{abbr}_n"] = merged["cn"] if total_n is None else merged["cn"]
        merged[f"{abbr}_velocity"] = merged["cvelo"] / merged["cn"]
        merged[f"{abbr}_spin"] = merged["cspin"] / merged["cn"]
        merged = merged.rename(columns={"cn": f"cn_{grp}", "cvelo": f"cvelo_{grp}", "cspin": f"cspin_{grp}"})
        feature_cols += [f"{abbr}_velocity", f"{abbr}_spin"]

    total_n = merged[["cn_fastball", "cn_breaking", "cn_offspeed"]].sum(axis=1)
    for grp in PITCH_GROUPS:
        abbr = ABBR[grp]
        merged[f"{abbr}_usage"] = merged[f"cn_{grp}"] / total_n
        feature_cols.append(f"{abbr}_usage")

    merged["fb_brk_velocity_sep"] = merged["fb_velocity"] - merged["brk_velocity"]
    merged["fb_off_velocity_sep"] = merged["fb_velocity"] - merged["off_velocity"]
    feature_cols += ["fb_brk_velocity_sep", "fb_off_velocity_sep"]

    usage = merged[["fb_usage", "brk_usage", "off_usage"]].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(usage > 0, usage * np.log(usage), 0.0)
    merged["repertoire_entropy"] = -np.nansum(terms, axis=1)
    merged.loc[total_n.isna() | (total_n == 0), "repertoire_entropy"] = np.nan
    feature_cols.append("repertoire_entropy")

    return merged.sort_index(), feature_cols


def main():
    mapping = get_confident_mapping()
    df, feature_cols = build_everything(load("train.csv"), mapping)
    print(f"Trackman 파생 피처 총 {len(feature_cols)}개")
    print(f"커버리지: {df[feature_cols[0]].notna().mean():.1%}")

    bc.FEATURES = list(bc.FEATURES) + feature_cols
    train_df, valid_df = bc.time_split(df, 2024)
    model = bc.train_catboost(train_df, valid_df)
    m = bc.evaluate(model, valid_df)
    print(f"\n+ Trackman 전부({len(feature_cols)}개): score={m['score (리더보드 산식)']:.2f} brier={m['brier']:.6f}")
    print("(비교 기준: baseline 734.49)")

    imp = pd.Series(model.get_feature_importance(bc.to_pool(valid_df)), index=bc.FEATURES)
    tm_imp = imp[feature_cols].sort_values(ascending=False)
    print("\nTrackman 파생 피처 importance 순위:")
    print(tm_imp.to_string())


if __name__ == "__main__":
    main()
