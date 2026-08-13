"""
계층형 EB(Empirical Bayes) target-encoding + 시즌 경계 분해 피처.

다른 참가자 공개 레포 V25.5 노트북(REJECT됐지만 R 쪽만 놓고 보면 로컬 792.65로 우리 baseline보다 높았음)의
구조를 간소화해서 이식: raw pitcher_id/batter_id 대신 global -> pitcher/batter -> pitcher×상황 순으로
shrink 강도를 다르게 준 사후평균을 피처로 쓴다. 그리고 공식 asof_pitcher_n/asof_pitcher_success_rate는
커리어 누적이므로, 우리가 미리 집계한 "이전 시즌 총합"을 빼면 "이번 시즌 들어 지금까지"를 대수적으로
복원할 수 있다 — 투수별 상수 하나만 빼는 연산이라 test 행 순서가 필요 없어서 recent_k_pitch_rate가
LB에서 무너진 것과 같은 스냅샷 문제를 안 밟는다.

leak-safe: 각 행의 season보다 이전 시즌의 "R" 데이터만으로 집계한다(우리 데이터엔 날짜/경기ID가 없어
season 단위가 현실적인 leak-safe 최소 단위 — 원본 asof_*보다는 성기다). rolling 폴드마다 target_year가
다르므로 폴드마다 새로 집계해야 한다(과거 회차 EDA2/regime_asof_features와 동일한 원칙).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load, TARGET  # noqa: E402

PRIOR = 0.52  # global fallback (2019처럼 이전 시즌이 아예 없는 경우)

# (이름, group_keys, parent, shrink strength) — leaf일수록 strength를 크게 줘서 더 세게 당긴다.
HIER = [
    ("pitcher", ["pitcher_id"], "global", 200.0),
    ("batter", ["batter_id"], "global", 250.0),
    ("pitcher_bhand", ["pitcher_id", "batter_hand"], "pitcher", 100.0),
    ("pitcher_count", ["pitcher_id", "count_state"], "pitcher", 150.0),
    ("pitcher_pressure", ["pitcher_id", "pressure_state"], "pitcher", 150.0),
    ("batter_phand", ["batter_id", "pitcher_hand"], "batter", 150.0),
    ("team_match", ["pitcher_team_id", "batter_team_id"], "global", 500.0),
]
SEASON_K = 60.0  # 시즌 경계 분해값의 shrink 강도

EB_FEATURES = (
    ["eb_global_rate"]
    + [f"eb_{name}_rate" for name, *_ in HIER]
    + [f"eb_{name}_rel" for name, *_ in HIER]
    + ["season_pitcher_n", "season_pitcher_rate"]
)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["count_state"] = df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)
    full = (df["balls_before"] == 3) & (df["strikes_before"] == 2)
    high = (df["balls_before"] == 3) | (df["strikes_before"] == 2)
    df["pressure_state"] = np.select([full, high], ["full", "high"], default="normal")
    return df


def _agg(hist: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return hist.groupby(keys)[TARGET].agg(n="size", s="sum")


def _map(agg: pd.DataFrame, rows: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    idx = pd.MultiIndex.from_frame(rows[keys]) if len(keys) > 1 else pd.Index(rows[keys[0]])
    n = agg["n"].reindex(idx).fillna(0).to_numpy(float)
    s = agg["s"].reindex(idx).fillna(0).to_numpy(float)
    return n, s


def _build_for_rows(hist_r: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """hist_r: rows보다 이전 시즌의 R 행. rows: 이 시즌의 행(R/F 무관)."""
    out = pd.DataFrame(index=rows.index)
    n_hist = len(hist_r)
    global_s = float(hist_r[TARGET].sum()) if n_hist else 0.0
    global_rate = (global_s + 800.0 * PRIOR) / (n_hist + 800.0)
    out["eb_global_rate"] = global_rate
    parent_rate = {"global": np.full(len(rows), global_rate)}

    for name, keys, parent, k in HIER:
        if n_hist == 0:
            post = parent_rate[parent]
            out[f"eb_{name}_rate"] = post
            out[f"eb_{name}_rel"] = 0.0
        else:
            agg = _agg(hist_r, keys)
            n, s = _map(agg, rows, keys)
            pr = parent_rate[parent]
            post = (s + k * pr) / (n + k)
            out[f"eb_{name}_rate"] = post.astype("float32")
            out[f"eb_{name}_rel"] = (n / (n + k)).astype("float32")
        parent_rate[name] = post

    # 시즌 경계 분해: asof_pitcher_n/success_rate(커리어 누적) - 이전 시즌 총합 = 이번 시즌 누적.
    if n_hist:
        ph = hist_r.groupby("pitcher_id")[TARGET].agg(n="size", s="sum")
        idx = pd.Index(rows["pitcher_id"])
        hn = ph["n"].reindex(idx).fillna(0).to_numpy(float)
        hs = ph["s"].reindex(idx).fillna(0).to_numpy(float)
    else:
        hn = np.zeros(len(rows))
        hs = np.zeros(len(rows))
    an = rows["asof_pitcher_n"].fillna(0).clip(lower=0).to_numpy(float)
    ar = rows["asof_pitcher_success_rate"].fillna(PRIOR).to_numpy(float)
    cn = np.clip(an - hn, 0, None)
    cs = np.clip(an * ar - hs, 0, cn)
    prior = out["eb_pitcher_rate"].to_numpy(float)
    out["season_pitcher_n"] = cn.astype("float32")
    out["season_pitcher_rate"] = ((cs + SEASON_K * prior) / (cn + SEASON_K)).astype("float32")
    return out


def add_hier_features(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """season <= target_year인 모든 행에 대해, 각 행 자기 시즌 이전의 R 이력만으로 EB 피처를 붙인다."""
    df = _prep(df)
    scoped = df[df["season"] <= target_year]
    pieces = []
    for y in sorted(scoped["season"].unique()):
        rows = scoped[scoped["season"] == y]
        hist_r = df[(df["season"] < y) & (df["game_type"] == "R")]
        pieces.append(_build_for_rows(hist_r, rows))
    hier = pd.concat(pieces)
    return scoped.join(hier)


def run(raw: pd.DataFrame, with_eb: bool, label: str) -> dict:
    base_features, base_cats = list(bc.FEATURES), list(bc.CAT_FEATURES)
    if with_eb:
        bc.FEATURES = base_features + EB_FEATURES

    # primary(2024) 조기종료로 iterations 고정 (baseline_catboost.py와 동일 절차)
    frame_2024 = add_hier_features(raw, 2024) if with_eb else _prep(raw[raw["season"] <= 2024])
    train_df, valid_df = bc.time_split(frame_2024, 2024)
    valid_r = valid_df[valid_df["game_type"] == "R"]
    primary = bc.train_catboost(train_df, valid_r)
    iterations = primary.get_best_iteration() + 1
    print(f"\n{label}: primary(2024, R만 평가) best_iteration={iterations}")

    per_fold = {}
    weighted_brier = 0.0
    weighted_score = 0.0
    for valid_season, weight in bc.ROLLING_FOLDS:
        frame = add_hier_features(raw, valid_season) if with_eb else _prep(raw[raw["season"] <= valid_season])
        tr, va = bc.time_split(frame, valid_season)
        va_r = va[va["game_type"] == "R"]
        model = bc.train_catboost_fixed(tr, iterations)
        m = bc.evaluate(model, va_r)
        m["weight"] = weight
        per_fold[valid_season] = m
        weighted_brier += weight * m["brier"]
        weighted_score += weight * m["score (리더보드 산식)"]
    per_fold["weighted"] = {"brier": round(weighted_brier, 6), "score": round(weighted_score, 2)}
    for season, m in per_fold.items():
        print(f"  {season}: {m}")

    bc.FEATURES, bc.CAT_FEATURES = base_features, base_cats
    return per_fold


def main():
    raw = load("train.csv")
    run(raw, False, "baseline")
    run(raw, True, "+hierarchical EB")


if __name__ == "__main__":
    main()
