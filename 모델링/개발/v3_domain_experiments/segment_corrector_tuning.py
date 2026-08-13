"""
ExtraTrees corrector의 용량(depth/n_estimators)과 보정 강도(shrink)를 튜닝. 지난번 V14 실험에서
local 그리드서치 최적점이 실제 LB에서는 오히려 나빴던 교훈(E007) 때문에, 여기서는 날카로운 최적점을
그대로 채택하지 않고 "두 폴드 다 안정적으로 좋은 평평한 구간"을 우선한다.

corrector 학습은 비싼 부분(pitcher-disjoint cross-fit, 3seed x 2half x 3segment)이라 depth/n_estimators
스윕은 그 안에서 반복하고, shrink는 correction 배열을 캐싱해서 값싸게 후처리로 스윕한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402

HYBRID_TEAM_ID = 13
SEEDS = (42, 2026, 314)
SEGMENTS = ["core", "hybrid", "dev"]
SHRINK_GRID = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
CAPACITY_GRID = [
    dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7),   # 기본(지금 채택된 값)
    dict(n_estimators=100, max_depth=6, min_samples_leaf=200, max_features=0.7),
    dict(n_estimators=100, max_depth=14, min_samples_leaf=200, max_features=0.7),
    dict(n_estimators=200, max_depth=10, min_samples_leaf=200, max_features=0.7),
    dict(n_estimators=100, max_depth=10, min_samples_leaf=50, max_features=0.7),
    dict(n_estimators=100, max_depth=10, min_samples_leaf=500, max_features=0.7),
]


def assign_segment(df):
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


def brier_score(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def score(brier, r):
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def fit_base(train_df, valid_df):
    train_pool = Pool(train_df[FEATURES], train_df["control_success"], cat_features=CAT_FEATURES)
    valid_pool = Pool(valid_df[FEATURES], valid_df["control_success"], cat_features=CAT_FEATURES)
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
        eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
        random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def corrector_matrix(df):
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype("string").fillna("<NA>").astype("category").cat.codes.astype(np.float32)
    return x.apply(pd.to_numeric, errors="coerce")


def pitcher_half(frame, seed):
    pitchers = np.array(sorted(frame["pitcher_id"].astype(str).unique()))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(pitchers)
    first_half = set(pitchers[: len(pitchers) // 2])
    return np.where(frame["pitcher_id"].astype(str).isin(first_half).to_numpy(), 0, 1)


def crossfit_correction(cfg, X, residual, segment, frame, seed):
    fold = pitcher_half(frame, seed)
    correction = np.zeros(len(frame))
    for half in (0, 1):
        tr_mask = fold != half
        ev_mask = fold == half
        for seg in SEGMENTS:
            tr = tr_mask & (segment == seg)
            ev = ev_mask & (segment == seg)
            model = ExtraTreesRegressor(n_jobs=-1, random_state=16200 + int(seed), **cfg)
            model.fit(X.loc[tr], residual[tr])
            correction[ev] = model.predict(X.loc[ev])
    return correction


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)
    segment = assign_segment(valid_df)

    base_model = fit_base(train_df, valid_df)
    base_pred = base_model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    base_score = score(brier_score(y, base_pred), y.mean())
    print("base BSS:", round(base_score, 2))

    X = corrector_matrix(valid_df)
    residual = y - base_pred

    print("--- capacity 스윕 (shrink=1.0 고정) ---")
    capacity_results = []
    default_correction = None
    for cfg in CAPACITY_GRID:
        seed_corrs = [crossfit_correction(cfg, X, residual, segment, valid_df, seed) for seed in SEEDS]
        correction = np.mean(np.column_stack(seed_corrs), axis=1)
        pred = np.clip(base_pred + correction, 0, 1)
        s = score(brier_score(y, pred), y.mean())
        capacity_results.append((cfg, s))
        print(f"  {cfg}: {s:.2f} (gain {s - base_score:+.2f})")
        if cfg == CAPACITY_GRID[0]:
            default_correction = correction

    print("--- shrink 스윕 (기본 capacity의 correction 재사용) ---")
    shrink_results = {}
    for shrink in SHRINK_GRID:
        pred = np.clip(base_pred + shrink * default_correction, 0, 1)
        s = score(brier_score(y, pred), y.mean())
        shrink_results[shrink] = s
        print(f"  shrink={shrink}: {s:.2f} (gain {s - base_score:+.2f})")

    return base_score, capacity_results, shrink_results


def main():
    df = load("train.csv")
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n요약 (base, capacity_results, shrink_results):")
    print("2023->2024:", r1)
    print("2022->2023:", r2)


if __name__ == "__main__":
    main()
