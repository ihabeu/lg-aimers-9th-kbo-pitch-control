"""
corrector 입력에 base(CatBoost) 예측값 자체를 피처로 추가하면 도움되는지 확인. 지금까지는 원본
44피처만 corrector에 넣었는데, base가 얼마나 확신 있게(0.9 근처) 예측했는지 애매하게(0.5 근처)
예측했는지에 따라 오차 패턴이 다를 수 있다는 아이디어(V14도 이 방식을 씀) -- 아직 우리 파이프라인엔
없었음.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402

HYBRID_TEAM_ID = 13
SEEDS = (42, 2026, 314)
SEGMENTS = ["core", "hybrid", "dev"]
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)


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


def corrector_matrix(df, base_pred=None):
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype("string").fillna("<NA>").astype("category").cat.codes.astype(np.float32)
    x = x.apply(pd.to_numeric, errors="coerce")
    if base_pred is not None:
        x = x.copy()
        x["base_pred"] = np.asarray(base_pred, dtype=np.float32)
        logit = np.clip(base_pred, 1e-6, 1 - 1e-6)
        x["base_pred_logit"] = np.log(logit / (1 - logit)).astype(np.float32)
        x["base_pred_confidence"] = np.abs(base_pred - 0.5).astype(np.float32)
    return x


def pitcher_half(frame, seed):
    pitchers = np.array(sorted(frame["pitcher_id"].astype(str).unique()))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(pitchers)
    first_half = set(pitchers[: len(pitchers) // 2])
    return np.where(frame["pitcher_id"].astype(str).isin(first_half).to_numpy(), 0, 1)


def crossfit_correction(X, residual, segment, frame, seed):
    fold = pitcher_half(frame, seed)
    correction = np.zeros(len(frame))
    for half in (0, 1):
        tr_mask = fold != half
        ev_mask = fold == half
        for seg in SEGMENTS:
            tr = tr_mask & (segment == seg)
            ev = ev_mask & (segment == seg)
            model = ExtraTreesRegressor(n_jobs=-1, random_state=16200 + int(seed), **CORRECTOR_CFG)
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

    residual = y - base_pred

    X_without = corrector_matrix(valid_df)
    seed_preds = [np.clip(base_pred + crossfit_correction(X_without, residual, segment, valid_df, s), 0, 1) for s in SEEDS]
    without_score = score(brier_score(y, np.mean(np.column_stack(seed_preds), axis=1)), y.mean())
    print(f"  base_pred 피처 없음 (지금 채택된 것): {without_score:.2f} (gain {without_score - base_score:+.2f})")

    X_with = corrector_matrix(valid_df, base_pred=base_pred)
    seed_preds2 = [np.clip(base_pred + crossfit_correction(X_with, residual, segment, valid_df, s), 0, 1) for s in SEEDS]
    with_score = score(brier_score(y, np.mean(np.column_stack(seed_preds2), axis=1)), y.mean())
    print(f"  base_pred 피처 추가:              {with_score:.2f} (gain {with_score - base_score:+.2f})")
    print(f"  차이: {with_score - without_score:+.2f}")
    return base_score, without_score, with_score


def main():
    df = load("train.csv")
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n요약 (base, without, with):")
    print("2023->2024:", r1)
    print("2022->2023:", r2)


if __name__ == "__main__":
    main()
