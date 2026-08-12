"""
segment_residual_corrector.py(R/F 2-way)를 3-way로 확장. 3번째 segment는 우리 자체 EDA로 이미
확인한 발견을 근거로 함: team 13이 F 참여 비율 38.15%로 다른 정상 팀(3~10%)보다 유독 높음
(HANDOFF.md "데이터 핵심 발견" 절 -- 이 세션 초반에 독립적으로 찾은 것).

segment 정의:
  dev  : game_type == 'F'
  hybrid: game_type == 'R' and (pitcher_team_id==13 or batter_team_id==13)
  core : 나머지 R
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
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
SEEDS = (42, 2026, 314)
SEGMENTS = ["core", "hybrid", "dev"]


def assign_segment(df: pd.DataFrame) -> np.ndarray:
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


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)
    segment = assign_segment(valid_df)
    print("segment 분포:", pd.Series(segment).value_counts().to_dict())

    base_model = fit_base(train_df, valid_df)
    base_pred = base_model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    base_score = score(brier_score(y, base_pred), y.mean())
    print("base BSS:", round(base_score, 2))
    for seg in SEGMENTS:
        mask = segment == seg
        if mask.sum() < 50:
            continue
        print(f"  base {seg}: {score(brier_score(y[mask], base_pred[mask]), y[mask].mean()):.2f} (n={int(mask.sum())})")

    X = corrector_matrix(valid_df)
    residual = y - base_pred

    seed_preds = []
    for seed in SEEDS:
        fold = pitcher_half(valid_df, seed)
        correction = np.zeros(len(valid_df))
        for half in (0, 1):
            tr_mask = fold != half
            ev_mask = fold == half
            for seg in SEGMENTS:
                tr = tr_mask & (segment == seg)
                ev = ev_mask & (segment == seg)
                if tr.sum() < 500 or ev.sum() < 100:
                    correction[ev] = 0.0
                    continue
                model = ExtraTreesRegressor(n_jobs=-1, random_state=16200 + int(seed), **CORRECTOR_CFG)
                model.fit(X.loc[tr], residual[tr])
                correction[ev] = model.predict(X.loc[ev])
        seed_preds.append(np.clip(base_pred + correction, 0, 1))
    consensus = np.mean(np.column_stack(seed_preds), axis=1)
    consensus_score = score(brier_score(y, consensus), y.mean())
    print("corrected BSS (3-seed consensus):", round(consensus_score, 2), " gain:", round(consensus_score - base_score, 2))
    for seg in SEGMENTS:
        mask = segment == seg
        if mask.sum() < 50:
            continue
        print(f"  corrected {seg}: {score(brier_score(y[mask], consensus[mask]), y[mask].mean()):.2f}")
    return base_score, consensus_score


def main():
    df = load("train.csv")
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n요약: 2023->2024 (base, corrected) =", r1, " / 2022->2023 =", r2)
    print("(비교: R/F 2-way corrector -- 2023->2024 base734.49/corrected790.33, 2022->2023 base10.25/corrected634.06)")


if __name__ == "__main__":
    main()
