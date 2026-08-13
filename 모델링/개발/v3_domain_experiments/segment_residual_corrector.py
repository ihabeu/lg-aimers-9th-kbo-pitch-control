"""
우리 CatBoost baseline(789.23, 전체 44피처+전체데이터, 완전 독립개발)을 "base"로 그대로 두고,
그 오차(y - base예측)를 game_type(R/F) segment별로 ExtraTrees가 따로 학습해서 보정하는 구조.

이전에 실패했던 "R/F 완전 분리 모델"(623.31)과 다른 점: 여기선 base 모델 자체는 여전히 전체
데이터(R+F 같이)로 학습한다 -- 표본이 줄어드는 손해가 없다. 오직 "base가 남긴 오차의 패턴"만
segment별로 따로 본다.

방법론(이 프로젝트에서 이미 여러 번 쓴 것 그대로): pitcher-disjoint cross-fit으로 residual
corrector를 검증(다른 투수로 학습 -> 안 본 투수로 평가). 2023->2024(primary), 2022->2023(stress)
두 폴드 다 확인.
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

CORRECTOR_FEATURES = FEATURES  # base와 같은 44피처를 corrector 입력으로도 재사용
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
SEEDS = (42, 2026, 314)


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
    x = df[CORRECTOR_FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype("string").fillna("<NA>").astype("category").cat.codes.astype(np.float32)
    return x.apply(pd.to_numeric, errors="coerce")


def pitcher_half(frame, seed):
    pitchers = np.array(sorted(frame["pitcher_id"].astype(str).unique()))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(pitchers)
    first_half = set(pitchers[: len(pitchers) // 2])
    return np.where(frame["pitcher_id"].astype(str).isin(first_half).to_numpy(), 0, 1)


def run_fold(df, source_seasons_end, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season]
    y = valid_df["control_success"].to_numpy(np.float64)

    base_model = fit_base(train_df, valid_df)
    base_pred = base_model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    base_score = score(brier_score(y, base_pred), y.mean())
    print("base BSS:", round(base_score, 2))
    for gt in ["R", "F"]:
        mask = (valid_df["game_type"] == gt).to_numpy()
        print(f"  base {gt}: {score(brier_score(y[mask], base_pred[mask]), y[mask].mean()):.2f}")

    X = corrector_matrix(valid_df)
    residual = y - base_pred
    game_type = valid_df["game_type"].to_numpy()

    seed_preds = []
    for seed in SEEDS:
        fold = pitcher_half(valid_df, seed)
        correction = np.zeros(len(valid_df))
        for half in (0, 1):
            tr_mask = fold != half
            ev_mask = fold == half
            for gt in ["R", "F"]:
                tr = tr_mask & (game_type == gt)
                ev = ev_mask & (game_type == gt)
                model = ExtraTreesRegressor(n_jobs=-1, random_state=16200 + int(seed), **CORRECTOR_CFG)
                model.fit(X.loc[tr], residual[tr])
                correction[ev] = model.predict(X.loc[ev])
        seed_preds.append(np.clip(base_pred + correction, 0, 1))
    consensus = np.mean(np.column_stack(seed_preds), axis=1)
    consensus_score = score(brier_score(y, consensus), y.mean())
    print("corrected BSS (3-seed consensus):", round(consensus_score, 2), " gain:", round(consensus_score - base_score, 2))
    for gt in ["R", "F"]:
        mask = (valid_df["game_type"] == gt).to_numpy()
        print(f"  corrected {gt}: {score(brier_score(y[mask], consensus[mask]), y[mask].mean()):.2f}")
    return base_score, consensus_score


def main():
    df = load("train.csv")
    r1 = run_fold(df, None, 2024, "PRIMARY")
    r2 = run_fold(df, None, 2023, "STRESS")
    print("\n요약: 2023->2024 (base, corrected) =", r1, " / 2022->2023 =", r2)
    print("(비교: baseline 단일 2024 폴드 734.49)")


if __name__ == "__main__":
    main()
