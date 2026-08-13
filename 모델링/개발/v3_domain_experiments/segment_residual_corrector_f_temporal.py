"""
dev(F) segment를 early/late 두 개로 더 쪼개서 각각 별도 corrector를 붙이는 게 도움되는지 확인.
예전에 f_season_progress를 base(CatBoost) "피처"로 추가했을 때는 기각됐는데(monotone_constraints
실험과 같은 이유로 CatBoost가 이미 game_month를 갖고 있어서 중복), 이번엔 다른 방식이다 --
"corrector를 어느 모델에 라우팅할지" 나누는 구조적 분리라, base_state 자체를 하나 더 늘리는 게 아님.

change point는 우리 자체 EDA로 재확인한 F 월별 성공률 패턴(2022/2023은 4월이 가장 높고 이후 하락)을
근거로 4월 vs 5월 이후로 나눈다. season 경계 안 넘는 값(그 시즌 라벨만으로 계산)이라 leak 없음.
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
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
SEEDS = (42, 2026, 314)
SEGMENTS_3WAY = ["core", "hybrid", "dev"]
SEGMENTS_5WAY = ["core", "hybrid", "dev_early", "dev_late"]
MIN_TRAIN, MIN_EVAL = 500, 100


def assign_segment_3way(df):
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


def assign_segment_f_temporal(df):
    is_f = df["game_type"] == "F"
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    is_early = df["game_month"] <= 4
    return np.where(is_f & is_early, "dev_early",
           np.where(is_f, "dev_late",
           np.where(involves_hybrid, "hybrid", "core")))


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


def crossfit_score(segments, X, residual, segment, base_pred, y, frame):
    seed_preds = []
    for seed in SEEDS:
        fold = pitcher_half(frame, seed)
        correction = np.zeros(len(frame))
        for half in (0, 1):
            tr_mask = fold != half
            ev_mask = fold == half
            for seg in segments:
                tr = tr_mask & (segment == seg)
                ev = ev_mask & (segment == seg)
                if tr.sum() < MIN_TRAIN or ev.sum() < MIN_EVAL:
                    correction[ev] = 0.0
                    continue
                model = ExtraTreesRegressor(n_jobs=-1, random_state=16200 + int(seed), **CORRECTOR_CFG)
                model.fit(X.loc[tr], residual[tr])
                correction[ev] = model.predict(X.loc[ev])
        seed_preds.append(np.clip(base_pred + correction, 0, 1))
    consensus = np.mean(np.column_stack(seed_preds), axis=1)
    return score(brier_score(y, consensus), y.mean())


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)

    base_model = fit_base(train_df, valid_df)
    base_pred = base_model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    base_score = score(brier_score(y, base_pred), y.mean())
    print("base BSS:", round(base_score, 2))

    X = corrector_matrix(valid_df)
    residual = y - base_pred

    seg3 = assign_segment_3way(valid_df)
    seg5 = assign_segment_f_temporal(valid_df)
    print("3-way dev n:", int((seg3 == "dev").sum()))
    print("f_temporal dev_early/dev_late n:", int((seg5 == "dev_early").sum()), int((seg5 == "dev_late").sum()))

    s3 = crossfit_score(SEGMENTS_3WAY, X, residual, seg3, base_pred, y, valid_df)
    s5 = crossfit_score(SEGMENTS_5WAY, X, residual, seg5, base_pred, y, valid_df)
    print(f"  3-way (dev 통합): {s3:.2f}")
    print(f"  F를 early/late로 분리: {s5:.2f} (차이 {s5 - s3:+.2f})")
    return base_score, s3, s5


def main():
    df = load("train.csv")
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n요약 (base, 3way, f_temporal_5way):")
    print("2023->2024:", r1)
    print("2022->2023:", r2)


if __name__ == "__main__":
    main()
