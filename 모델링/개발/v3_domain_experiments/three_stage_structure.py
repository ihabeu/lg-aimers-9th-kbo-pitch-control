"""
3-stage 구조 실험: 각 stage가 서로 다른 정보만 보도록 강제로 분리.
  Stage 1 (ability)  : 선수 이력/커리어 정보만 (asof_*, season, game_type, team_id) -> "이 선수는 원래 어떤가"
  Stage 2 (situation) : 상황 정보만(카운트/이닝/점수차/주자/승리기대치/월/요일/손 등, 선수 이력 제외)
                         -> Stage1 잔차를 상황만으로 설명
  Stage 3 (segment correction) : 지금 쓰는 CORE/HYBRID/DEV ExtraTrees corrector, 전체 44피처로
                         Stage1+Stage2 잔차를 마저 보정 (기존 구조 그대로 마지막에 얹음)

기존 2-stage(CatBoost 전체피처 + corrector, local 801.93)와 비교.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402

HYBRID_TEAM_ID = 13
SEGMENTS = ["core", "hybrid", "dev"]
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
SEEDS = (42, 2026, 314)

ABILITY_FEATURES = [
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "season", "game_type", "pitcher_team_id", "batter_team_id",
]
ABILITY_CAT = ["game_type", "pitcher_team_id", "batter_team_id"]

SITUATION_FEATURES = [
    "balls_before", "strikes_before", "outs_before", "num_runners_on",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "home_win_expectancy", "away_win_expectancy", "li", "inning",
    "top_bottom", "pitcher_hand", "batter_hand",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "base_state",
    "game_month", "game_dayofweek",
]
SITUATION_CAT = ["top_bottom", "pitcher_hand", "batter_hand", "runner_on_1b", "runner_on_2b",
                  "runner_on_3b", "base_state", "game_month", "game_dayofweek"]


def assign_segment(df):
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


def score(brier, r):
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


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


def crossfit_stage3(X, residual, segment, base_pred, y, frame):
    seed_preds = []
    for seed in SEEDS:
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
        seed_preds.append(np.clip(base_pred + correction, 0, 1))
    consensus = np.mean(np.column_stack(seed_preds), axis=1)
    return score(float(np.mean((consensus - y) ** 2)), float(y.mean())), consensus


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y_train = train_df["control_success"].to_numpy(np.float64)
    y = valid_df["control_success"].to_numpy(np.float64)
    segment = assign_segment(valid_df)

    # ----- 기존 2-stage 기준선 -----
    base_pool_tr = Pool(train_df[FEATURES], y_train, cat_features=CAT_FEATURES)
    base_pool_va = Pool(valid_df[FEATURES], y, cat_features=CAT_FEATURES)
    base_model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
                                     eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
                                     random_seed=42, thread_count=-1, verbose=False)
    base_model.fit(base_pool_tr, eval_set=base_pool_va, use_best_model=True)
    base_pred = base_model.predict_proba(base_pool_va)[:, 1]
    base_residual = y - base_pred
    X_full = corrector_matrix(valid_df)
    s_2stage, _ = crossfit_stage3(X_full, base_residual, segment, base_pred, y, valid_df)
    print(f"  기존 2-stage(전체피처 base + corrector): {s_2stage:.2f}")

    # ----- Stage 1: ability만 -----
    s1_pool_tr = Pool(train_df[ABILITY_FEATURES], y_train, cat_features=ABILITY_CAT)
    s1_pool_va = Pool(valid_df[ABILITY_FEATURES], y, cat_features=ABILITY_CAT)
    stage1 = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
                                 eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
                                 random_seed=42, thread_count=-1, verbose=False)
    stage1.fit(s1_pool_tr, eval_set=s1_pool_va, use_best_model=True)
    p1_train = stage1.predict_proba(Pool(train_df[ABILITY_FEATURES], cat_features=ABILITY_CAT))[:, 1]
    p1_valid = stage1.predict_proba(s1_pool_va)[:, 1]
    r = float(y.mean())
    print(f"  Stage1(ability만) 단독: {score(float(np.mean((p1_valid - y) ** 2)), r):.2f}")

    # ----- Stage 2: situation만으로 Stage1 잔차 설명 -----
    resid1_train = y_train - p1_train
    s2_pool_tr = Pool(train_df[SITUATION_FEATURES], resid1_train, cat_features=SITUATION_CAT)
    stage2 = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, loss_function="RMSE",
                                l2_leaf_reg=L2_LEAF_REG, random_seed=42, thread_count=-1, verbose=False)
    stage2.fit(s2_pool_tr)
    p2_valid = stage2.predict(Pool(valid_df[SITUATION_FEATURES], cat_features=SITUATION_CAT))
    stage12_pred = np.clip(p1_valid + p2_valid, 0, 1)
    print(f"  Stage1+Stage2(ability+situation 분리): {score(float(np.mean((stage12_pred - y) ** 2)), r):.2f}")

    # ----- Stage 3: 기존 segment corrector를 Stage1+2 잔차에 적용 -----
    stage12_residual = y - stage12_pred
    s_3stage, _ = crossfit_stage3(X_full, stage12_residual, segment, stage12_pred, y, valid_df)
    print(f"  Stage1+2+3(3-stage 전체): {s_3stage:.2f}")

    return s_2stage, s_3stage


def main():
    df = load("train.csv")
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n요약 (2stage, 3stage):")
    print("2023->2024:", r1)
    print("2022->2023:", r2)


if __name__ == "__main__":
    main()
