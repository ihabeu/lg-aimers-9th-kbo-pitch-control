"""
"선수 과거 능력 × 현재 상황" 조건부 피처. corrector importance에서 batter_hand/pitcher_hand/
strikes_before/game_month이 상위였던 것에 근거 -- 단순 hand_matchup(같은 손인지 아닌지 플래그, 이미
로컬은 좋았는데 실제 LB 하락 전례 있음)이 아니라, "이 투수가 좌타자 상대로는 실제 성공률이 얼마였는지"
같은 EB-smoothed 조건부 historical rate를 만든다.

leak-safe: target_season 이전 데이터로만 계산(2024 검증이면 <=2023 누적).
EB smoothing: (성공 + k*prior) / (n + k) 로 표본 적을 때 league prior 쪽으로 당김 (k=count 계열
regularization, "실력 × 표본 신뢰도"를 하나로 합치는 표준 통계 기법).

corrector 입력 피처로 추가(base_pred 실험처럼 base가 아니라 corrector에 넣는다 -- 이 세션에서
"구조/corrector 변경은 통하고 base feature 추가는 자주 실패"했던 패턴을 따름).
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
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
EB_K = 50.0  # 조건부 표본이 50건 정도일 때 절반씩 반영되는 smoothing 강도


def assign_segment(df):
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


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


def eb_conditional_rate(history: pd.DataFrame, keys: list, prior: pd.Series, k: float) -> pd.DataFrame:
    g = history.groupby(keys)["control_success"].agg(["size", "mean"]).reset_index()
    g = g.rename(columns={"size": "_n", "mean": "_raw"})
    return g


def apply_conditional_rate(rows: pd.DataFrame, table: pd.DataFrame, keys: list, league_prior: float, k: float, name: str) -> pd.Series:
    merged = rows[keys].merge(table, on=keys, how="left")
    n = merged["_n"].fillna(0.0).to_numpy()
    raw = merged["_raw"].fillna(league_prior).to_numpy()
    smoothed = (raw * n + league_prior * k) / (n + k)
    return pd.Series(smoothed.astype(np.float32), index=rows.index, name=name)


def corrector_matrix(df, extra: pd.DataFrame = None):
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype("string").fillna("<NA>").astype("category").cat.codes.astype(np.float32)
    x = x.apply(pd.to_numeric, errors="coerce")
    if extra is not None:
        x = pd.concat([x.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    return x


def pitcher_half(frame, seed):
    pitchers = np.array(sorted(frame["pitcher_id"].astype(str).unique()))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(pitchers)
    first_half = set(pitchers[: len(pitchers) // 2])
    return np.where(frame["pitcher_id"].astype(str).isin(first_half).to_numpy(), 0, 1)


def crossfit_score(X, residual, segment, base_pred, y, frame):
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
    return score(float(np.mean((consensus - y) ** 2)), float(y.mean()))


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)
    segment = assign_segment(valid_df)

    base_model = fit_base(train_df, valid_df)
    base_pred = base_model.predict_proba(Pool(valid_df[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    residual = y - base_pred
    league_prior = float(train_df["control_success"].mean())

    # 투수 x 타자손 조건부 성공률 (leak-safe: train_df<target_season 까지만)
    t1 = eb_conditional_rate(train_df, ["pitcher_id", "batter_hand"], None, EB_K)
    f1 = apply_conditional_rate(valid_df, t1, ["pitcher_id", "batter_hand"], league_prior, EB_K, "pitcher_x_batterhand_rate")
    # 투수 x 카운트(볼-스트라이크) 조건부 성공률
    train_df2 = train_df.assign(count_state=train_df["balls_before"].astype(str) + "_" + train_df["strikes_before"].astype(str))
    valid_df2 = valid_df.assign(count_state=valid_df["balls_before"].astype(str) + "_" + valid_df["strikes_before"].astype(str))
    t2 = eb_conditional_rate(train_df2, ["pitcher_id", "count_state"], None, EB_K)
    f2 = apply_conditional_rate(valid_df2, t2, ["pitcher_id", "count_state"], league_prior, EB_K, "pitcher_x_count_rate")
    # 투수 x 월 조건부 성공률
    t3 = eb_conditional_rate(train_df, ["pitcher_id", "game_month"], None, EB_K)
    f3 = apply_conditional_rate(valid_df, t3, ["pitcher_id", "game_month"], league_prior, EB_K, "pitcher_x_month_rate")

    extra = pd.concat([f1, f2, f3], axis=1)

    X_without = corrector_matrix(valid_df)
    X_with = corrector_matrix(valid_df, extra)

    s_without = crossfit_score(X_without, residual, segment, base_pred, y, valid_df)
    s_with = crossfit_score(X_with, residual, segment, base_pred, y, valid_df)
    print(f"  조건부 피처 없음(현재 채택): {s_without:.2f}")
    print(f"  조건부 피처 추가:          {s_with:.2f} (차이 {s_with - s_without:+.2f})")
    return s_without, s_with


def main():
    df = load("train.csv")
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n요약 (without, with):")
    print("2023->2024:", r1)
    print("2022->2023:", r2)


if __name__ == "__main__":
    main()
