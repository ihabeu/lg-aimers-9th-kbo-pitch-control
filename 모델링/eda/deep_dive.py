"""
EDA.md를 더 상세하게 확장하기 위한 3가지 심층 분석. 아이디어(ablation z-검정, 분산 상한 분해,
permutation 교호작용 검정)는 외부 참고 노트북에서 얻었지만, 구현/수치는 전부 우리 데이터로
CatBoost 기반으로 독립적으로 재작성했다(원본 코드/숫자는 안 씀).

CLI: python deep_dive.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402

VALID_SEASON = 2024


def score(brier, r):
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def fit_predict(train_df, valid_df, features, cat_features):
    pool_tr = Pool(train_df[features], train_df["control_success"], cat_features=cat_features)
    pool_va = Pool(valid_df[features], valid_df["control_success"], cat_features=cat_features)
    model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
                                eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
                                random_seed=42, thread_count=-1, verbose=False)
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
    return model.predict_proba(pool_va)[:, 1]


def pitcher_bootstrap_z(d, pitcher_ids, n_boot=500, seed=42):
    """d: per-row (ablated_sqerr - full_sqerr). 양수면 ablation이 손해.
    z>0 이고 클수록 그 그룹을 빼는 게 유의하게 나쁘다(=그 그룹이 중요하다)는 뜻."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(pitcher_ids)))
    idx_by_pitcher = {p: np.where(pitcher_ids == p)[0] for p in uniq}
    means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_pitcher[p] for p in sample])
        means[b] = d[idx].mean()
    se = means.std(ddof=1)
    z = d.mean() / se if se > 0 else np.nan
    return float(d.mean()), float(z)


# 피처를 의미 단위로 묶은 그룹 (전부 이미 44피처 안에 있는 것들의 재분류)
FEATURE_GROUPS = {
    "투수 통산 성공/반대/가운데 (career core)": [
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    ],
    "투수 볼/스트라이크 비율": ["asof_pitcher_ball_rate", "asof_pitcher_strike_rate"],
    "투수 최근 폼(1/3/5경기)": [
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    ],
    "투수 구종 성향(fastball/breaking/offspeed)": [
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    ],
    "타자 이력": ["asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate"],
    "표본수(asof_pitcher_n)": ["asof_pitcher_n"],
    "카운트/아웃/주자(순간 상황)": [
        "balls_before", "strikes_before", "outs_before", "num_runners_on",
        "runner_on_1b", "runner_on_2b", "runner_on_3b", "base_state",
    ],
    "점수/승부 중요도(score+leverage)": [
        "run_top_before", "run_bot_before", "run_total_before",
        "score_diff_home", "score_diff_pitcher_team", "home_win_expectancy", "away_win_expectancy", "li",
    ],
    "투수/타자 손, 팀 ID": ["pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"],
    "시간(season/month/dayofweek/inning/top_bottom)": [
        "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    ],
    "game_type": ["game_type"],
}


def run_ablation(df):
    print("\n===== 1. Ablation z-검정 (그룹 제거 시 손해, pitcher-cluster bootstrap z) =====")
    train_df = df[df["season"] < VALID_SEASON]
    valid_df = df[df["season"] == VALID_SEASON].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)
    r = float(y.mean())
    pitcher_ids = valid_df["pitcher_id"].to_numpy()

    p_full = fit_predict(train_df, valid_df, FEATURES, CAT_FEATURES)
    full_score = score(float(np.mean((p_full - y) ** 2)), r)
    sqerr_full = (p_full - y) ** 2
    print(f"  전체 44피처: {full_score:.2f}")

    rows = []
    for name, cols in FEATURE_GROUPS.items():
        feats = [c for c in FEATURES if c not in cols]
        cats = [c for c in CAT_FEATURES if c not in cols]
        p_ab = fit_predict(train_df, valid_df, feats, cats)
        ab_score = score(float(np.mean((p_ab - y) ** 2)), r)
        sqerr_ab = (p_ab - y) ** 2
        d = sqerr_ab - sqerr_full
        mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
        rows.append({"그룹": name, "제거후 score": round(ab_score, 2),
                      "score 손해": round(full_score - ab_score, 2), "z": round(z, 2)})
        print(f"  -{name}: score={ab_score:.2f} (손해 {full_score - ab_score:+.2f}, z={z:.2f})")

    result = pd.DataFrame(rows).sort_values("score 손해", ascending=False)
    result.to_csv(Path(__file__).resolve().parent / "eda_outputs" / "deep_dive_ablation.csv", index=False)
    return result


def anova_variance_component(df, group_cols):
    """1-way random effects ANOVA method-of-moments 분산성분 추정.
    그룹별 진짜 성공률의 분산(표본 노이즈 제거)을 unbiased하게 추정."""
    g = df.groupby(group_cols, observed=True)["control_success"]
    n_g = g.size().to_numpy(dtype=float)
    mean_g = g.mean().to_numpy(dtype=float)
    N = n_g.sum()
    k = len(n_g)
    grand_mean = (n_g * mean_g).sum() / N

    ss_between = (n_g * (mean_g - grand_mean) ** 2).sum()
    group_mean_per_row = g.transform("mean")
    ss_within = ((df["control_success"] - group_mean_per_row) ** 2).sum()
    msb = ss_between / (k - 1)
    msw = ss_within / (N - k)
    n0 = (N - (n_g ** 2).sum() / N) / (k - 1)
    sigma2_between = max(0.0, (msb - msw) / n0)
    return sigma2_between, k, N


def run_variance_ceiling(df):
    print("\n===== 2. 분산 상한 분해 (완벽한 모델의 brier 이론적 하한) =====")
    R = df[df["game_type"] == "R"].copy()
    r = float(R["control_success"].mean())
    total_var = r * (1 - r)
    print(f"  1군(R) 행 {len(R):,}개, base rate {r:.5f}, 전체 분산 r(1-r)={total_var:.6f}")
    print("  완벽한 모델의 brier = r(1-r) - Var(그룹별 진짜 성공률). 그룹을 세밀하게 쪼갤수록 상한(=낮출 수 있는 brier)이 커진다.")

    groupings = {
        "투수(통산)": ["pitcher_id"],
        "투수 x 손 매치업": ["pitcher_id", "batter_hand"],
        "투수 x 타자": ["pitcher_id", "batter_id"],
    }
    rows = []
    for name, cols in groupings.items():
        sigma2, k, N = anova_variance_component(R, cols)
        ceiling_brier = max(0.0, total_var - sigma2)
        ceiling_score = score(ceiling_brier, r)
        rows.append({"기준": name, "그룹수": k, "진짜분산(추정)": round(sigma2, 6),
                      "상한 brier": round(ceiling_brier, 6), "상한 score": round(ceiling_score, 2)})
        print(f"  {name} (그룹 {k}개): 진짜분산={sigma2:.6f}  상한 brier={ceiling_brier:.6f}  상한 score={ceiling_score:.2f}")

    result = pd.DataFrame(rows)
    result.to_csv(Path(__file__).resolve().parent / "eda_outputs" / "deep_dive_variance_ceiling.csv", index=False)
    return result


def interaction_strength(df, col_a, bins_a, col_b, bins_b, target="control_success"):
    a = bins_a(df[col_a]) if callable(bins_a) else df[col_a]
    b = bins_b(df[col_b]) if callable(bins_b) else df[col_b]
    y = df[target].to_numpy(dtype=float)
    d = pd.DataFrame({"a": a, "b": b, "y": y})
    grand = d["y"].mean()
    row_mean = d.groupby("a")["y"].transform("mean")
    col_mean = d.groupby("b")["y"].transform("mean")
    additive_pred = row_mean + col_mean - grand
    cell_mean = d.groupby(["a", "b"])["y"].transform("mean")
    n = len(d)
    interaction_ss = ((cell_mean - additive_pred) ** 2).sum() / n
    return interaction_ss, d


def permutation_test(df, col_a, bins_a, col_b, bins_b, n_perm=200, seed=42):
    obs, d = interaction_strength(df, col_a, bins_a, col_b, bins_b)
    rng = np.random.default_rng(seed)
    b_vals = d["b"].to_numpy()
    null = np.empty(n_perm)
    grand = d["y"].mean()
    for i in range(n_perm):
        perm_b = rng.permutation(b_vals)
        dd = pd.DataFrame({"a": d["a"].to_numpy(), "b": perm_b, "y": d["y"].to_numpy()})
        row_mean = dd.groupby("a")["y"].transform("mean")
        col_mean = dd.groupby("b")["y"].transform("mean")
        additive_pred = row_mean + col_mean - grand
        cell_mean = dd.groupby(["a", "b"])["y"].transform("mean")
        null[i] = ((cell_mean - additive_pred) ** 2).sum() / len(dd)
    p_value = float((null >= obs).mean())
    return float(obs), p_value, float(null.mean()), float(null.std())


def run_interaction_tests(df):
    print("\n===== 3. Permutation 기반 교호작용 유의성 검정 =====")
    R = df[df["game_type"] == "R"].copy()

    def qbin(s, q=5):
        return pd.qcut(s.rank(method="first"), q, labels=False)

    tests = [
        ("pitcher_hand", None, "batter_hand", None, "투수손 x 타자손"),
        ("season", None, "asof_pitcher_success_rate", lambda s: qbin(s, 5), "season x 투수 통산성공률(5분위)"),
        ("game_type", None, "season", None, "game_type x season"),
    ]
    rows = []
    for col_a, bins_a, col_b, bins_b, label in tests:
        obs, p, null_mean, null_sd = permutation_test(R, col_a, bins_a, col_b, bins_b)
        rows.append({"교호작용": label, "관측 interaction_ss": round(obs, 6),
                      "null 평균": round(null_mean, 6), "null sd": round(null_sd, 6), "p-value": round(p, 4)})
        print(f"  {label}: 관측={obs:.6f}  null평균={null_mean:.6f}±{null_sd:.6f}  p={p:.4f}")

    result = pd.DataFrame(rows)
    result.to_csv(Path(__file__).resolve().parent / "eda_outputs" / "deep_dive_interactions.csv", index=False)
    return result


def main():
    df = load("train.csv")
    run_ablation(df)
    run_variance_ceiling(df)
    run_interaction_tests(df)


if __name__ == "__main__":
    main()
