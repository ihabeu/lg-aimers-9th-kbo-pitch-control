"""
오늘 밤 EDA 심층분석(eda/deep_dive.py)에서 두 가지가 새로 확인됐다:
1. 투수x타자손 매치업 기준 분산 상한(939)이 현재 champion(879.80)보다 높다 -- 이론적 여지가 있음.
2. 투수손x타자손 교호작용이 permutation 검정으로 강하게 유의하다(p<0.0001).

그런데 hand_matchup을 "피처로 추가"하는 방식(E1, EXPERIMENTS.md 6차 제출)은 로컬은 이겼는데 실LB에서
졌다. 이번엔 다른 구조로 시도한다 -- 피처가 아니라 이미 로컬/실제가 정합했던 성공 패턴(segment
residual corrector, E013)과 같은 방식으로, hand match 여부를 corrector의 "라우팅 축"에 추가한다
(core/hybrid/dev 각각을 same_hand/diff_hand로 한 번 더 나눠 최대 6-way). 표본이 너무 작아지면
안 되므로(4-way F팀분리가 실패한 이유) 각 6개 segment 표본 크기를 반드시 같이 본다.
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


def assign_segment_3way(df):
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


def assign_segment_6way(df):
    base = assign_segment_3way(df)
    same_hand = (df["pitcher_hand"] == df["batter_hand"]).to_numpy()
    hand_tag = np.where(same_hand, "_same", "_diff")
    return np.array([b + h for b, h in zip(base, hand_tag)])


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
    return model.predict_proba(valid_pool)[:, 1]


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


def apply_corrector(X, residual, segment, segment_list, base_pred, y, frame):
    seed_preds = []
    for seed in SEEDS:
        fold = pitcher_half(frame, seed)
        correction = np.zeros(len(frame))
        for half in (0, 1):
            tr_mask = fold != half
            ev_mask = fold == half
            for seg in segment_list:
                tr = tr_mask & (segment == seg)
                ev = ev_mask & (segment == seg)
                if tr.sum() < 500 or ev.sum() < 50:
                    correction[ev] = 0.0
                    continue
                model = ExtraTreesRegressor(n_jobs=-1, random_state=16200 + int(seed), **CORRECTOR_CFG)
                model.fit(X.loc[tr], residual[tr])
                correction[ev] = model.predict(X.loc[ev])
        seed_preds.append(np.clip(base_pred + correction, 0, 1))
    consensus = np.mean(np.column_stack(seed_preds), axis=1)
    return score(brier_score(y, consensus), y.mean()), consensus


def pitcher_bootstrap_z(d, pitcher_ids, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(pitcher_ids)))
    idx_by_pitcher = {p: np.where(pitcher_ids == p)[0] for p in uniq}
    means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_pitcher[p] for p in sample])
        means[b] = d[idx].mean()
    se = means.std(ddof=1)
    return float(d.mean()), float(d.mean() / se) if se > 0 else float("nan")


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)
    r = float(y.mean())

    base_pred = fit_base(train_df, valid_df)
    base_score = score(brier_score(y, base_pred), r)
    X = corrector_matrix(valid_df)
    residual = y - base_pred

    seg3 = assign_segment_3way(valid_df)
    score3, pred3 = apply_corrector(X, residual, seg3, SEGMENTS_3WAY, base_pred, y, valid_df)

    seg6 = assign_segment_6way(valid_df)
    seg6_list = sorted(set(seg6))
    print("  6-way segment 분포:", pd.Series(seg6).value_counts().to_dict())
    score6, pred6 = apply_corrector(X, residual, seg6, seg6_list, base_pred, y, valid_df)

    print(f"  base={base_score:.2f}  3-way(기존 champion)={score3:.2f}  6-way(+hand routing)={score6:.2f}  차이={score6 - score3:+.2f}")

    # 6-way가 3-way보다 나은 게 pitcher-cluster bootstrap으로 유의한지 확인
    d = (pred3 - y) ** 2 - (pred6 - y) ** 2  # 양수면 6-way가 이김
    pitcher_ids = valid_df["pitcher_id"].to_numpy()
    mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
    print(f"  6-way 우위 pitcher-bootstrap: mean_d={mean_d:.6f}  z={z:.2f}")
    return base_score, score3, score6, z


def main():
    df = load("train.csv")
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 두 폴드 동시 비교 (3-way vs 6-way) =====")
    print(f"  PRIMARY: 3-way={r1[1]:.2f}  6-way={r1[2]:.2f}  차이={r1[2]-r1[1]:+.2f}  z={r1[3]:.2f}")
    print(f"  STRESS:  3-way={r2[1]:.2f}  6-way={r2[2]:.2f}  차이={r2[2]-r2[1]:+.2f}  z={r2[3]:.2f}")


if __name__ == "__main__":
    main()
