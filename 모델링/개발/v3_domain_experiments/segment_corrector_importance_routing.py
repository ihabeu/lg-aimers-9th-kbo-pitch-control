"""
E021(hand match 라우팅)이 기각된 근본 원인은 pitcher_hand/batter_hand가 CatBoost feature importance
최하위권(각 0.0021, 44개 중 40위권 밖)이었다는 것 -- 애초에 신호가 약한 축이었다.

이번엔 반대로 "game_type처럼 중요도가 실제로 높은 변수"를 라우팅 축으로 쓴다
(eda/eda_outputs/train_feature_importance.csv 기준):

- asof_pitcher_success_rate (중요도 1위, 0.1247) -- 투수 커리어 제구 수준
- asof_batter_success_rate  (중요도 5위, 0.0671) -- 타자 상대 제구 허용 수준
- li                        (중요도 15위, 0.0138) -- 레버리지(경기 중요도)

각각을 기존 3-way(core/hybrid/dev) 안에서 중앙값 기준 2분할해 6-way로 확장(E021과 동일한 구조 --
hand 대신 축만 교체, 비교 가능하게). 판정도 E021과 동일하게 pitcher-cluster bootstrap z-검정.
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
AXES = ["asof_pitcher_success_rate", "asof_batter_success_rate", "li"]


def assign_segment_3way(df):
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


def assign_segment_6way(df, axis, cutoff):
    base = assign_segment_3way(df)
    high = (df[axis] >= cutoff).to_numpy()
    tag = np.where(high, "_hi", "_lo")
    return np.array([b + t for b, t in zip(base, tag)])


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
    print(f"  base={base_score:.2f}  3-way(champion)={score3:.2f}")

    pitcher_ids = valid_df["pitcher_id"].to_numpy()
    results = {}
    for axis in AXES:
        cutoff = float(train_df[axis].median())  # train 기준 중앙값 -- leak-safe
        seg6 = assign_segment_6way(valid_df, axis, cutoff)
        seg6_list = sorted(set(seg6))
        sizes = pd.Series(seg6).value_counts().to_dict()
        score6, pred6 = apply_corrector(X, residual, seg6, seg6_list, base_pred, y, valid_df)
        d = (pred3 - y) ** 2 - (pred6 - y) ** 2
        mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
        print(f"  [{axis}] 6-way={score6:.2f}  차이={score6 - score3:+.2f}  z={z:.2f}  세그먼트크기={sizes}")
        results[axis] = (score3, score6, z)
    return results


def main():
    df = load("train.csv")
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 요약 (3-way vs 축별 6-way) =====")
    for axis in AXES:
        p3, p6, pz = r1[axis]
        s3, s6, sz = r2[axis]
        print(f"  {axis}: PRIMARY {p3:.2f}->{p6:.2f}({p6-p3:+.2f}, z={pz:.2f})  STRESS {s3:.2f}->{s6:.2f}({s6-s3:+.2f}, z={sz:.2f})")


if __name__ == "__main__":
    main()
