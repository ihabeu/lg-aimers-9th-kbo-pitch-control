"""
같은 champion 아키텍처를 시드만 바꿔 여러 개 학습해서 평균(bagging)하면 분산이 줄어드는지 확인.
오늘(2026-08-13) 멀티모델 블렌드(CatBoost+LightGBM+XGBoost)가 로컬은 이겼는데 실LB에서 진 이유로
"다른 모델 패밀리 자체의 미래 일반화가 약함"을 지목했었다 -- 이 실험은 그 가설이 맞다면 리스크가
낮을 것으로 예상되는 대안이다: 다른 패밀리를 섞는 게 아니라 같은 CatBoost를 시드만 바꿔서 평균하므로
일반화 성격 자체는 바뀌지 않고 순수하게 분산만 줄어든다.

corrector(segment 3-way, ExtraTrees)는 기존과 동일 구조를 그대로 재사용 -- base_pred만
N-시드 평균으로 바꾼 뒤 residual을 다시 계산해서 corrector를 새로 학습한다.
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
CORRECTOR_SEEDS = (42, 2026, 314)
SEGMENTS = ["core", "hybrid", "dev"]
BASE_SEED_SETS = {
    "단일시드(기존 champion)": (42,),
    "3시드 평균": (42, 2026, 314),
    "5시드 평균": (42, 2026, 314, 7, 123),
}


def assign_segment(df):
    involves_hybrid = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    return np.where(df["game_type"] == "F", "dev", np.where(involves_hybrid, "hybrid", "core"))


def brier_score(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def score(brier, r):
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def fit_base(train_df, valid_df, seed):
    train_pool = Pool(train_df[FEATURES], train_df["control_success"], cat_features=CAT_FEATURES)
    valid_pool = Pool(valid_df[FEATURES], valid_df["control_success"], cat_features=CAT_FEATURES)
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
        eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
        random_seed=seed, thread_count=-1, verbose=False,
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


def apply_corrector(X, residual, segment, base_pred, y, frame):
    seed_preds = []
    for seed in CORRECTOR_SEEDS:
        fold = pitcher_half(frame, seed)
        correction = np.zeros(len(frame))
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
    return score(brier_score(y, consensus), y.mean())


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)
    r = float(y.mean())
    segment = assign_segment(valid_df)
    X = corrector_matrix(valid_df)

    all_seeds_needed = sorted({s for seeds in BASE_SEED_SETS.values() for s in seeds})
    preds_by_seed = {}
    for seed in all_seeds_needed:
        p = fit_base(train_df, valid_df, seed)
        preds_by_seed[seed] = p
        print(f"  seed={seed} base BSS: {score(brier_score(y, p), r):.2f}")

    results = {}
    for name, seeds in BASE_SEED_SETS.items():
        base_pred = np.mean(np.column_stack([preds_by_seed[s] for s in seeds]), axis=1)
        base_score = score(brier_score(y, base_pred), r)
        residual = y - base_pred
        corrected_score = apply_corrector(X, residual, segment, base_pred, y, valid_df)
        results[name] = (base_score, corrected_score)
        print(f"  [{name}] base={base_score:.2f}  +corrector={corrected_score:.2f}")
    return results


def main():
    df = load("train.csv")
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 두 폴드 동시 비교 (base -> +corrector) =====")
    for name in BASE_SEED_SETS:
        print(f"  {name}: primary {r1[name][0]:.2f}->{r1[name][1]:.2f}  stress {r2[name][0]:.2f}->{r2[name][1]:.2f}")


if __name__ == "__main__":
    main()
