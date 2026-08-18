"""
팀 깃헙(iamdbstjd/LGAIMERS)의 E17-A 아이디어를 참고해서 독립적으로 재구현한 실험 (코드/하이퍼파라미터
그대로 재현 아님 -- 대회 규정 §11 원칙에 따라 아이디어만 참고하고 직접 새로 작성/검증).

아이디어: 실패 유형(reverse/middle/outside)을 별도 hazard 서브모델로 예측한 뒤, "예측값 자체"가
아니라 "실패 유형 간 비율(log-ratio)"을 champion의 residual corrector 입력에 메타피처로 추가하면
도움이 될까? -- "모델이 틀렸다/안 틀렸다"가 아니라 "이 상황에서 모델이 예상하는 실패의 '모양'
(가운데로 몰릴지 완전히 벗어날지)"이라는, 지금까지 안 써본 축.

R/M/O 라벨은 이미 이 프로젝트에 있는 `rmo_labels.py`(leak-safe 복원, E002/E003에서 검증됨)를 그대로
쓴다. E003은 이 hazard를 "전체 모델 대체용 innovation 블렌드"로 썼는데 효과가 작고 노이즈성이었다
(+3.14@beta=0.1, rolling 미확인). 이번엔 다른 용도 -- corrector 입력에 메타피처로 추가.

주의할 선례: `segment_corrector_base_pred_feature.py`가 "보조 정보(base_pred)를 corrector 피처에
추가"하는 것과 같은 카테고리 실험이었는데 stress -34.45로 기각됐다. log-ratio가 base_pred와 다른
정보(잔차 크기가 아니라 실패 유형 구성비)이긴 하지만, "보조 모델 출력을 corrector에 얹기"라는 큰
카테고리 자체가 이미 한 번 실패한 적이 있다는 걸 염두에 두고 본다.
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
from rmo_labels import add_rmo_labels  # noqa: E402

HYBRID_TEAM_ID = 13
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
SEEDS = (42, 2026, 314)
SEGMENTS_3WAY = ["core", "hybrid", "dev"]
EPS = 1e-3


def assign_segment_3way(df):
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
    return model.predict_proba(valid_pool)[:, 1]


def fit_hazard_sub(train_df, target_col):
    y = train_df[target_col].to_numpy()
    pool = Pool(train_df[FEATURES], y, cat_features=CAT_FEATURES)
    model = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=6, loss_function="Logloss",
        l2_leaf_reg=L2_LEAF_REG, random_seed=42, thread_count=-1, verbose=False,
    )
    model.fit(pool)
    return model


def rmo_logratio_features(train_df, valid_df):
    """R/M/O hazard 서브모델을 train_df로 학습해서 valid_df에 대한 log-ratio 메타피처 2개를 만든다."""
    rmo_train = train_df.dropna(subset=["reverse_label", "middle_label"])
    qR_model = fit_hazard_sub(rmo_train, "reverse_label")
    not_reverse = rmo_train[rmo_train["reverse_label"] == 0]
    qM_model = fit_hazard_sub(not_reverse, "middle_label")
    not_rm = not_reverse[not_reverse["middle_label"] == 0]
    not_rm = not_rm[not_rm["outside_label"].isin([0, 1])]
    qO_model = fit_hazard_sub(not_rm, "outside_label")

    valid_pool = Pool(valid_df[FEATURES], cat_features=CAT_FEATURES)
    qR = qR_model.predict_proba(valid_pool)[:, 1]
    qM = qM_model.predict_proba(valid_pool)[:, 1]
    qO = qO_model.predict_proba(valid_pool)[:, 1]

    log_ratio_mr = np.log((qM + EPS) / (qR + EPS))
    log_ratio_or = np.log((qO + EPS) / (qR + EPS))
    return log_ratio_mr, log_ratio_or


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


def apply_corrector(X, residual, segment, base_pred, y, frame, corrector_cfg=None):
    corrector_cfg = corrector_cfg or CORRECTOR_CFG
    seed_preds = []
    for seed in SEEDS:
        fold = pitcher_half(frame, seed)
        correction = np.zeros(len(frame))
        for half in (0, 1):
            tr_mask = fold != half
            ev_mask = fold == half
            for seg in SEGMENTS_3WAY:
                tr = tr_mask & (segment == seg)
                ev = ev_mask & (segment == seg)
                if tr.sum() < 500 or ev.sum() < 50:
                    correction[ev] = 0.0
                    continue
                model = ExtraTreesRegressor(n_jobs=-1, random_state=16200 + int(seed), **corrector_cfg)
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

    base_pred = fit_base(train_df, valid_df)
    residual = y - base_pred
    seg3 = assign_segment_3way(valid_df)

    X = corrector_matrix(valid_df)
    score_base, pred_base = apply_corrector(X, residual, seg3, base_pred, y, valid_df)
    print(f"  기존 3-way corrector(메타피처 없음): {score_base:.2f}")

    log_ratio_mr, log_ratio_or = rmo_logratio_features(train_df, valid_df)
    X_meta = X.copy()
    X_meta["rmo_log_ratio_mr"] = log_ratio_mr
    X_meta["rmo_log_ratio_or"] = log_ratio_or
    score_meta, pred_meta = apply_corrector(X_meta, residual, seg3, base_pred, y, valid_df)
    print(f"  + R/M/O log-ratio 메타피처: {score_meta:.2f}  차이={score_meta - score_base:+.2f}")

    d = (pred_base - y) ** 2 - (pred_meta - y) ** 2
    pitcher_ids = valid_df["pitcher_id"].to_numpy()
    mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
    print(f"  pitcher-bootstrap: mean_d={mean_d:.6f}  z={z:.2f}")
    return score_base, score_meta, z


def main():
    df = add_rmo_labels(load("train.csv"))
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 요약 =====")
    print(f"PRIMARY: {r1[0]:.2f} -> {r1[1]:.2f} ({r1[1]-r1[0]:+.2f}, z={r1[2]:.2f})")
    print(f"STRESS:  {r2[0]:.2f} -> {r2[1]:.2f} ({r2[1]-r2[0]:+.2f}, z={r2[2]:.2f})")


if __name__ == "__main__":
    main()
