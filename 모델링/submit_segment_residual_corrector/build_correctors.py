"""
789.23 champion CatBoost(model/catboost_baseline.cbm, 이미 2019~2024 전체로 학습됨, 안 건드림)를
base로 그대로 쓰고, game_type(R/F) segment별 잔차 보정기(ExtraTrees)만 오프라인으로 학습해서
model/correctors.joblib에 저장. v3_domain_experiments/segment_residual_corrector.py에서 검증한 것과
같은 구조를 실제 배포용으로 재현(그 스크립트는 cross-fit 검증용, 이건 전체 라벨로 최종 학습).

잔차 학습 신호: "2019~2023으로 학습한 CatBoost가 2024를 예측했을 때 남긴 오차"를 2024 라벨 전체로
학습(비-cross-fit, 최종 배포용). 이 오차 패턴이 2025에도 비슷하게 나타난다고 가정하고, 실제 champion
(2019~2024 전체 학습)의 test.csv 예측에 그대로 적용한다.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402

ROOT = Path(__file__).resolve().parent
CORRECTOR_CFG = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features=0.7)
SEEDS = (42, 2026, 314)
SEGMENTS = ["R", "F"]


def fit_category_maps(df: pd.DataFrame) -> dict:
    """train.csv 전체에서 카테고리 매핑을 한 번 고정한다 -- corrector 학습(2024)과 실제 추론(test)이
    서로 다른 값 집합을 볼 수 있어서, 매번 그 데이터프레임 안에서만 .cat.codes를 새로 매기면 같은
    코드값이 다른 카테고리를 가리키는 정합성 버그가 생긴다."""
    maps = {}
    for c in CAT_FEATURES:
        values = df[c].astype("string").fillna("<NA>")
        maps[c] = {v: i for i, v in enumerate(sorted(values.unique()))}
    return maps


def corrector_matrix(df: pd.DataFrame, category_maps: dict) -> pd.DataFrame:
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        values = x[c].astype("string").fillna("<NA>")
        x[c] = values.map(category_maps[c]).fillna(-1).astype(np.float32)
    return x.apply(pd.to_numeric, errors="coerce")


def main():
    df = load("train.csv")
    residual_source_train = df[df["season"] < 2024]
    residual_target = df[df["season"] == 2024].reset_index(drop=True)
    y_target = residual_target["control_success"].to_numpy(np.float64)

    print("residual-source 모델 학습 (2019~2023 -> 2024 예측용, 배포에는 안 씀)")
    src_pool = Pool(residual_source_train[FEATURES], residual_source_train["control_success"], cat_features=CAT_FEATURES)
    tgt_pool_labeled = Pool(residual_target[FEATURES], residual_target["control_success"], cat_features=CAT_FEATURES)
    residual_source_model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
        eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
        random_seed=42, thread_count=-1, verbose=False,
    )
    residual_source_model.fit(src_pool, eval_set=tgt_pool_labeled, use_best_model=True)
    residual_source_pred = residual_source_model.predict_proba(Pool(residual_target[FEATURES], cat_features=CAT_FEATURES))[:, 1]

    brier = float(np.mean((residual_source_pred - y_target) ** 2))
    r = float(y_target.mean())
    sanity_score = max(0.0, 100000 * (1 - brier / (r * (1 - r))))
    print("residual-source 2024 BSS (sanity, segment_residual_corrector.py 734.49와 같아야 함):", round(sanity_score, 2))

    residual = y_target - residual_source_pred
    category_maps = fit_category_maps(df)
    X = corrector_matrix(residual_target, category_maps)
    game_type = residual_target["game_type"].to_numpy()

    correctors = {}
    for seed in SEEDS:
        for seg in SEGMENTS:
            mask = game_type == seg
            model = ExtraTreesRegressor(n_jobs=-1, random_state=16200 + int(seed), **CORRECTOR_CFG)
            model.fit(X.loc[mask], residual[mask])
            correctors[(seg, seed)] = model
            print(f"  corrector fit segment={seg} seed={seed} n={int(mask.sum())}")

    # sanity: champion(2019~2024 전체 학습) + corrector를 로컬 test.csv(5행)에 적용
    champion = CatBoostClassifier()
    champion.load_model(str(ROOT / "model" / "catboost_baseline.cbm"))
    test = load("test.csv")
    champion_pred = champion.predict_proba(Pool(test[FEATURES], cat_features=CAT_FEATURES))[:, 1]
    X_test = corrector_matrix(test, category_maps)
    test_game_type = test["game_type"].to_numpy()
    correction = np.zeros(len(test))
    for seed in SEEDS:
        seed_corr = np.zeros(len(test))
        for seg in SEGMENTS:
            mask = test_game_type == seg
            if not mask.any():
                continue
            seed_corr[mask] = correctors[(seg, seed)].predict(X_test.loc[mask])
        correction += seed_corr / len(SEEDS)
    final = np.clip(champion_pred + correction, 0, 1)
    print("SANITY (로컬 test.csv 5행) champion base:", champion_pred, "final:", final)

    joblib.dump(
        {"correctors": correctors, "seeds": list(SEEDS), "segments": SEGMENTS, "category_maps": category_maps},
        ROOT / "model" / "correctors.joblib", compress=3,
    )
    print("saved:", ROOT / "model" / "correctors.joblib")


if __name__ == "__main__":
    main()
