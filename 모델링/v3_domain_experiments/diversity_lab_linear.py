"""
"Diversity Lab" 1순위: Ridge/Logistic/ElasticNet이 CatBoost와 얼마나 다른 실수를 하는지
(residual correlation) 확인. 정확도 순위만 보면 안 되고, 상관관계가 낮으면서 BSS가 어느 정도
받쳐주면 블렌드 후보가 될 수 있다는 가설 검증 — model_diversity 실험(CatBoost/LightGBM/XGBoost
상관 0.83~0.96)의 연장선. 선형모델은 트리와 완전히 다른 귀납적 편향이라 상관관계가 더 낮을 걸로
기대됨.

modeling/elastic_net.py의 기존 전처리 파이프라인(build_pipeline/train/evaluate/prepare)을 그대로
재사용 — penalty만 바꿔서 Ridge(l2)/Logistic(규제없음)/ElasticNet 세 버전.
"""
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402
from elastic_net import prepare, time_split, train, evaluate, NUMERIC_FEATURES, CATEGORICAL_FEATURES, MISSING_FLAGS  # noqa: E402

VARIANTS = {
    "Ridge(l2)": {"penalty": "l2", "l1_ratio": None},
    "Logistic(규제없음)": {"penalty": None, "l1_ratio": None},
    "ElasticNet": {"penalty": "elasticnet", "l1_ratio": 0.5},
}


def fit_catboost(train_df, valid_df):
    pool_tr = Pool(train_df[FEATURES], train_df["control_success"], cat_features=CAT_FEATURES)
    pool_va = Pool(valid_df[FEATURES], valid_df["control_success"], cat_features=CAT_FEATURES)
    model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, loss_function="Logloss",
                                eval_metric="BrierScore", l2_leaf_reg=L2_LEAF_REG, early_stopping_rounds=100,
                                random_seed=42, thread_count=-1, verbose=False)
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
    return model.predict_proba(pool_va)[:, 1]


def run_fold(df, valid_season, label):
    print(f"\n===== {label}: <{valid_season} -> {valid_season} =====")
    train_df, valid_df = time_split(df, valid_season)
    p_cat = fit_catboost(train_df, valid_df)

    X_va = valid_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + MISSING_FLAGS]
    for name, params in VARIANTS.items():
        pipe = train(train_df, penalty=params["penalty"], l1_ratio=params["l1_ratio"])
        m = evaluate(pipe, valid_df)
        p_lin = pipe.predict_proba(X_va)[:, 1]
        corr = np.corrcoef(p_cat, p_lin)[0, 1]
        print(f"  {name}: BSS={m['score']:.2f}  corr(vs CatBoost)={corr:.4f}")


def main():
    df = prepare(load("train.csv"))
    run_fold(df, 2024, "PRIMARY")
    run_fold(df, 2023, "STRESS")


if __name__ == "__main__":
    main()
