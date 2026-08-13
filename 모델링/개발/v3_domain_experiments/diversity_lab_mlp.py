"""
"Diversity Lab" Tier 2: MLP. Ridge/Logistic/ElasticNet(E012)은 선형이라 game_type×season 반전
교호작용을 표현 못 해서 stress에서 BSS=0으로 무너졌다. MLP는 은닉층이 있어서 이런 교호작용을
구조적으로 표현할 수 있다 -- 같은 전처리 파이프라인에서 classifier만 바꿔서 stress가 살아나는지,
그러면서 CatBoost와 residual correlation은 얼마나 낮은지 확인.
"""
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, L2_LEAF_REG  # noqa: E402
from elastic_net import prepare, time_split, evaluate, NUMERIC_FEATURES, CATEGORICAL_FEATURES, MISSING_FLAGS  # noqa: E402

HIDDEN_LAYER_SIZES = (64, 32)


def build_mlp_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_FEATURES + MISSING_FLAGS),
    ])
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation="relu", alpha=1e-4,
                         batch_size=2048, learning_rate_init=1e-3, max_iter=100,
                         early_stopping=True, n_iter_no_change=5, random_state=42)
    return Pipeline([("prep", preprocess), ("clf", clf)])


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

    X_tr = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + MISSING_FLAGS]
    y_tr = train_df["control_success"]
    X_va = valid_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + MISSING_FLAGS]

    pipe = build_mlp_pipeline()
    pipe.fit(X_tr, y_tr)
    m = evaluate(pipe, valid_df)
    p_mlp = pipe.predict_proba(X_va)[:, 1]
    corr = float(np.corrcoef(p_cat, p_mlp)[0, 1])
    print(f"  MLP{HIDDEN_LAYER_SIZES}: BSS={m['score']:.2f}  AUC={m['auc']}  corr(vs CatBoost)={corr:.4f}  "
          f"iters={pipe.named_steps['clf'].n_iter_}")
    return m["score"], corr


def main():
    df = prepare(load("train.csv"))
    run_fold(df, 2024, "PRIMARY")
    run_fold(df, 2023, "STRESS")


if __name__ == "__main__":
    main()
