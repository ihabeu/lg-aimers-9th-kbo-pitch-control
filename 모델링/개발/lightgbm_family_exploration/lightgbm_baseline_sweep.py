"""
"다른 모델 패밀리로 처음부터" 탐색 트랙 1단계: LightGBM baseline을 CatBoost baseline과 동등하게
제대로 튜닝해서 만든다 (기존 multimodel_base_ensemble.py의 LightGBM은 파라미터 하나만 대충 골라
썼던 것 -- 이번엔 CatBoost의 l2_leaf_reg 스윕과 같은 방식으로 실제 탐색한다).

같은 44피처, 같은 single-split(2019-23->24) 기준. CAT_FEATURES는 LightGBM 네이티브 categorical
지원(pandas category dtype)으로 그대로 넘긴다 -- one-hot 안 함(CatBoost와 동일 원칙, 트리 모델이라
필요 없음).

champion 교체/실LB 제출과 무관한 순수 탐색용. 이 스크립트는 base 모델 하나만 비교한다
(corrector는 lightgbm_segment_corrector.py에서 별도로 얹어본다).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from eda import load  # noqa: E402
from baseline_catboost import FEATURES, CAT_FEATURES, time_split  # noqa: E402

GRID = {
    "num_leaves": [15, 31, 63],
    "min_child_samples": [200, 1000],
    "reg_lambda": [1.0, 10.0, 50.0],
}
FIXED = dict(n_estimators=3000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
             random_state=42, n_jobs=-1, verbosity=-1)


def to_lgbm_frame(df):
    x = df[FEATURES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype("string").fillna("<NA>").astype("category")
    return x


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def score(brier_val, r):
    return max(0.0, 100000 * (1 - brier_val / (r * (1 - r))))


def fit_eval(X_tr, y_tr, X_va, y_va, **params):
    model = LGBMClassifier(**FIXED, **params)
    model.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="binary_logloss",
        categorical_feature=CAT_FEATURES,
        callbacks=[__import__("lightgbm").early_stopping(100, verbose=False)],
    )
    p = model.predict_proba(X_va)[:, 1]
    r = float(y_va.mean())
    return score(brier(y_va.to_numpy(), p), r), model.best_iteration_


def main():
    df = load("train.csv")
    train_df, valid_df = time_split(df, 2024)
    X_tr, X_va = to_lgbm_frame(train_df), to_lgbm_frame(valid_df)
    y_tr, y_va = train_df["control_success"], valid_df["control_success"]

    results = []
    for nl in GRID["num_leaves"]:
        for mcs in GRID["min_child_samples"]:
            for rl in GRID["reg_lambda"]:
                s, best_iter = fit_eval(X_tr, y_tr, X_va, y_va, num_leaves=nl, min_child_samples=mcs, reg_lambda=rl)
                results.append((nl, mcs, rl, s, best_iter))
                print(f"num_leaves={nl:3d} min_child_samples={mcs:5d} reg_lambda={rl:5.1f}  score={s:.2f}  best_iter={best_iter}")

    results.sort(key=lambda r: -r[3])
    print("\n===== 상위 5개 =====")
    for nl, mcs, rl, s, bi in results[:5]:
        print(f"  num_leaves={nl} min_child_samples={mcs} reg_lambda={rl}  score={s:.2f}  best_iter={bi}")
    print(f"\n(비교 기준: CatBoost baseline single-split 734.49)")


if __name__ == "__main__":
    main()
