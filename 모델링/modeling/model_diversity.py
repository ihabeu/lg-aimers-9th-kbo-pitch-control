"""
모델 다양성(OOF residual correlation) 체크: CatBoost / LightGBM / XGBoost를 동일 44피처로 학습해서
2024 holdout(셋 다 학습에 안 쓴 진짜 out-of-sample)에서 residual(y-p) 상관관계를 본다.

목적: "단독 성능이 낮아도 CatBoost와 다르게 틀리면 stacking 가치가 있다"는 가설을 검증하는 첫 단계.
상관관계가 낮은 모델이 있으면 그 모델만 다음 단계(실제 stacking/weighted ensemble)로 가져간다.
Elastic Net(385점)이 blending에서 실패한 건 "blending이 안 된다"가 아니라 "Elastic Net이 좋은
complementary 모델이 아니었다"로 재해석 — 이번엔 트리 계열 다른 라이브러리로 다시 확인.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402


def score(brier: float, r: float) -> float:
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def run_catboost(train_df, valid_df):
    model = bc.train_catboost(train_df, valid_df)
    p = model.predict_proba(bc.to_pool(valid_df, with_label=False))[:, 1]
    return p


def run_lightgbm(train_df, valid_df):
    from lightgbm import LGBMClassifier
    Xtr, Xva = train_df[bc.FEATURES].copy(), valid_df[bc.FEATURES].copy()
    for c in bc.CAT_FEATURES:
        Xtr[c] = Xtr[c].astype(str).astype("category")
        Xva[c] = pd.Categorical(Xva[c].astype(str), categories=Xtr[c].cat.categories)
    clf = LGBMClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, reg_lambda=15.0,
                          verbosity=-1, random_state=42)
    clf.fit(Xtr, train_df[bc.TARGET], eval_set=[(Xva, valid_df[bc.TARGET])],
            categorical_feature=bc.CAT_FEATURES,
            callbacks=[__import__("lightgbm").early_stopping(50, verbose=False)])
    return clf.predict_proba(Xva)[:, 1]


def run_xgboost(train_df, valid_df):
    from xgboost import XGBClassifier
    cat_levels = {c: sorted(train_df[c].astype(str).unique()) for c in bc.CAT_FEATURES}

    def to_x(df):
        X = df[bc.FEATURES].copy()
        for c in bc.CAT_FEATURES:
            X[c] = pd.Categorical(X[c].astype(str), categories=cat_levels[c])
        return X

    Xtr, Xva = to_x(train_df), to_x(valid_df)
    clf = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, reg_lambda=15.0,
                         enable_categorical=True, tree_method="hist", eval_metric="logloss",
                         early_stopping_rounds=50, random_state=42, n_jobs=-1)
    clf.fit(Xtr, train_df[bc.TARGET], eval_set=[(Xva, valid_df[bc.TARGET])], verbose=False)
    return clf.predict_proba(Xva)[:, 1]


def main():
    df = load("train.csv")
    train_df, valid_df = bc.time_split(df, 2024)
    y = valid_df[bc.TARGET].to_numpy()
    r = y.mean()

    preds = {}
    for name, fn in [("CatBoost", run_catboost), ("LightGBM", run_lightgbm), ("XGBoost", run_xgboost)]:
        p = fn(train_df, valid_df)
        b = float(np.mean((p - y) ** 2))
        preds[name] = p
        print(f"{name}: standalone brier={b:.6f} score={score(b, r):.2f}")

    print("\nresidual(y-p) 상관관계 (2024 holdout, 셋 다 학습에 안 쓴 진짜 OOS):")
    names = list(preds.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r1 = y - preds[names[i]]
            r2 = y - preds[names[j]]
            corr = np.corrcoef(r1, r2)[0, 1]
            print(f"  corr({names[i]}, {names[j]}) = {corr:.4f}")


if __name__ == "__main__":
    main()
