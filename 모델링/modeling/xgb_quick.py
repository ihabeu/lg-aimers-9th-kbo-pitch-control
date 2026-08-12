import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import TRAIN_TEST_NUMERIC, TRAIN_TEST_BINARY, TRAIN_TEST_CATEGORICAL, TARGET, load  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

TIME_SERIES_NUMERIC = ["season", "inning"]
TIME_SERIES_CATEGORICAL = ["game_month", "game_dayofweek"]
FEATURES = TRAIN_TEST_NUMERIC + TIME_SERIES_NUMERIC + TRAIN_TEST_BINARY + TRAIN_TEST_CATEGORICAL + TIME_SERIES_CATEGORICAL
CAT_FEATURES = TRAIN_TEST_BINARY + TRAIN_TEST_CATEGORICAL + TIME_SERIES_CATEGORICAL


def to_X(df, cat_levels):
    X = df[FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c].astype(str), categories=cat_levels[c])
    return X


def brier(y, p):
    return float(np.mean((p - y) ** 2))


df = load("train.csv")
cat_levels = {c: sorted(df[c].astype(str).unique()) for c in CAT_FEATURES}
train_df = df[df["season"] < 2024]
valid_df = df[df["season"] == 2024]

Xtr, ytr = to_X(train_df, cat_levels), train_df[TARGET]
Xva, yva = to_X(valid_df, cat_levels), valid_df[TARGET]

clf = XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=6,
    reg_lambda=15.0, enable_categorical=True, tree_method="hist",
    eval_metric="logloss", early_stopping_rounds=50, random_state=42, n_jobs=-1,
)
clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

p = clf.predict_proba(Xva)[:, 1]
y = yva.to_numpy()
r = y.mean()
b = brier(y, p)
score = max(0.0, 100000 * (1 - b / (r * (1 - r))))
best_it = clf.best_iteration + 1
print(f"valid brier={b:.6f} score={score:.2f} best_iteration={best_it}")

# 전체 데이터로 최종 재학습 (2024 포함)
Xall, yall = to_X(df, cat_levels), df[TARGET]
final = XGBClassifier(
    n_estimators=best_it, learning_rate=0.05, max_depth=6,
    reg_lambda=15.0, enable_categorical=True, tree_method="hist",
    random_state=42, n_jobs=-1,
)
final.fit(Xall, yall)
final.save_model(str(MODEL_DIR / "xgb_baseline.json"))
print("saved", MODEL_DIR / "xgb_baseline.json")
print("SCORE_FOR_DECISION", score)
