"""
모델 다양성 2차: NN / Logistic(Elastic Net) / RandomForest를 CatBoost와 2024 holdout에서 비교.

tree 계열(CatBoost/LightGBM/XGBoost) residual 상관관계가 0.999+로 사실상 동일 모델임을 확인했으니
(model_diversity.py), 이번엔 아키텍처 자체가 다른 모델들을 본다. 각 모델은 자기 방식대로 최선의 설정을
쓴다(NN=nn_baseline.py 확립된 lr/seed, Elastic Net=elastic_net.py 확립된 파이프라인, RF=baseline 44피처).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_catboost as bc  # noqa: E402
import nn_baseline as nnb  # noqa: E402
import elastic_net as en  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402


def score(brier: float, r: float) -> float:
    return max(0.0, 100000 * (1 - brier / (r * (1 - r))))


def main():
    df = load("train.csv")
    train_df, valid_df = bc.time_split(df, 2024)
    y = valid_df[bc.TARGET].to_numpy()
    r = y.mean()

    preds = {}

    # CatBoost (기준)
    cat_model = bc.train_catboost(train_df, valid_df)
    preds["CatBoost"] = cat_model.predict_proba(bc.to_pool(valid_df, with_label=False))[:, 1]

    # NN
    nn_train, nn_valid = nnb.time_split(nnb.add_missing_flags(df), 2024)
    nn_model, nn_prep, _, _ = nnb.train_nn(nn_train, nn_valid)
    nn_num, nn_cat = nn_prep.transform(nn_valid)
    import torch
    nn_model.eval()
    with torch.no_grad():
        num_t = torch.tensor(nn_num, dtype=torch.float32).to(nnb.DEVICE)
        cat_t = torch.tensor(nn_cat, dtype=torch.long).to(nnb.DEVICE)
        preds["NN"] = torch.sigmoid(nn_model(num_t, cat_t)).cpu().numpy()

    # Logistic (Elastic Net)
    en_df = en.prepare(df)
    en_train, en_valid = en.time_split(en_df, 2024)
    en_pipe = en.train(en_train)
    preds["Logistic"] = en_pipe.predict_proba(en_valid[en.NUMERIC_FEATURES + en.CATEGORICAL_FEATURES + en.MISSING_FLAGS])[:, 1]

    # RandomForest (baseline 44피처, categorical은 factorize)
    from sklearn.ensemble import RandomForestClassifier
    rf_train, rf_valid = train_df.copy(), valid_df.copy()
    rf_features = list(bc.FEATURES)
    for c in bc.CAT_FEATURES:
        codes, uniques = pd.factorize(pd.concat([rf_train[c], rf_valid[c]]).astype(str))
        rf_train[c] = codes[:len(rf_train)]
        rf_valid[c] = codes[len(rf_train):]
    for c in rf_features:
        rf_train[c] = rf_train[c].fillna(-1)
        rf_valid[c] = rf_valid[c].fillna(-1)
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(rf_train[rf_features], rf_train[bc.TARGET])
    preds["RandomForest"] = rf.predict_proba(rf_valid[rf_features])[:, 1]

    print("\n=== standalone 성능 ===")
    for name, p in preds.items():
        b = float(np.mean((p - y) ** 2))
        print(f"{name}: brier={b:.6f} score={score(b, r):.2f}")

    print("\n=== residual(y-p) 상관관계 vs CatBoost (2024 holdout, 전부 학습에 안 쓴 진짜 OOS) ===")
    resid_cat = y - preds["CatBoost"]
    for name in ["NN", "Logistic", "RandomForest"]:
        resid = y - preds[name]
        corr = np.corrcoef(resid_cat, resid)[0, 1]
        print(f"corr(CatBoost, {name}) = {corr:.4f}")


if __name__ == "__main__":
    main()
