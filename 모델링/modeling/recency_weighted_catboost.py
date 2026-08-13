"""
Recency-weighted CatBoost.

EDA2(../eda/eda2.ipynb)에서 확인한 것: game_type=F의 성공률이 2019~2022엔 0.59~0.71로 높다가
2023부터 0.46~0.47로 정확히 뒤집힌다. 이런 갑작스런 반전은 사전에 예측할 수 없지만(out-of-distribution),
모델이 오래된 연도의 패턴에 과도하게 의존하지 않도록 최근 연도 표본에 지수적으로 더 큰 가중치를
주면 완화할 수 있다. 참고한 다른 팀(다른 참가자 공개 레포)이 이 방식(E13, exponential season weighting)을
실험해서 λ=0.50을 최적으로 찾았다고 함 — 여기서도 λ=0.5로 시작한다.

피처는 baseline_catboost.py의 "엔지니어링 전" 기본 피처셋(44개)을 그대로 쓴다. season_regime 등
엔지니어링 피처 4개는 이전 실험(rolling OOT)에서 2024 폴드 성능을 오히려 깎아먹어서 뺐다 —
recency weighting 효과만 순수하게 보기 위한 ablation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import TRAIN_TEST_NUMERIC, TRAIN_TEST_BINARY, TRAIN_TEST_CATEGORICAL, TARGET, load  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "models"
L2_LEAF_REG = 15.0
RECENCY_LAMBDA = 0.5  # 다른 참가자 E13 실험 최적값
FIXED_ITERATIONS = 204  # baseline_catboost.py 첫 실험(734.49/789.23)과 동일 조건으로 비교하기 위해 고정

TIME_SERIES_NUMERIC = ["season", "inning"]
TIME_SERIES_CATEGORICAL = ["game_month", "game_dayofweek"]

FEATURES = TRAIN_TEST_NUMERIC + TIME_SERIES_NUMERIC + TRAIN_TEST_BINARY + TRAIN_TEST_CATEGORICAL + TIME_SERIES_CATEGORICAL
CAT_FEATURES = TRAIN_TEST_BINARY + TRAIN_TEST_CATEGORICAL + TIME_SERIES_CATEGORICAL

ROLLING_FOLDS = [(2022, 0.2), (2023, 0.3), (2024, 0.5)]


def time_split(df: pd.DataFrame, valid_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["season"] < valid_season], df[df["season"] == valid_season]


def recency_weights(seasons: pd.Series, lam: float = RECENCY_LAMBDA) -> np.ndarray:
    """가장 최근 연도=가중치 1.0, 오래될수록 exp(-lam * 연도차)로 작아짐."""
    max_season = seasons.max()
    return np.exp(-lam * (max_season - seasons)).to_numpy()


def to_pool(df: pd.DataFrame, with_label: bool = True, use_recency_weight: bool = False) -> Pool:
    X = df[FEATURES]
    y = df[TARGET] if with_label else None
    w = recency_weights(df["season"]) if use_recency_weight else None
    return Pool(X, y, cat_features=CAT_FEATURES, weight=w)


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def evaluate(model: CatBoostClassifier, valid_df: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score

    p = model.predict_proba(to_pool(valid_df, with_label=False))[:, 1]
    y = valid_df[TARGET].to_numpy()
    r = y.mean()
    brier = brier_score(y, p)
    baseline_brier = r * (1 - r)
    score = max(0.0, 100000 * (1 - brier / baseline_brier))
    return {
        "n": len(valid_df), "r": round(r, 4), "brier": round(brier, 6),
        "score": round(score, 2), "auc": round(roc_auc_score(y, p), 4),
    }


def train(train_df: pd.DataFrame, iterations: int, use_recency_weight: bool) -> CatBoostClassifier:
    pool = to_pool(train_df, use_recency_weight=use_recency_weight)
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="BrierScore",
        l2_leaf_reg=L2_LEAF_REG,
        random_seed=42,
        thread_count=-1,
        verbose=False,
    )
    model.fit(pool)
    return model


def rolling_compare(df: pd.DataFrame, iterations: int = FIXED_ITERATIONS, folds=ROLLING_FOLDS) -> pd.DataFrame:
    """폴드마다 recency weighting 있는 버전/없는 버전을 같은 iterations로 학습해서 비교."""
    rows = []
    for valid_season, weight in folds:
        train_df, valid_df = time_split(df, valid_season)
        for use_w in [False, True]:
            model = train(train_df, iterations, use_w)
            m = evaluate(model, valid_df)
            m["valid_season"] = valid_season
            m["fold_weight"] = weight
            m["recency_weighted"] = use_w
            rows.append(m)
    return pd.DataFrame(rows)


def weighted_summary(compare_df: pd.DataFrame) -> pd.DataFrame:
    out = compare_df.groupby("recency_weighted").apply(
        lambda g: pd.Series({
            "weighted_brier": (g["brier"] * g["fold_weight"]).sum(),
            "weighted_score": (g["score"] * g["fold_weight"]).sum(),
        }),
        include_groups=False,
    )
    return out.round(4)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    df = load("train.csv")
    print(f"피처 {len(FEATURES)}개, iterations={FIXED_ITERATIONS} 고정")

    compare_df = rolling_compare(df)
    print("\n[폴드별 비교: recency_weighted False vs True]")
    print(compare_df.to_string(index=False))

    print("\n[가중평균]")
    print(weighted_summary(compare_df))

    final_model = train(df, FIXED_ITERATIONS, use_recency_weight=True)
    model_path = MODEL_DIR / "catboost_recency_weighted.cbm"
    final_model.save_model(str(model_path))
    print(f"\nsaved final (recency-weighted, full data) model to {model_path}")


if __name__ == "__main__":
    main()
