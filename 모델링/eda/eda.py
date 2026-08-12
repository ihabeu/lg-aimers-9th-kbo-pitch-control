"""
lg aimers 데이터 로드 + EDA 라이브러리.

컬럼 분류는 dtype이 아니라 data_description.md 기준 도메인 지식으로 고정한다.
- 식별자: 값 자체에 통계적 의미 없음 (매칭/구분 용도)
- 수치형: 값의 크기가 의미 있는 연속형/카운트형 (스케일링 대상)
- 시계열형: 시간 위치를 나타내는 값 (season, month, dayofweek, inning) — 전처리 방식이 수치형과 다를 수 있어 별도 취급
- 이진형: 값이 정확히 2개뿐인 컬럼
- 범주형(명목형): 순서 없고 값도 3개 이상인 컬럼

`asof_pitcher_pitchmix_n`은 `asof_pitcher_n`과 값이 100% 동일한 완전 중복이라 스키마에서 제외함 (검증: eda.ipynb 4절).

이 모듈은 함수/스키마만 정의한다. 실행 결과와 설명은 eda.ipynb 참고.
CLI로 `python eda.py`를 돌리면 전체 EDA를 텍스트로 출력한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "eda_outputs"
TARGET = "control_success"
IMPORTANCE_SAMPLE_N = 200_000  # RandomForest 중요도 계산용 서브샘플 (147만행 전체는 불필요하게 느림)

# --- train.csv / test.csv 공통 스키마 ---
TRAIN_TEST_ID = ["row_id", "pitcher_id", "batter_id"]

TRAIN_TEST_TIME_SERIES = ["season", "game_month", "game_dayofweek", "inning"]

TRAIN_TEST_NUMERIC = [
    "balls_before", "strikes_before", "outs_before", "num_runners_on",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "home_win_expectancy", "away_win_expectancy", "li",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]

TRAIN_TEST_BINARY = [
    "top_bottom", "game_type", "pitcher_hand", "batter_hand",
    "runner_on_1b", "runner_on_2b", "runner_on_3b",
]

TRAIN_TEST_CATEGORICAL = ["base_state", "pitcher_team_id", "batter_team_id"]

# 타겟 대비 분포를 대표로 보여줄 변수 (변수 중요도 상위권 + 도메인상 의심 지점)
KEY_VARS_FOR_TARGET_PLOT = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_batter_success_rate",
    "season", "game_type", "pitcher_team_id",
]

# --- trackman_history.csv 스키마 (train/test와 별개, 1:1 조인 테이블 아님) — 같은 기준 적용 ---
TRACKMAN_TIME_SERIES = ["season", "game_date", "game_month", "game_dayofweek", "inning", "pitch_no"]
TRACKMAN_NUMERIC = [
    "balls_before", "strikes_before", "outs_before", "pitch_of_pa",
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
]
TRACKMAN_BINARY = ["top_bottom", "pitcher_hand", "batter_hand"]
TRACKMAN_CATEGORICAL = [
    "pitcher_team", "batter_team", "tagged_pitch_type", "auto_pitch_type", "pitch_type_group",
]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def numeric_summary(df: pd.DataFrame, cols: list[str], target: str | None = None) -> pd.DataFrame:
    """수치형 컬럼의 count/mean/std/min/중앙값/max/결측률/왜도. target 있으면 그룹별 평균 + 상관계수도."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    desc = df[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc["missing_ratio"] = df[cols].isna().mean().round(4)
    desc["skew"] = df[cols].skew(numeric_only=True)
    if target and target in df.columns:
        desc["mean_target0"] = df.loc[df[target] == 0, cols].mean()
        desc["mean_target1"] = df.loc[df[target] == 1, cols].mean()
        desc["corr_with_target"] = df[cols].corrwith(df[target]).round(4)
    return desc.round(4)


def categorical_summary(df: pd.DataFrame, cols: list[str], target: str | None = None, top_n: int = 12) -> pd.DataFrame:
    """이진형/범주형/시계열형 공용. 값별 건수, target 있으면 값별 성공률도."""
    cols = [c for c in cols if c in df.columns]
    rows = []
    for col in cols:
        vc = df[col].value_counts(dropna=False).head(top_n)
        row = {
            "column": col,
            "n_unique": df[col].nunique(dropna=True),
            "missing_ratio": round(df[col].isna().mean(), 4),
        }
        if target and target in df.columns:
            rate = df.groupby(col, dropna=False)[target].mean().reindex(vc.index)
            row["top_values(count)"] = ", ".join(f"{k}={v}" for k, v in vc.items())
            row["success_rate_by_value"] = ", ".join(f"{k}={r:.3f}" for k, r in rate.items())
        else:
            row["top_values(count)"] = ", ".join(f"{k}={v}" for k, v in vc.items())
        rows.append(row)
    return pd.DataFrame(rows)


def print_categorical(cat_df: pd.DataFrame, with_rate: bool = True) -> None:
    for _, r in cat_df.iterrows():
        print(f"- {r['column']} (unique={r['n_unique']}, missing={r['missing_ratio']})")
        if with_rate and "success_rate_by_value" in cat_df.columns:
            print(f"    {r['success_rate_by_value']}")
        else:
            print(f"    {r['top_values(count)']}")


def missing_overview(df: pd.DataFrame) -> pd.DataFrame:
    """전체 컬럼 대상 결측치 개수/비율, 결측 있는 것만 내림차순."""
    miss = df.isna().sum().to_frame("missing_count")
    miss["missing_ratio"] = (miss["missing_count"] / len(df)).round(4)
    return miss[miss["missing_count"] > 0].sort_values("missing_ratio", ascending=False)


def success_rate_by_bin(df: pd.DataFrame, col: str, target: str, bins: int = 10) -> pd.DataFrame:
    """연속형 변수를 등빈도(bins)로 나눠 구간별 성공률. 연속형-타겟 관계를 표로 보는 용도."""
    binned = pd.qcut(df[col], q=bins, duplicates="drop")
    out = df.groupby(binned, observed=True)[target].agg(["mean", "count"])
    out.columns = ["success_rate", "n"]
    return out.round(4)


def feature_importance(df: pd.DataFrame, non_numeric_cols: list[str], numeric_cols: list[str], target: str) -> pd.Series:
    """RandomForest 기반 변수 중요도. 서브샘플(IMPORTANCE_SAMPLE_N)로 학습 (EDA용 근사치)."""
    non_numeric_cols = [c for c in non_numeric_cols if c in df.columns]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    sample = df.sample(n=min(IMPORTANCE_SAMPLE_N, len(df)), random_state=42)

    X = pd.DataFrame(index=sample.index)
    for c in non_numeric_cols:
        X[c] = pd.factorize(sample[c].astype(str))[0]
    for c in numeric_cols:
        X[c] = sample[c].fillna(sample[c].median())

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=42)
    rf.fit(X, sample[target])
    imp = pd.Series(rf.feature_importances_, index=non_numeric_cols + numeric_cols, name="importance")
    return imp.sort_values(ascending=False)


def cold_start_check(df: pd.DataFrame) -> None:
    """asof_pitcher_n == 0 인 행에서 asof_pitcher_* rate 컬럼이 실제로 결측인지 확인 (설명서 언급 사항)."""
    if "asof_pitcher_n" not in df.columns:
        return
    rate_cols = [c for c in TRAIN_TEST_NUMERIC if "asof_pitcher" in c and c.endswith("_rate")]
    zero_n = df["asof_pitcher_n"] == 0
    print(f"asof_pitcher_n == 0 인 행: {zero_n.sum()}건")
    if zero_n.sum() > 0:
        print(df.loc[zero_n, rate_cols].isna().mean().round(3).to_string())


def base_state_consistency_check(df: pd.DataFrame) -> None:
    """base_state 문자열이 runner_on_1b/2b/3b(이진형)와 일치하는지 확인."""
    needed = {"base_state", "runner_on_1b", "runner_on_2b", "runner_on_3b"}
    if not needed.issubset(df.columns):
        return
    expected = df["runner_on_1b"].map({1: "1", 0: "_"})
    expected += df["runner_on_2b"].map({1: "2", 0: "_"})
    expected += df["runner_on_3b"].map({1: "3", 0: "_"})
    mismatch = (expected != df["base_state"]).sum()
    print(f"base_state 불일치 행: {mismatch}건 / {len(df)}건")


def correlation_matrix(df: pd.DataFrame, numeric_cols: list[str], binary_cols: list[str]) -> pd.DataFrame:
    """수치형+시계열형(그대로) + 이진형(0/1로 인코딩) 컬럼의 피어슨 상관행렬. 범주형(명목형)은 순서가 없어서 제외."""
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    binary_cols = [c for c in binary_cols if c in df.columns]
    X = df[numeric_cols].copy()
    for c in binary_cols:
        X[c] = pd.factorize(df[c].astype(str))[0]
    return X.corr()


def top_correlated_pairs(corr: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """상관행렬에서 자기 자신/중복 쌍을 빼고 절댓값 기준 상위 top_n 쌍만 추출."""
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().reset_index()
    pairs.columns = ["col1", "col2", "corr"]
    pairs["abs_corr"] = pairs["corr"].abs()
    return pairs.sort_values("abs_corr", ascending=False).head(top_n).drop(columns="abs_corr").reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    train_df = load("train.csv")
    print(f"train shape={train_df.shape}")
    print("타겟 분포:", train_df[TARGET].value_counts(normalize=True).round(4).to_dict())

    missing = missing_overview(train_df)
    print("\n[결측치]")
    print(missing)

    num = numeric_summary(train_df, TRAIN_TEST_NUMERIC, target=TARGET)
    ts = numeric_summary(train_df, TRAIN_TEST_TIME_SERIES, target=TARGET)
    binc = categorical_summary(train_df, TRAIN_TEST_BINARY, target=TARGET)
    cat = categorical_summary(train_df, TRAIN_TEST_CATEGORICAL, target=TARGET)

    print("\n[수치형 상관계수 상위]")
    print(num.sort_values("corr_with_target", key=abs, ascending=False).head(10))

    corr = correlation_matrix(train_df, TRAIN_TEST_NUMERIC + TRAIN_TEST_TIME_SERIES, TRAIN_TEST_BINARY)
    print("\n[상관관계 상위 쌍]")
    print(top_correlated_pairs(corr, top_n=15))

    imp = feature_importance(
        train_df,
        TRAIN_TEST_BINARY + TRAIN_TEST_CATEGORICAL,
        TRAIN_TEST_NUMERIC + TRAIN_TEST_TIME_SERIES,
        TARGET,
    )
    print("\n[변수 중요도]")
    print(imp)

    num.to_csv(OUT_DIR / "train_numeric.csv")
    ts.to_csv(OUT_DIR / "train_time_series.csv")
    binc.to_csv(OUT_DIR / "train_binary.csv", index=False)
    cat.to_csv(OUT_DIR / "train_categorical.csv", index=False)
    missing.to_csv(OUT_DIR / "train_missing.csv")
    imp.to_csv(OUT_DIR / "train_feature_importance.csv")
    print("\nsaved to", OUT_DIR)


if __name__ == "__main__":
    main()
