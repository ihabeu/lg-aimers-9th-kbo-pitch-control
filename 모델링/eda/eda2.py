"""
2023 vs 나머지 연도 비교 분석.

지난 실험에서 2019~2022로 학습한 모델이 2023 예측에서 베이스라인보다 못했다(Brier > r(1-r)).
"2023이 특이한 연도"라는 가설을 데이터로 직접 확인한다: (1) 타겟 자체의 연도별 추세에 특이점이 있는가,
(2) game_type=F의 의미가 실제로 뒤집히는가, (3) 입력 피처 분포 자체가 2023에 달라지는가(covariate shift),
(4) 처음 보는 투수/타자 비율이 2023에 튀는가, (5) 팀 구성이 바뀌는가.

함수만 정의. 실행 결과는 eda2.ipynb 참고.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda import DATA_DIR, TARGET, TRAIN_TEST_NUMERIC, load  # noqa: E402


def yearly_target_summary(df: pd.DataFrame) -> pd.DataFrame:
    """연도별 성공률, 표본수, 전년 대비 변화폭."""
    out = df.groupby("season")[TARGET].agg(["mean", "count"]).rename(columns={"mean": "success_rate"})
    out["yoy_delta"] = out["success_rate"].diff().round(4)
    return out.round(4)


def regime_crosstab(df: pd.DataFrame, cat_col: str = "game_type") -> pd.DataFrame:
    """연도 x 범주값 별 성공률 피벗. game_type=F의 의미가 실제로 뒤집히는지 확인용."""
    return df.groupby(["season", cat_col])[TARGET].agg(["mean", "count"]).unstack(cat_col).round(4)


def feature_shift_2023_vs_rest(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """수치형 피처의 2023 평균 vs 나머지 연도 평균. 입력 분포 자체가 바뀌었는지(covariate shift) 확인."""
    cols = cols or TRAIN_TEST_NUMERIC
    cols = [c for c in cols if c in df.columns]
    is_2023 = df["season"] == 2023
    rest_mean = df.loc[~is_2023, cols].mean()
    y2023_mean = df.loc[is_2023, cols].mean()
    out = pd.DataFrame({"rest_mean": rest_mean, "y2023_mean": y2023_mean})
    out["diff"] = out["y2023_mean"] - out["rest_mean"]
    out["diff_pct"] = (out["diff"] / out["rest_mean"].abs()).round(4)
    return out.round(4).sort_values("diff_pct", key=abs, ascending=False)


def new_entity_ratio_by_year(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """연도별로 그 해 처음 등장하는 id 비율. '처음 보는 투수/타자'가 특정 연도에 튀는지 확인."""
    seen = set()
    rows = []
    for season, group in df.sort_values("season").groupby("season"):
        ids = group[id_col].unique()
        new_ids = set(ids) - seen
        rows.append({
            "season": season,
            "unique_ids": len(ids),
            "new_ids": len(new_ids),
            "new_ratio": round(len(new_ids) / len(ids), 4),
        })
        seen |= set(ids)
    return pd.DataFrame(rows).set_index("season")


def team_composition_by_year(df: pd.DataFrame, team_col: str = "pitcher_team_id") -> pd.DataFrame:
    """연도별 팀 등장 비율 (팀 구성/증감 여부 확인)."""
    return pd.crosstab(df["season"], df[team_col], normalize="index").round(4)
