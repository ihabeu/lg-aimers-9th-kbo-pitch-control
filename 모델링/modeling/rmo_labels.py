"""
R/M/O(reverse/middle/outside) 보조 라벨 복원.

`asof_pitcher_reverse_rate`/`asof_pitcher_middle_rate`는 현재 행 "이전"까지의 누적 비율이다.
row_id가 실제 시간순임을 확인했고(이 세션에서 검증됨), 같은 투수의 asof_pitcher_n이 row_id 순으로
항상 정확히 +1씩 증가한다(전체 데이터 100% 확인). 그러므로 투수의 "다음" 행의 누적 reverse count에서
"현재" 행의 누적 reverse count를 빼면, 현재 행 자체가 reverse였는지(0/1)를 정확히 복원할 수 있다:

reverse_count(i) = asof_pitcher_reverse_rate(i) * asof_pitcher_n(i)
reverse_indicator(i) = reverse_count(다음 행) - reverse_count(i)   # 0 또는 1

각 투수의 마지막 행은 "다음 행"이 없어 복원 불가 -> NaN (해당 행은 R/M/O 서브모델 학습에서 제외).

정의상 상호배타적: control_success=1이면 reverse=middle=outside=0.
control_success=0이면 reverse/middle/outside 중 정확히 하나만 1.
outside_indicator = (1-control_success) - reverse_indicator - middle_indicator 로 역산.
"""
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402


def add_rmo_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["row_num"] = df["row_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("row_num")

    for comp in ["reverse", "middle"]:
        rate_col = f"asof_pitcher_{comp}_rate"
        count_col = f"_{comp}_count"
        df[count_col] = df[rate_col] * df["asof_pitcher_n"]

    g = df.groupby("pitcher_id")
    df["_reverse_count_next"] = g["_reverse_count"].shift(-1)
    df["_middle_count_next"] = g["_middle_count"].shift(-1)

    df["reverse_label"] = (df["_reverse_count_next"] - df["_reverse_count"]).round().clip(0, 1)
    df["middle_label"] = (df["_middle_count_next"] - df["_middle_count"]).round().clip(0, 1)
    # 마지막 행(다음 행 없음)은 NaN 유지
    no_next = df["_reverse_count_next"].isna()
    df.loc[no_next, ["reverse_label", "middle_label"]] = np.nan

    df["outside_label"] = (1 - df["control_success"]) - df["reverse_label"] - df["middle_label"]

    return df.drop(columns=["_reverse_count", "_middle_count", "_reverse_count_next", "_middle_count_next"]).sort_index()


def validate(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["reverse_label", "middle_label"])
    print(f"복원 가능 행: {len(valid):,} / {len(df):,} ({len(valid)/len(df):.2%})")

    # 상호배타성 검증
    failure = valid[valid["control_success"] == 0]
    success = valid[valid["control_success"] == 1]
    print(f"\ncontrol_success=1인 행에서 reverse/middle 전부 0인 비율: "
          f"{((success['reverse_label']==0) & (success['middle_label']==0)).mean():.4f}")
    print(f"control_success=0인 행에서 outside_label이 0 또는 1인 비율: "
          f"{failure['outside_label'].isin([0,1]).mean():.4f}")
    print(f"\nfailure 행 중 reverse/middle/outside 분포:")
    print(failure[["reverse_label", "middle_label", "outside_label"]].sum())
    print(f"\nreverse_label 값 분포(0/1 외 이상치 확인):")
    print(valid["reverse_label"].value_counts())


if __name__ == "__main__":
    df = add_rmo_labels(load("train.csv"))
    validate(df)
