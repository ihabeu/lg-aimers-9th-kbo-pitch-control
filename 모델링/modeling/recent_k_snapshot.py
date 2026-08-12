"""
제출(test.csv, 2025)용 투수별 "최근 K구 성공률" 스냅샷 생성.

규정상 test.csv 내부 행 순서 기반 rolling/expanding은 금지라서, train.csv(2019~2024) 마지막 시점까지의
값을 투수별로 딱 하나씩 고정(freeze)해서 만들고, 그 투수의 모든 2025 test 행에 동일하게 적용한다.
train.csv에 없던(2025 신인 등) 투수는 결측 -> CatBoost 네이티브 결측 처리.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

KS = [5, 10, 20, 50]


def build_snapshot() -> pd.DataFrame:
    df = load("train.csv")[["row_id", "pitcher_id", "control_success"]].copy()
    df["row_num"] = df["row_id"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("row_num")

    rows = []
    for pid, g in df.groupby("pitcher_id"):
        y = g["control_success"].to_numpy()
        rec = {"pitcher_id": pid}
        for k in KS:
            tail = y[-k:] if len(y) >= 1 else y
            rec[f"recent_{k}_pitch_rate"] = tail.mean() if len(tail) > 0 else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    snap = build_snapshot()
    out_path = Path(__file__).resolve().parent / "models" / "recent_k_snapshot.csv"
    out_path.parent.mkdir(exist_ok=True)
    snap.to_csv(out_path, index=False)
    print(f"saved {out_path}, {len(snap)}명")
    print(snap.describe())
