"""
train.csv의 pitcher_id <-> trackman_history.csv의 pitcher_trackman_id 매핑 복원.

DACON 운영진 Q&A 확인 사항(2026-08-07, HANDOFF.md 참고): 대응 관계 추정 허용, 그 대응으로 투구 이전
Trackman 통계치를 투수 단위 요약 피처로 쓰는 것도 허용.

train.csv엔 game_date/game_id/투구순번이 없어 개별 투구 1:1 정렬(row-level join)은 불가능하다.
대신 투수별 (season, game_month) 투구수 히스토그램을 지문(fingerprint)으로 써서 최근접 이웃 매칭한다.

품질 검증 (pitcher_hand는 두 파일에 다 있고 실제로 안 바뀌는 고정 속성이라 검증용으로만 쓰고 매칭 신호로는
안 씀 - 매핑이 맞다면 반드시 일치해야 하는 필요조건):
- 무작위 매칭 hand 일치율(기준선): 64.8%
- 지문 최근접 매칭 전체 hand 일치율: 81.1% (신호 있음, 순수 노이즈 아님)
- 확신도(1위-2위 거리 상대 gap) 상위 25%: 97.0%, 상위 50%: 91.7%

따라서: hand 불일치는 확정적으로 틀린 매칭이라 하드 필터로 제외하고, 그중에서도 gap이 임계값 이상인
것만 채택한다. hand는 매칭 신호 생성에 쓰지 않았으므로 이 필터링은 순환 검증(circular)이 아니다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

TRAIN_HAND_MAP = {2: "Right", 1: "Left"}  # train 다수값(2)이 trackman 다수값(Right)에 대응
REL_GAP_THRESHOLD = 0.222  # 상위 50% 확신도 컷 (hand 일치율 91.7%)


def build_fingerprints(train: pd.DataFrame, tm: pd.DataFrame):
    train_fp = train.groupby(["pitcher_id", "season", "game_month"]).size().unstack(["season", "game_month"], fill_value=0)
    tm_fp = tm.groupby(["pitcher_trackman_id", "season", "game_month"]).size().unstack(["season", "game_month"], fill_value=0)
    all_cols = sorted(set(train_fp.columns) | set(tm_fp.columns))
    train_fp = train_fp.reindex(columns=all_cols, fill_value=0)
    tm_fp = tm_fp.reindex(columns=all_cols, fill_value=0)
    return train_fp, tm_fp


def build_mapping(rel_gap_threshold: float = REL_GAP_THRESHOLD) -> pd.DataFrame:
    """반환: pitcher_id, pitcher_trackman_id, rel_gap, hand_match 컬럼을 가진 신뢰 매핑 테이블."""
    train = load("train.csv")[["pitcher_id", "season", "game_month", "pitcher_hand"]]
    tm = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "data" / "trackman_history.csv",
                      usecols=["pitcher_trackman_id", "season", "game_month", "pitcher_hand"])

    train_fp, tm_fp = build_fingerprints(train, tm)
    train_hand = train.drop_duplicates("pitcher_id").set_index("pitcher_id")["pitcher_hand"].map(TRAIN_HAND_MAP)
    tm_hand = tm.drop_duplicates("pitcher_trackman_id").set_index("pitcher_trackman_id")["pitcher_hand"]

    D = cdist(train_fp.to_numpy().astype(float), tm_fp.to_numpy().astype(float), metric="euclidean")
    train_ids = train_fp.index.to_numpy()
    tm_ids = tm_fp.index.to_numpy()

    sorted_D = np.sort(D, axis=1)
    best_dist, second_dist = sorted_D[:, 0], sorted_D[:, 1]
    rel_gap = (second_dist - best_dist) / (best_dist + 1.0)
    best_tm_ids = tm_ids[D.argmin(axis=1)]

    mapping = pd.DataFrame({
        "pitcher_id": train_ids,
        "pitcher_trackman_id": best_tm_ids,
        "rel_gap": rel_gap,
    })
    mapping["hand_match"] = [
        train_hand.get(pid) == tm_hand.get(tid) for pid, tid in zip(mapping.pitcher_id, mapping.pitcher_trackman_id)
    ]

    confident = mapping[mapping.hand_match & (mapping.rel_gap >= rel_gap_threshold)].copy()
    dup = confident.pitcher_trackman_id.duplicated().sum()
    # many-to-one 충돌: 같은 trackman_id에 여러 train pitcher_id가 몰리면 최소 하나는 틀린 매칭이므로
    # rel_gap(확신도) 가장 높은 것만 남기고 나머지는 버린다.
    confident = confident.sort_values("rel_gap", ascending=False).drop_duplicates("pitcher_trackman_id", keep="first")
    print(f"전체 train 투수 {len(mapping)}명 중 신뢰 매핑 {len(confident)}명 "
          f"({len(confident) / len(mapping):.1%}), many-to-one 충돌 제거 {dup}건")
    return confident.drop(columns=["hand_match"]).reset_index(drop=True)


if __name__ == "__main__":
    m = build_mapping()
    print(m.head(10))
