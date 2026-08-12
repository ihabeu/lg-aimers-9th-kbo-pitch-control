"""
Trackman 매핑 v3: 구단(team) crosswalk를 매칭 제약으로 추가.

v2(월별 히스토그램 fingerprint + 헝가리안, hand는 사후검증만)의 신뢰 매핑 395명으로 구단 crosstab을
뽑아보니 train pitcher_team_id 13개 중 10개가 실제 KBO 10개 구단과 뚜렷하게 1:1 대응됨이 확인됐다
(예: 12=두산DOO_BEA, 17=한화HAN_EAG, 18=삼성SAM_LIO 등). 22/23/25는 표본이 너무 적어(각 1건) 매핑 불가.

hand(2값)보다 team(10개 클럽, 1군+2군 코드 페어 포함)이 식별력이 훨씬 높은 하드 제약이 될 수 있다는
가설로, team 불일치 쌍을 헝가리안 비용에서 배제하고 재매칭한다. team 정보 자체가 v2의 confident
매핑에서 유도된 것이라 완전히 독립적인 검증은 아니지만(hand는 여전히 독립적), 이미 신뢰도 높은
집합에서 나온 crosswalk를 나머지 매칭 개선에 쓰는 건 준지도학습적으로 타당하다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trackman_mapping_v2 import build_month_fingerprints, build_mapping_v2, TRAIN_HAND_MAP  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eda"))
from eda import load  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def build_team_crosswalk() -> dict:
    """v2 confident 매핑(395명)으로 train_team_id -> trackman_team 다수결 crosswalk 유도."""
    train = load("train.csv")[["pitcher_id", "pitcher_team_id"]]
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv", usecols=["pitcher_trackman_id", "pitcher_team"])

    mapping = build_mapping_v2()
    m = mapping.merge(train.drop_duplicates("pitcher_id"), on="pitcher_id")
    tm_team_mode = tm.groupby("pitcher_trackman_id")["pitcher_team"].agg(lambda s: s.value_counts().idxmax())
    m["trackman_team"] = m["pitcher_trackman_id"].map(tm_team_mode)

    crosswalk = {}
    for train_team, g in m.groupby("pitcher_team_id"):
        counts = g["trackman_team"].value_counts()
        if len(counts) == 0 or counts.iloc[0] < 3:  # 표본 3건 미만은 신뢰 부족 -> crosswalk 없음
            continue
        top_team = counts.index[0]
        # 1군 코드와 2군(MIN_) 코드 둘 다 허용 (train game_type=R/F로 실제 어느 쪽인지 갈리므로)
        allowed = {top_team}
        base_code = top_team.split("_")[-1] if not top_team.startswith("MIN_") else top_team[4:]
        for code in tm["pitcher_team"].unique():
            if code.endswith(base_code) or code == f"MIN_{top_team[:3]}":
                allowed.add(code)
        crosswalk[train_team] = allowed
    return crosswalk


def build_mapping_v3(cost_threshold: float = 0.2323) -> pd.DataFrame:
    train = load("train.csv")[["pitcher_id", "pitcher_team_id", "season", "game_month", "pitcher_hand"]]
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv",
                      usecols=["pitcher_trackman_id", "pitcher_team", "season", "game_month", "pitcher_hand"])

    crosswalk = build_team_crosswalk()
    print(f"team crosswalk 유도됨: {len(crosswalk)}개 train team_id (전체 13개 중)")

    train_fp, tm_fp = build_month_fingerprints(train[["pitcher_id", "season", "game_month"]], tm[["pitcher_trackman_id", "season", "game_month"]])
    train_ids = train_fp.index.to_numpy()
    tm_ids = tm_fp.index.to_numpy()

    train_team = train.drop_duplicates("pitcher_id").set_index("pitcher_id")["pitcher_team_id"]
    tm_team_mode = tm.groupby("pitcher_trackman_id")["pitcher_team"].agg(lambda s: s.value_counts().idxmax())

    D = cdist(train_fp.to_numpy(), tm_fp.to_numpy(), metric="euclidean")

    # crosswalk가 있는 train_team_id에 대해서만 team 불일치를 하드 제약으로 (없으면 제약 없음)
    penalty = np.zeros_like(D)
    for i, pid in enumerate(train_ids):
        tteam = train_team.get(pid)
        if tteam not in crosswalk:
            continue
        allowed = crosswalk[tteam]
        for j, tid in enumerate(tm_ids):
            if tm_team_mode.get(tid) not in allowed:
                penalty[i, j] = 1e6
    D_constrained = D + penalty

    row_idx, col_idx = linear_sum_assignment(D_constrained)
    cost = D[row_idx, col_idx]  # 검증엔 penalty 뺀 순수 거리 사용
    infeasible = D_constrained[row_idx, col_idx] >= 1e6

    mapping = pd.DataFrame({
        "pitcher_id": train_ids[row_idx], "pitcher_trackman_id": tm_ids[col_idx], "cost": cost,
    })
    mapping = mapping[~infeasible]
    confident = mapping[mapping["cost"] <= cost_threshold].reset_index(drop=True)
    print(f"team 제약 헝가리안: 전체 {len(mapping)}명 중 cost<={cost_threshold} 신뢰 매핑 {len(confident)}명")
    return confident, mapping, train_ids, tm_ids, D, D_constrained, row_idx, col_idx


def get_confident_mapping(cost_threshold: float = 0.2323) -> pd.DataFrame:
    """다른 스크립트에서 재사용할 간단한 인터페이스: pitcher_id, pitcher_trackman_id, cost만 반환."""
    confident, *_ = build_mapping_v3(cost_threshold)
    return confident[["pitcher_id", "pitcher_trackman_id", "cost"]]


if __name__ == "__main__":
    confident, mapping, train_ids, tm_ids, D, D_constrained, row_idx, col_idx = build_mapping_v3()

    # 검증: hand 일치율 (여전히 매칭에 안 쓰고 독립 검증)
    train = load("train.csv")[["pitcher_id", "pitcher_hand"]]
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv", usecols=["pitcher_trackman_id", "pitcher_hand"])
    train_hand = train.drop_duplicates("pitcher_id").set_index("pitcher_id")["pitcher_hand"].map(TRAIN_HAND_MAP)
    tm_hand = tm.drop_duplicates("pitcher_trackman_id").set_index("pitcher_trackman_id")["pitcher_hand"]

    matched_pids = mapping["pitcher_id"].to_numpy()
    matched_tids = mapping["pitcher_trackman_id"].to_numpy()
    hand_match = np.array([train_hand.get(p) == tm_hand.get(t) for p, t in zip(matched_pids, matched_tids)])
    print(f"\nv3 전체 hand 일치율(독립검증): {hand_match.mean():.1%} (v2는 84.97%였음)")

    conf_pids = confident["pitcher_id"].to_numpy()
    conf_tids = confident["pitcher_trackman_id"].to_numpy()
    conf_hand_match = np.array([train_hand.get(p) == tm_hand.get(t) for p, t in zip(conf_pids, conf_tids)])
    print(f"v3 confident({len(confident)}명) hand 일치율: {conf_hand_match.mean():.1%} (v2 395명은 93.7%였음)")
