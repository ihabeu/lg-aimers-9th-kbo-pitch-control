"""
공식 병합된 팀원의 프로덕션 pipeline(N601/N322/N329 등)에서 "적용 공식과 alpha/k/beta 값"만 참고해서
독립적으로 재구현한 3개의 EB(empirical Bayes) lookup 보정치. 실제 학습 로직(테이블을 어떻게
채웠는지)은 공유받은 파일에 없어서(추론 코드만 있음) 우리 자체 train.csv로 새로 집계해서 만든다.
함수/변수 이름은 원본과 겹치지 않게 새로 지음.

- pitcher_count_hand_correction (N601 참고): (pitcher_id, balls_before, strikes_before, batter_hand)
  조합별 잔차 평균을 EB로 축소(pseudo-count=100).
- pressure_backoff_correction (N322 참고): 3단계 계층(투수x압박x타자손 k=50 -> 투수x타자손 k=200 ->
  투수손x압박x타자손 k=1000) crossed-EB backoff. 1단계가 신뢰도 낮을수록(n 적을수록) 상위 단계로
  fallback.
- anchor_count_pressure_correction (N329 참고): R_ANCHOR(팀13 관여) 전용, 정확count(balls-strikes
  12칸)별 잔차 EB(k=200) + historical cap.

전부 leak-safe(train_df로만 테이블 학습, valid_df에 적용) + corrector 메타피처로 추가해서
champion(v14, 13개) 대비 테스트. Ridge/Lasso처럼 서로 겹칠 수 있는 것끼리는(전부 "잔차 평균 EB
lookup" 계열이라 겹칠 위험 있음) 조합이 아니라 각각 단독으로 먼저 확인.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_corrector_rmo_logratio_feature import (  # noqa: E402
    fit_base, corrector_matrix, assign_segment_3way, apply_corrector, pitcher_bootstrap_z,
)
from segment_corrector_joint_softmax_meta import add_joint_label, fit_joint_softmax, joint_softmax_meta  # noqa: E402
from segment_corrector_meta_source_sweep import fit_hazard_family as fit_hazard_family_v1  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eda"))
from rmo_labels import add_rmo_labels  # noqa: E402
from eda import load  # noqa: E402

CHAMPION_META = ["qR", "qM", "qO", "mr", "or", "om"]
JOINT_META = ["joint_p_success", "joint_p_reverse", "joint_p_middle", "joint_p_outside"]
HYBRID_TEAM_ID = 13


def count_pressure(balls: pd.Series, strikes: pd.Series) -> pd.Series:
    b, s = balls.astype(int), strikes.astype(int)
    return pd.Series(np.where((b == 3) & (s == 2), "full", np.where(b == 3, "3ball", np.where(s == 2, "2strike", "normal"))), index=balls.index)


def fit_pitcher_count_hand_correction(train_df, residual, alpha=100.0):
    key = train_df["pitcher_id"].astype(str) + "|" + train_df["balls_before"].astype(str) + "|" + train_df["strikes_before"].astype(str) + "|" + train_df["batter_hand"].astype(str)
    g = pd.DataFrame({"key": key.to_numpy(), "resid": residual}).groupby("key")["resid"].agg(["mean", "count"])
    shrunk = g["mean"] * (g["count"] / (g["count"] + alpha))
    return shrunk.to_dict()


def apply_pitcher_count_hand_correction(valid_df, table):
    key = valid_df["pitcher_id"].astype(str) + "|" + valid_df["balls_before"].astype(str) + "|" + valid_df["strikes_before"].astype(str) + "|" + valid_df["batter_hand"].astype(str)
    return key.map(table).fillna(0.0).to_numpy()


def _group_mean_n(df, keys, residual):
    g = pd.DataFrame({**{k: df[k].astype(str) for k in keys}, "resid": residual}).groupby(keys)["resid"].agg(["mean", "count"])
    return g


def fit_pressure_backoff_correction(train_df, residual, k1=50.0, k2=200.0, k3=1000.0):
    df = train_df.copy()
    df["pressure"] = count_pressure(df["balls_before"], df["strikes_before"])
    g1 = _group_mean_n(df, ["pitcher_id", "pressure", "batter_hand"], residual)
    g2 = _group_mean_n(df, ["pitcher_id", "batter_hand"], residual)
    g3 = _group_mean_n(df, ["pitcher_hand", "pressure", "batter_hand"], residual)
    return {"g1": g1, "g2": g2, "g3": g3, "k1": k1, "k2": k2, "k3": k3}


def apply_pressure_backoff_correction(valid_df, tables):
    df = valid_df.copy()
    df["pressure"] = count_pressure(df["balls_before"], df["strikes_before"])
    g1, g2, g3 = tables["g1"], tables["g2"], tables["g3"]
    k1, k2, k3 = tables["k1"], tables["k2"], tables["k3"]

    idx1 = pd.MultiIndex.from_frame(df[["pitcher_id", "pressure", "batter_hand"]].astype(str))
    idx2 = pd.MultiIndex.from_frame(df[["pitcher_id", "batter_hand"]].astype(str))
    idx3 = pd.MultiIndex.from_frame(df[["pitcher_hand", "pressure", "batter_hand"]].astype(str))

    n1 = idx1.map(g1["count"]).to_numpy(dtype=float); n1 = np.nan_to_num(n1)
    d1 = idx1.map(g1["mean"]).to_numpy(dtype=float); d1 = np.nan_to_num(d1)
    n2 = idx2.map(g2["count"]).to_numpy(dtype=float); n2 = np.nan_to_num(n2)
    d2 = idx2.map(g2["mean"]).to_numpy(dtype=float); d2 = np.nan_to_num(d2)
    n3 = idx3.map(g3["count"]).to_numpy(dtype=float); n3 = np.nan_to_num(n3)
    d3 = idx3.map(g3["mean"]).to_numpy(dtype=float); d3 = np.nan_to_num(d3)

    w1 = n1 / (n1 + k1)
    w2 = n2 / (n2 + k2)
    w3 = n3 / (n3 + k3)
    delta2 = w2 * d2
    delta3 = w3 * d3
    den = w2 + w3
    parent_prior = np.divide(w2 * delta2 + w3 * delta3, den, out=np.zeros_like(den), where=den > 0)
    return (1.0 - w1) * parent_prior


def fit_anchor_count_pressure_correction(train_df, residual, k=200.0, cap=0.01):
    df = train_df.copy()
    is_anchor = (df["pitcher_team_id"] == HYBRID_TEAM_ID) | (df["batter_team_id"] == HYBRID_TEAM_ID)
    df = df[is_anchor]
    residual_a = residual[is_anchor.to_numpy()]
    count_state = df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)
    g = pd.DataFrame({"key": count_state.to_numpy(), "resid": residual_a}).groupby("key")["resid"].agg(["mean", "count"])
    shrunk = (g["mean"] * (g["count"] / (g["count"] + k))).clip(-cap, cap)
    return shrunk.to_dict()


def apply_anchor_count_pressure_correction(valid_df, table):
    is_anchor = ((valid_df["pitcher_team_id"] == HYBRID_TEAM_ID) | (valid_df["batter_team_id"] == HYBRID_TEAM_ID)).to_numpy()
    count_state = valid_df["balls_before"].astype(str) + "-" + valid_df["strikes_before"].astype(str)
    corr = count_state.map(table).fillna(0.0).to_numpy()
    corr[~is_anchor] = 0.0
    return corr


def run_fold(df, target_season, label):
    print(f"\n===== {label}: <{target_season} -> {target_season} =====")
    train_df = df[df["season"] < target_season]
    valid_df = df[df["season"] == target_season].reset_index(drop=True)
    y = valid_df["control_success"].to_numpy(np.float64)

    base_pred = fit_base(train_df, valid_df)
    residual = y - base_pred

    # EB 테이블용 잔차: champion corrector 자체와 같은 "residual-source" 패턴 -- train_df의 마지막
    # 연도를 target으로 잡고, 그 이전 연도로 학습한 모델의 잔차를 사용(train_df 전체에 대한 잔차를
    # 만들려다 자기 자신을 학습에 써버리는 leak을 피함).
    table_target_season = int(train_df["season"].max())
    table_source = train_df[train_df["season"] < table_target_season]
    table_target = train_df[train_df["season"] == table_target_season].reset_index(drop=True)
    table_base_pred = fit_base(table_source, table_target)
    train_residual = table_target["control_success"].to_numpy(np.float64) - table_base_pred

    seg3 = assign_segment_3way(valid_df)
    X = corrector_matrix(valid_df)
    pitcher_ids = valid_df["pitcher_id"].to_numpy()

    print("  캐싱: champion 메타피처(CatBoost hazard6 + joint4 + Lasso3)...")
    cat_meta = fit_hazard_family_v1(train_df, valid_df, "catboost")
    joint_model = fit_joint_softmax(train_df)
    joint_meta = joint_softmax_meta(joint_model, valid_df)
    lasso_meta = fit_hazard_family_v1(train_df, valid_df, "lasso")

    X_champion = X.copy()
    for c in CHAMPION_META:
        X_champion[f"rmo_{c}"] = cat_meta[c]
    for c in JOINT_META:
        X_champion[c] = joint_meta[c]
    for c in ["mr", "or", "om"]:
        X_champion[f"lasso_{c}"] = lasso_meta[c]
    champion_score, champion_pred = apply_corrector(X_champion, residual, seg3, base_pred, y, valid_df)
    print(f"  champion(v14, 메타피처 13개): {champion_score:.2f}  (기준)")

    pch_table = fit_pitcher_count_hand_correction(table_target, train_residual)
    backoff_tables = fit_pressure_backoff_correction(table_target, train_residual)
    anchor_table = fit_anchor_count_pressure_correction(table_target, train_residual)

    pch_corr = apply_pitcher_count_hand_correction(valid_df, pch_table)
    backoff_corr = apply_pressure_backoff_correction(valid_df, backoff_tables)
    anchor_corr = apply_anchor_count_pressure_correction(valid_df, anchor_table)

    results = {}
    for name, corr in [("pitcher_count_hand", pch_corr), ("pressure_backoff", backoff_corr), ("anchor_count_pressure", anchor_corr)]:
        X_new = X_champion.copy()
        X_new[f"eb_{name}"] = corr
        s, pred = apply_corrector(X_new, residual, seg3, base_pred, y, valid_df)
        d = (champion_pred - y) ** 2 - (pred - y) ** 2
        mean_d, z = pitcher_bootstrap_z(d, pitcher_ids)
        print(f"  +{name}: {s:.2f}  차이={s - champion_score:+.2f}  z={z:.2f}")
        results[name] = (s, z)
    return champion_score, results


def main():
    df = add_rmo_labels(load("train.csv"))
    df = add_joint_label(df)
    r1 = run_fold(df, 2024, "PRIMARY")
    r2 = run_fold(df, 2023, "STRESS")
    print("\n===== 요약 =====")
    for name in r1[1]:
        s1, z1 = r1[1][name]
        s2, z2 = r2[1][name]
        print(f"  [+{name}] PRIMARY {s1:.2f}(z={z1:.2f})  STRESS {s2:.2f}(z={z2:.2f})")


if __name__ == "__main__":
    main()
