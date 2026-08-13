**마지막 업데이트: 2026-08-13**

# 멀티모델(CatBoost+LightGBM+XGBoost) 가중 블렌드 + 3-way segment residual corrector

기존 champion(879.80)의 base를 CatBoost 단독에서 **CatBoost(0.6)+LightGBM(0.2)+XGBoost(0.2) 가중
블렌드**로 바꾸고, 그 위에 동일한 3-way segment(core/hybrid/dev) ExtraTrees corrector를 그대로
얹은 구조. 세 모델 간 residual 상관관계가 0.83~0.96(같은 계열 모델끼리의 0.998+보다 낮음)로 확인돼
블렌드에 진짜 다양성이 있다는 근거로 시도함. 하이퍼파라미터·가중치 그리드 전부 자체 설정.

## 로컬 검증 (`v3_domain_experiments/multimodel_weighted_blend.py`, 가중치 7종 스윕)

| weight(cat,lgb,xgb) | primary(2023→2024) | stress(2022→2023) |
|---|---:|---:|
| (1.0,0,0) 기존 champion | 801.93 | 755.63 |
| (0.8,.1,.1) | 816.00 | 799.16 |
| (0.7,.15,.15) | 814.23 | 821.55 |
| **(0.6,.2,.2) 채택** | **815.15** | **833.05** |
| (0.5,.25,.25) | 812.32 | 834.15 |
| (0.4,.3,.3) | 806.68 | 843.97 |
| (1/3,1/3,1/3) | 797.07 | 841.12 |

(0.8,.1,.1)부터 (0.4,.3,.3)까지 전 구간이 두 폴드 모두 기존 champion을 이김. (0.6,.2,.2)는
primary 최고점(816.00)에 근접하면서 stress도 최고점(843.97)에 가까운 균형점이라 채택.

## 아키텍처

```
0.6×CatBoost + 0.2×LightGBM + 0.2×XGBoost (전부 2019~2024 전체 학습)
        │
   test row 예측 = base
        │
  + segment(core/hybrid/dev)별 ExtraTrees 보정치 (seed 3개 평균)
        │
     최종 예측
```

corrector는 `submit_segment_residual_corrector`와 같은 구조(segment 정의, ExtraTrees 하이퍼파라미터,
pitcher-disjoint cross-fit 검증 방법론)를 그대로 재사용 — 상세는 그쪽 README 참고.

## 패키징

- `model/cat_final.cbm`: CatBoost, 2019~2024 전체 재학습
- `model/artifacts.joblib`: LightGBM/XGBoost 최종 모델, 가중치, corrector 9개(segment 3 × seed 3),
  라벨인코딩 맵(LightGBM/XGBoost는 CatBoost와 카테고리 처리 방식이 달라 고정 맵 필요)
- `script.py`: `data/test.csv` → 3-model 블렌드 + segment 보정 → `output/submission.csv`
- 로컬 test.csv(5행) 재현값이 `build_artifacts.py`와 `script.py` 사이에 소수점까지 일치 확인

## 재생성 방법

```bash
python3 build_artifacts.py   # model/ 재생성 (train.csv 필요)
zip -r submit.zip model script.py requirements.txt
```

## 상태

**로컬 검증 완료, 실제 LB 제출 전.** [`lg_aimers_submit_zip_policy`] 원칙대로 두 폴드 모두 신기록이라
zip을 빌드했지만, 실제 제출은 사용자 결정 대기 — 제출 후 이 절을 갱신할 것.
