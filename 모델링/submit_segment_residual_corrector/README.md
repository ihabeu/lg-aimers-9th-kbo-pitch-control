**마지막 업데이트: 2026-08-12**

# 789.23 champion + game_type(R/F) segment residual corrector

우리 확정 champion CatBoost(`submit/model/catboost_baseline.cbm`, raw 44피처, 2019~2024 전체 학습,
실제 LB 789.23, **완전 독립개발**)를 그대로 base로 쓰고, base가 놓친 오차를 game_type(R/F) segment별로
ExtraTrees가 따로 학습해서 보정하는 구조. V14 아키텍처의 일반적인 발상(base + segment residual
correction)만 참고했고, 구체적 구현(어떤 segment로 나눌지, 어떤 하이퍼파라미터를 쓸지)은 전부 우리
자체 EDA/검증으로 새로 결정함 — 다른참가자의 hybrid team 발견이나 구체적 수치는 재사용 안 함.

## 아키텍처

```
champion CatBoost (789.23, 손대지 않음)
        │
   test row 예측 = base
        │
  + game_type(R/F) segment별 ExtraTrees 보정치 (seed 3개 평균)
        │
     최종 예측
```

## 로컬 검증 (`v3_domain_experiments/segment_residual_corrector.py`, pitcher-disjoint cross-fit)

| 폴드 | base | corrected | 개선 |
| --- | --- | --- | --- |
| 2023→2024 (primary) | 734.49 | **790.33** | +55.83 |
| 2022→2023 (stress) | 10.25 | **634.06** | +623.81 |

두 폴드 다 크게 개선. base 모델은 그대로 두고 오차 패턴만 segment별로 보정하는 구조라, 예전에
실패했던 "R/F 완전 분리 모델"(623.31, 표본이 줄어드는 손해)과 다르다.

## 패키징

- `model/catboost_baseline.cbm`: 기존 champion 그대로 복사(재학습 안 함)
- `model/correctors.joblib`: `build_correctors.py`가 오프라인으로 학습. 잔차 신호는 "2019~2023
  학습 모델이 2024를 예측했을 때 남긴 오차"(2024 라벨 전체로 최종 학습, non-cross-fit). segment(R/F) ×
  seed(3개) = 6개 ExtraTrees + 카테고리 인코딩 맵(학습/추론 시 동일한 고정 매핑 사용 — 매번 새로
  `.cat.codes`를 매기면 학습 때와 추론 때 코드값이 어긋나는 버그가 생겨서 고정 dict로 수정함)
- `script.py`: `data/test.csv` → champion 예측 + segment 보정 → `output/submission.csv`
- 로컬 test.csv(5행) 재현값이 `build_correctors.py`와 `script.py` 사이에 소수점까지 일치 확인
- 245,789행 스트레스 테스트: 약 2.3초 (제한 10분)

## 재생성 방법

```bash
python3 build_correctors.py   # model/correctors.joblib 재생성 (train.csv 필요)
zip -r submit.zip model script.py requirements.txt
```

## 상태

로컬 검증 완료, 패키징 완료. 실제 LB 제출은 사용자 진행 예정.
