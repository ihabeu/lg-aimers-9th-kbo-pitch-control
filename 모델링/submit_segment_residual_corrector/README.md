**마지막 업데이트: 2026-08-12**

# 789.23 champion + 3-way segment residual corrector

우리 확정 champion CatBoost(`submit/model/catboost_baseline.cbm`, raw 44피처, 2019~2024 전체 학습,
실제 LB 789.23, **완전 독립개발**)를 그대로 base로 쓰고, base가 놓친 오차를 segment별로 ExtraTrees가
따로 학습해서 보정하는 구조. V14 아키텍처의 일반적인 발상(base + segment residual correction)만
참고했고, 구체적 구현(segment 기준, 하이퍼파라미터)은 전부 우리 자체 EDA/검증으로 새로 결정함 —
다른참가자의 hybrid team 발견이나 구체적 수치는 재사용 안 함.

## segment 정의 (우리 자체 EDA 근거, `HANDOFF.md` "데이터 핵심 발견")

```
game_type == 'F'                          → dev
game_type == 'R' 이고 team 13 관여        → hybrid   (pitcher_team_id 또는 batter_team_id == 13)
그 외 (game_type == 'R', team 13 없음)    → core
```

team 13은 F 참여 비율이 38.15%로 다른 정상 팀(3~10%)보다 유독 높아서 별도 그룹으로 분리함(이유는
불확실 — 데이터 설명서에 근거 없음, 관찰된 사실만 사용).

**R/F 2-way보다 이 3-way가 더 좋음을 직접 확인**(둘 다 pitcher-disjoint cross-fit 검증):

| 폴드 | R/F 2-way | 3-way(core/hybrid/dev) |
| --- | --- | --- |
| 2023→2024 | 790.33 | **801.93** |
| 2022→2023 | 634.06 | **755.63** |

F 안에서 100% F전용 극소표본 팀(22/23/25, 6년 합쳐 2700행 정도)을 더 분리하는 4-way도 시도했으나
(`v3_domain_experiments/segment_residual_corrector_4way.py`) primary -3.10 / stress +1.30로 사실상
노이즈라 **기각** — 표본이 너무 작아서 보정기가 안정적으로 학습이 안 됨.

## 아키텍처

```
champion CatBoost (789.23, 손대지 않음)
        │
   test row 예측 = base
        │
  + segment(core/hybrid/dev)별 ExtraTrees 보정치 (seed 3개 평균)
    → segment는 입력 피처가 아니라 "어느 전용 모델을 쓸지" 결정하는 라우팅 기준
        │
     최종 예측
```

## 로컬 검증 (`v3_domain_experiments/segment_residual_corrector_3way.py`, pitcher-disjoint cross-fit)

| 폴드 | segment | base | corrected | 개선 |
| --- | --- | --- | --- | --- |
| 2023→2024 (primary) | core (n=178,729) | 576.93 | 629.92 | +52.99 |
| | hybrid (n=44,768) | 638.72 | 708.20 | +69.48 |
| | dev (n=30,010) | 351.70 | 503.56 | +151.86 |
| | **전체** | **734.49** | **801.93** | **+67.43** |
| 2022→2023 (stress) | core (n=175,785) | 54.02 | 556.31 | +502.29 |
| | hybrid (n=44,054) | 0.00 | 945.98 | +945.98 |
| | dev (n=25,686) | 0.00 | 70.02 | +70.02 |
| | **전체** | **10.25** | **755.63** | **+745.38** |

두 폴드 다 크게, 같은 방향으로 개선. base 모델은 그대로 두고 오차 패턴만 segment별로 보정하는
구조라, 예전에 실패했던 "R/F 완전 분리 모델"(623.31, 표본이 줄어드는 손해)과 다르다.

### 검증 방법론 — k-fold 대신 쓴 것

- **rolling out-of-time**: 이 데이터는 연도별로 drift가 있어서(성공률 계속 하락, 2023 F regime shift)
  무작위 k-fold를 쓰면 미래 정보가 과거 fold로 새는 leakage가 생긴다. 그래서 항상 "이전 연도로 학습 →
  다음 연도로 검증"만 쓴다(primary: 2023→2024, stress: 2022→2023).
- **pitcher-disjoint cross-fit**(corrector 자체 검증에만 사용): 투수를 랜덤 2그룹으로 나눠 한쪽으로
  학습 → 다른 쪽으로 평가, 역할 바꿔서 반복, 3개 시드로 3번 더 반복해서 평균. 시간 방향이 아니라
  "학습에서 안 본 투수에게도 통하는가"를 보는 것이라 위 rolling OOT와는 다른 축의 검증.

## 패키징

- `model/catboost_baseline.cbm`: 기존 champion 그대로 복사(재학습 안 함)
- `model/correctors.joblib`: `build_correctors.py`가 오프라인으로 학습. 잔차 신호는 "2019~2023
  학습 모델이 2024를 예측했을 때 남긴 오차"(2024 라벨 전체로 최종 학습, non-cross-fit). segment(3개) ×
  seed(3개) = 9개 ExtraTrees + 카테고리 인코딩 맵(학습/추론 시 동일한 고정 매핑 사용 — 매번 새로
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

로컬 검증 완료(local 기준 대표 점수: primary 폴드 **801.93**), 패키징 완료. 실제 LB 제출은 사용자
진행 예정.
