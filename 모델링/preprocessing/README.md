# 전처리 정리

**작성일: 2026-08-12**

모델 종류마다 필요한 전처리가 다르다는 걸 이 프로젝트에서 명확히 확인했다. "전처리 없음"이 CatBoost에서 최선이었던 건 CatBoost 특성이지, 다른 모델에도 그대로 적용되는 규칙이 아니다.

## CatBoost (v1 baseline) — 전처리 거의 없음

- **결측치**: 그대로 둠. CatBoost가 수치형 NaN은 "최솟값보다 작은 값"으로 내부 처리해서 분기 방향을 학습하고, 범주형 NaN은 하나의 카테고리로 자동 처리.
- **범주형 인코딩**: 없음. CatBoost native categorical(ordered target statistics)을 그대로 사용 — 원-핫이나 라벨인코딩을 사람이 만들면 오히려 손해(이 프로젝트에서 직접 확인).
- **스케일링**: 없음. 트리 기반이라 피처 스케일에 영향 안 받음.
- **다중공선성 제거**: 안 함. 완전 중복 변수를 제거했더니 오히려 성능이 떨어짐(734.49→709.35) — 트리는 계수를 추정하는 게 아니라서 다중공선성에 안 불안정해지고, 중복된 합계 변수가 depth 제한된 트리의 "지름길" 역할을 함.

## Elastic Net (선형모델) — 전처리 필수, CatBoost와 반대 철학

- **결측치**: `SimpleImputer(strategy="median")`, train 통계로만 fit (sklearn Pipeline 안에 넣어서 leak 방지).
- **범주형 인코딩**: `OneHotEncoder(drop="if_binary")`.
- **스케일링**: `StandardScaler` — 계수 크기 비교 가능하게 하고 saga solver 수렴 빠르게 함.
- **다중공선성 제거**: 함(CatBoost와 반대) — 선형모델은 다중공선성이 실제로 계수 추정을 불안정하게 만들기 때문.

## NN (Embedding MLP) — 전처리 필수, 혼합형

- **결측치**: 수치형은 median 대체. 범주형은 별도 "학습 때 못 본 값" 인덱스(0)로 처리.
- **범주형 인코딩**: 원-핫이 아니라 embedding(정수 인덱스 매핑, train 카테고리만으로 fit).
- **스케일링**: 수치형만 `StandardScaler`.

자세한 실험은 [`../EXPERIMENTS.md`](../EXPERIMENTS.md) 참고.
