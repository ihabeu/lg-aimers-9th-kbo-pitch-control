# 프로젝트 인덱스

**마지막 업데이트: 2026-08-14**

`모델링/` 전체를 처음 열었을 때 어디부터 봐야 할지 안내하는 문서. 각 항목의 상세 내용은 아래 표에서 링크한 문서에만 두고 여기서는 중복하지 않는다.

## 문서 안내 (처음 읽는 순서)

| 문서 | 내용 |
| --- | --- |
| [`대회 목적 및 규칙.md`](대회%20목적%20및%20규칙.md) | 대회 배경, 문제 정의, 평가 지표, 제출 규정, leak 방지 규정 |
| [`도메인.md`](도메인.md) | 야구 규칙, 컬럼이 의미하는 것(이닝/카운트/주자/제구 등) |
| [`전처리 및 인사이트.md`](전처리%20및%20인사이트.md) | 데이터 특성, 중요 변수, 시즌 drift, team 13 발견, 모델별 전처리 차이 |
| [`../eda/EDA.md`](../eda/EDA.md) / [`../eda/COLUMNS.md`](../eda/COLUMNS.md) | EDA 산출물 원본, 컬럼별 고유값·분류표 |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | 실험 로그(E-번호 순차 기록) |
| [`HANDOFF.md`](HANDOFF.md) | 현재 champion, 제출 이력, 폴더 구조, 전체 진행 상태 |

원본 컬럼 정의(운영진 제공)는 [`../../data/data_description.md`](../../data/data_description.md) 참고.

## 코드 폴더 위치

- [`../eda/`](../eda/) — EDA 스크립트/노트북/산출물
- [`../modeling/`](../modeling/) — 공유 모델 개발 라이브러리(`baseline_catboost.py` 등, 여러 스크립트가 여기서 import)
- [`../개발/`](../개발/) — 버전별 개발 코드(`v1_baseline_catboost/`, `v3_domain_experiments/`)
- [`../submit/`](../submit/) — 버전별 제출 패키지(`v1_baseline/` ~ `v10_multimodel_blend/`), 상세는 `HANDOFF.md`의 제출 이력 표 참고

코드 작성 원칙: 실제 로직은 `.py`에 함수로 두고, `.ipynb`는 그 `.py`를 import해서 셀 단위로 실행하며 결과와 의도(마크다운)를 같이 남긴다.
