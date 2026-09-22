# Modeling Plan

## Lag And Rolling Validation

기존 통합 데이터셋은 보존했습니다. 모델 실험용 데이터셋에서는 지자체별 실제 calendar day 기준으로 `lag_1`, `lag_7`, `rolling_mean_7`, `rolling_mean_14`를 재계산했습니다.

이전 버전의 rolling 변수는 `shift(1)` 뒤 `min_periods=1`로 계산되어 각 지자체의 첫 관측일만 결측이었습니다. 그래서 `rolling_mean_7`과 `rolling_mean_14` 초기 결측치가 각각 지자체 수와 같은 181개였습니다.

시계열 검증을 더 엄격하게 하기 위해 이번 실험 데이터셋은 정확히 이전 7일 또는 14일 calendar-day 값이 모두 존재할 때만 rolling 평균을 생성합니다. 현재 날짜의 `waste_amount`는 rolling 계산에 포함하지 않습니다.

## Date Continuity

- 날짜 공백이 있는 지자체 수: 15
- 전체 누락 calendar day 수: 1097

## Experimental Datasets

- `weather_validated_dataset.csv`: 102,070 rows, 96 municipalities
- `nationwide_dataset.csv`: 186,841 rows, 179 municipalities

`sumRn`은 원본 컬럼으로 보존하지만 1차 모델 feature에서는 제외합니다. 결측이 많고 강수 없음과 관측 결측을 명확히 구분하지 못했기 때문입니다. 같은 날 `discharge_count`도 target leakage 가능성이 있어 feature에서 제외합니다.

## Time Split

랜덤 분할은 사용하지 않습니다.
- test: 2023-07-01 ~ 2024-01-31, rows=38,349, municipalities=181
- train: 2021-01-01 ~ 2022-12-31, rows=120,899, municipalities=174
- validation: 2023-01-01 ~ 2023-06-30, rows=31,452, municipalities=176

## Feature Candidates

Weather validated model:
- sigungu_key, total_population, households, population_per_household, month, day_of_week, is_weekend, season, lag_1, lag_7, rolling_mean_7, rolling_mean_14, avgTa, minTa, maxTa, avgRhm

Nationwide model:
- sigungu_key, total_population, households, population_per_household, month, day_of_week, is_weekend, season, lag_1, lag_7, rolling_mean_7, rolling_mean_14

Target: `waste_amount`

Excluded from 1st experiment:
- `discharge_count`
- `sumRn`
- future or same-day post-outcome variables

## Target Distribution

- Mean: 15,251,202.57
- Median: 9,823,500.00
- Std: 18,503,905.08
- Top 1% threshold: 97,661,115.50
- Global IQR high rows: 7,939
- Municipality-level IQR high rows: 3,900

배출량 규모가 지자체별로 크게 다르므로 전역 IQR 초과 행을 자동 삭제하지 않습니다. 대형 지자체의 정상적인 규모 효과일 수 있어, 모델링 단계에서 원 target과 `log1p(waste_amount)` target 실험을 함께 비교하는 것을 권장합니다.
