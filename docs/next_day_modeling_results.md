# Next-Day Modeling Results

## Problem Definition

날짜 `t`에 사용 가능한 지자체, 인구, 달력, 과거 배출 이력 feature를 이용해 실제 calendar day `t+1`의 `target_next_day` 배출량을 예측합니다.

## Feature Availability And Leakage Control

- `target_next_day`는 지자체별 `date + 1 day`가 실제 존재할 때만 생성했습니다.
- 날짜 공백이 있는 경우 다음 관측일을 다음날로 연결하지 않습니다.
- 서비스용 prediction 함수는 target이 없는 `next_day_service_features.csv`만 읽습니다.
- 같은 날/다음날 `discharge_count`, 미래 target, 실제 관측 기상값은 사용하지 않습니다.

## Time Split

Split 기준은 `target_date`입니다.
- train: rows=116,477, municipalities=172, target_date=2021-01-22~2022-12-31
- validation: rows=31,094, municipalities=173, target_date=2023-01-01~2023-06-30
- test: rows=37,831, municipalities=179, target_date=2023-07-01~2024-01-31

## Features

- sigungu_key, total_population, households, population_per_household, target_month, target_day_of_week, target_is_weekend, target_season, waste_amount_t, lag_1, lag_7, rolling_mean_7, rolling_mean_14

## Baselines And Validation Comparison

| model | split | rows | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- |
| random_forest | validation | 31094 | 875,108.2957 | 2,386,739.1875 | 0.9805 |
| hist_gradient_boosting | validation | 31094 | 908,864.2514 | 2,293,869.3900 | 0.9820 |
| weekly_persistence | validation | 31094 | 1,088,909.2444 | 2,811,871.6817 | 0.9730 |
| ridge | validation | 31094 | 1,620,968.5619 | 3,823,175.9557 | 0.9501 |
| moving_average_7 | validation | 31094 | 1,621,066.8478 | 4,131,829.0743 | 0.9417 |
| persistence_today | validation | 31094 | 2,058,624.6428 | 5,868,682.6691 | 0.8823 |

## Final Service Model

- Selected model: `random_forest`
- Test MAE: 1,036,144.92
- Test RMSE: 2,590,857.82
- Test R2: 0.9794
- Test mean target: 15,107,815.69
- MAE / mean target: 6.86%

## Municipality Performance

| sigungu_key | rows | mean_target_next_day | mae | rmse | mae_to_mean_ratio |
| --- | --- | --- | --- | --- | --- |
| 경기도 화성시 | 215 | 94,739,099.7256 | 7,629,973.5386 | 11,044,383.5773 | 0.0805 |
| 경기도 용인시 | 215 | 107,233,101.9953 | 6,668,885.1823 | 9,617,520.4024 | 0.0622 |
| 강원도 동해시 | 215 | 25,273,903.9349 | 6,045,332.7223 | 17,562,388.1418 | 0.2392 |
| 경기도 하남시 | 215 | 47,697,780.2326 | 5,370,844.5163 | 7,515,146.2557 | 0.1126 |
| 전라북도 익산시 | 215 | 99,009,051.1442 | 5,128,008.2454 | 9,089,891.8810 | 0.0518 |
| 충청북도 청주시 | 215 | 53,326,379.2047 | 3,826,167.3231 | 5,247,575.3712 | 0.0717 |
| 경상남도 김해시 | 215 | 50,792,467.5302 | 3,333,091.6344 | 4,684,319.0665 | 0.0656 |
| 서울특별시 송파구 | 215 | 53,796,316.5209 | 3,018,059.3634 | 4,304,981.8383 | 0.0561 |
| 경기도 남양주시 | 215 | 42,337,229.1488 | 2,896,660.9973 | 3,971,090.9536 | 0.0684 |
| 제주특별자치도 제주시 | 215 | 53,928,087.3628 | 2,883,408.6725 | 4,366,251.9371 | 0.0535 |
| 경기도 수원시 영통구 | 215 | 43,346,606.2465 | 2,882,315.5922 | 3,945,251.9273 | 0.0665 |
| 인천광역시 서구 | 215 | 46,678,294.7581 | 2,877,608.4549 | 4,174,130.8322 | 0.0616 |
| 인천광역시 남동구 | 215 | 42,623,216.9442 | 2,815,809.8201 | 4,195,806.8716 | 0.0661 |
| 서울특별시 노원구 | 215 | 47,594,351.8744 | 2,811,149.3717 | 3,823,222.3838 | 0.0591 |
| 대구광역시 달서구 | 215 | 42,497,456.6977 | 2,774,964.9013 | 4,151,625.5588 | 0.0653 |
| 경상북도 구미시 | 215 | 38,079,727.1628 | 2,730,161.2843 | 3,893,255.8420 | 0.0717 |
| 강원도 원주시 | 215 | 32,570,069.1767 | 2,294,181.1531 | 3,160,848.9406 | 0.0704 |
| 부산광역시 해운대구 | 215 | 33,903,432.0465 | 2,289,488.3362 | 3,357,757.7954 | 0.0675 |
| 경기도 부천시 | 215 | 35,053,407.1163 | 2,287,231.7150 | 3,357,390.3805 | 0.0652 |
| 경기도 수원시 권선구 | 215 | 35,854,163.4047 | 2,163,373.5196 | 3,015,725.0332 | 0.0603 |

## Size Group Performance

| size_group | municipalities | mean_target_next_day | mean_mae | mean_mae_to_mean_ratio |
| --- | --- | --- | --- | --- |
| small | 60 | 1,408,247.5289 | 121,203.2048 | 0.1332 |
| medium | 59 | 9,994,320.1850 | 678,035.3909 | 0.0685 |
| large | 60 | 33,076,999.4005 | 2,251,757.0161 | 0.0688 |

## Surge Threshold Candidates

| candidate | attention_threshold_pct | high_threshold_pct | normal_rate | attention_rate | high_rate |
| --- | --- | --- | --- | --- | --- |
| q70_q90 | 4.6174 | 22.0700 | 0.7000 | 0.2000 | 0.1000 |
| q75_q90 | 7.2448 | 22.0700 | 0.7500 | 0.1500 | 0.1000 |
| q80_q95 | 10.5699 | 30.2741 | 0.8000 | 0.1500 | 0.0500 |

## Feature Importance

| feature | baseline_mae | permuted_mae | importance_mae_increase |
| --- | --- | --- | --- |
| rolling_mean_7 | 1,025,191.6630 | 16,300,467.4067 | 15,275,275.7437 |
| waste_amount_t | 1,025,191.6630 | 2,061,781.5219 | 1,036,589.8589 |
| target_day_of_week | 1,025,191.6630 | 1,909,068.5362 | 883,876.8732 |
| lag_1 | 1,025,191.6630 | 1,536,968.8915 | 511,777.2285 |
| rolling_mean_14 | 1,025,191.6630 | 1,470,328.6951 | 445,137.0321 |
| lag_7 | 1,025,191.6630 | 1,292,879.0006 | 267,687.3376 |
| households | 1,025,191.6630 | 1,288,808.5892 | 263,616.9262 |
| total_population | 1,025,191.6630 | 1,196,488.9867 | 171,297.3237 |
| population_per_household | 1,025,191.6630 | 1,077,077.4032 | 51,885.7402 |
| target_is_weekend | 1,025,191.6630 | 1,053,311.8591 | 28,120.1961 |
| sigungu_key | 1,025,191.6630 | 1,042,186.9425 | 16,995.2795 |
| target_month | 1,025,191.6630 | 1,031,422.5166 | 6,230.8537 |
| target_season | 1,025,191.6630 | 1,027,293.8624 | 2,102.1994 |

Feature importance is predictive diagnostic information, not causality.

## Service Usage

Example:

```json
{
  "municipality": "강원도 강릉시",
  "reference_date": "2023-06-30",
  "prediction_date": "2023-07-01",
  "predicted_waste_g": 8958161.513730315,
  "predicted_waste_kg": 8958.161513730316,
  "recent_7day_average_g": 8653148.0,
  "change_vs_recent_average_pct": 3.5248849751595035,
  "surge_level": "normal",
  "model_version": "foodzero-next-day-rf-v1"
}
```

## Limitations

이번 서비스용 1차 모델은 기상변수를 제외했습니다. 향후 예보 데이터 또는 lagged weather feature를 추가해 service-time availability를 지키는 weather 모델을 별도로 비교할 수 있습니다.
