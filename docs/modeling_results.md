# Modeling Results

## Datasets

- `weather_validated_dataset.csv`: validated ASOS weather experiment data
- `nationwide_dataset.csv`: nationwide non-weather experiment data

## Time Split

- Train: 2021-01-01 ~ 2022-12-31
- Validation: 2023-01-01 ~ 2023-06-30
- Test: 2023-07-01 ~ 2024-01-31

### weather_with_weather
- train: rows=64,780, municipalities=94, dates=2021-01-15~2022-12-31
- validation: rows=16,862, municipalities=94, dates=2023-01-01~2023-06-30
- test: rows=20,428, municipalities=96, dates=2023-07-01~2024-01-31

### weather_without_weather
- train: rows=64,780, municipalities=94, dates=2021-01-15~2022-12-31
- validation: rows=16,862, municipalities=94, dates=2023-01-01~2023-06-30
- test: rows=20,428, municipalities=96, dates=2023-07-01~2024-01-31

### nationwide
- train: rows=117,761, municipalities=172, dates=2021-01-15~2022-12-31
- validation: rows=31,150, municipalities=173, dates=2023-01-01~2023-06-30
- test: rows=37,930, municipalities=179, dates=2023-07-01~2024-01-31

## Features

- Weather model features: sigungu_key, total_population, households, population_per_household, month, day_of_week, is_weekend, season, lag_1, lag_7, rolling_mean_7, rolling_mean_14, avgTa, minTa, maxTa, avgRhm
- Nationwide model features: sigungu_key, total_population, households, population_per_household, month, day_of_week, is_weekend, season, lag_1, lag_7, rolling_mean_7, rolling_mean_14
- Excluded: `discharge_count`, `sumRn`, future target-derived variables

## Baselines

| experiment | model | target_transform | split | rows | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nationwide | moving_average_7 | none | validation | 31150 | 1,622,856.3234 | 4,137,094.9873 | 0.9416 |
| nationwide | persistence_lag_1 | none | validation | 31150 | 2,060,470.7077 | 5,874,404.3017 | 0.8822 |
| weather_with_weather | moving_average_7 | none | validation | 16862 | 1,765,094.7596 | 4,653,517.2579 | 0.9373 |
| weather_with_weather | persistence_lag_1 | none | validation | 16862 | 2,264,639.8398 | 6,878,394.2786 | 0.8629 |
| weather_without_weather | moving_average_7 | none | validation | 16862 | 1,765,094.7596 | 4,653,517.2579 | 0.9373 |
| weather_without_weather | persistence_lag_1 | none | validation | 16862 | 2,264,639.8398 | 6,878,394.2786 | 0.8629 |

## Validation Model Comparison

| experiment | model | target_transform | split | rows | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nationwide | random_forest | none | validation | 31150 | 890,468.8267 | 2,313,272.3955 | 0.9817 |
| nationwide | random_forest | log1p | validation | 31150 | 910,801.7607 | 2,456,636.8435 | 0.9794 |
| nationwide | hist_gradient_boosting | none | validation | 31150 | 915,326.0376 | 2,319,451.1011 | 0.9816 |
| nationwide | hist_gradient_boosting | log1p | validation | 31150 | 932,984.3582 | 2,448,019.0618 | 0.9795 |
| weather_with_weather | random_forest | none | validation | 16862 | 937,925.5115 | 2,240,532.8885 | 0.9855 |
| weather_with_weather | random_forest | log1p | validation | 16862 | 941,348.5888 | 2,250,097.4918 | 0.9853 |
| weather_without_weather | hist_gradient_boosting | none | validation | 16862 | 959,675.4810 | 2,204,797.7124 | 0.9859 |
| weather_without_weather | random_forest | none | validation | 16862 | 962,694.5847 | 2,280,989.0757 | 0.9849 |
| weather_without_weather | random_forest | log1p | validation | 16862 | 963,464.2250 | 2,274,561.7666 | 0.9850 |
| weather_without_weather | hist_gradient_boosting | log1p | validation | 16862 | 976,423.6584 | 2,315,585.0450 | 0.9845 |
| weather_with_weather | hist_gradient_boosting | none | validation | 16862 | 985,298.0103 | 2,235,923.6423 | 0.9855 |
| weather_with_weather | hist_gradient_boosting | log1p | validation | 16862 | 990,631.1772 | 2,373,088.0116 | 0.9837 |

## Final Model

- Selected by validation MAE: `nationwide / random_forest / none`
- Test MAE: 1,042,677.38
- Test RMSE: 2,709,563.58
- Test R2: 0.9774
- Test mean target: 15,068,541.34
- MAE / mean target: 6.92%

## Weather Effect

Weather effect is evaluated only inside the same `weather_validated_dataset.csv` rows by comparing `weather_with_weather` and `weather_without_weather`. Do not compare it directly with `nationwide`, because the municipality coverage differs.

## Municipality Performance

| sigungu_key | rows | mean_waste_amount | mae | rmse | mae_to_mean_ratio |
| --- | --- | --- | --- | --- | --- |
| 경기도 용인시 | 215 | 107,233,101.9953 | 7,121,532.4876 | 10,190,889.0006 | 0.0664 |
| 경기도 화성시 | 215 | 94,739,099.7256 | 6,459,457.0433 | 8,904,371.5249 | 0.0682 |
| 전라북도 익산시 | 215 | 99,009,051.1442 | 6,140,196.5218 | 10,869,230.0960 | 0.0620 |
| 강원도 동해시 | 215 | 25,273,903.9349 | 5,904,583.5533 | 20,205,930.5237 | 0.2336 |
| 경기도 하남시 | 215 | 47,697,780.2326 | 4,031,493.9444 | 6,419,280.1916 | 0.0845 |
| 충청북도 청주시 | 215 | 53,326,379.2047 | 3,920,408.4156 | 5,428,663.8415 | 0.0735 |
| 경상남도 김해시 | 215 | 50,792,467.5302 | 3,410,721.7171 | 4,958,248.7865 | 0.0672 |
| 서울특별시 송파구 | 215 | 53,796,316.5209 | 3,201,130.6502 | 4,576,519.1667 | 0.0595 |
| 제주특별자치도 제주시 | 215 | 53,928,087.3628 | 3,147,032.1786 | 4,768,378.7151 | 0.0584 |
| 인천광역시 서구 | 215 | 46,678,294.7581 | 3,009,828.7901 | 4,403,945.2545 | 0.0645 |
| 경기도 남양주시 | 215 | 42,337,229.1488 | 2,918,406.9713 | 3,982,541.1889 | 0.0689 |
| 서울특별시 노원구 | 215 | 47,594,351.8744 | 2,890,636.3530 | 3,844,449.4199 | 0.0607 |
| 인천광역시 남동구 | 215 | 42,623,216.9442 | 2,884,852.7341 | 4,314,397.8561 | 0.0677 |
| 경상북도 구미시 | 215 | 38,079,727.1628 | 2,869,547.7278 | 4,159,684.8692 | 0.0754 |
| 경기도 수원시 영통구 | 215 | 43,346,606.2465 | 2,821,026.0905 | 4,003,678.3219 | 0.0651 |
| 대구광역시 달서구 | 215 | 42,497,456.6977 | 2,735,746.0385 | 4,071,989.1688 | 0.0644 |
| 강원도 원주시 | 215 | 32,570,069.1767 | 2,331,195.9982 | 3,235,854.6672 | 0.0716 |
| 부산광역시 해운대구 | 215 | 33,903,432.0465 | 2,328,859.0261 | 3,430,196.8075 | 0.0687 |
| 경기도 부천시 | 215 | 35,053,407.1163 | 2,239,083.7289 | 3,169,616.0558 | 0.0639 |
| 광주광역시 광산구 | 215 | 35,953,491.2791 | 2,217,486.1597 | 2,998,391.4247 | 0.0617 |

## Feature Importance

| feature | baseline_mae | permuted_mae | importance_mae_increase |
| --- | --- | --- | --- |
| lag_7 | 1,023,069.8971 | 10,073,206.4746 | 9,050,136.5775 |
| rolling_mean_7 | 1,023,069.8971 | 3,696,594.5742 | 2,673,524.6771 |
| lag_1 | 1,023,069.8971 | 1,568,375.6544 | 545,305.7574 |
| day_of_week | 1,023,069.8971 | 1,386,857.9624 | 363,788.0653 |
| rolling_mean_14 | 1,023,069.8971 | 1,258,583.2836 | 235,513.3866 |
| households | 1,023,069.8971 | 1,070,584.5212 | 47,514.6242 |
| is_weekend | 1,023,069.8971 | 1,067,085.6972 | 44,015.8002 |
| population_per_household | 1,023,069.8971 | 1,050,102.9122 | 27,033.0152 |
| total_population | 1,023,069.8971 | 1,042,029.9203 | 18,960.0233 |
| sigungu_key | 1,023,069.8971 | 1,028,128.7141 | 5,058.8170 |
| season | 1,023,069.8971 | 1,028,017.1137 | 4,947.2167 |
| month | 1,023,069.8971 | 1,021,159.2618 | -1,910.6352 |

Feature importance is model behavior diagnostics, not causal interpretation.

## Limitations

The weather model uses same-day ASOS observations. These are not known at a real future prediction time, so this experiment is a historical backtest. A production service should use weather forecasts or lagged weather variables.
