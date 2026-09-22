from __future__ import annotations

import unittest

import pandas as pd

from src.foodzero.features import (
    add_date_features,
    add_lag_features,
    deduplicate_daily_authority,
    merge_population,
    merge_weather,
    merge_weather_mapping,
    parse_population_period,
    parse_population_wide,
    standardize_authority_name,
)


class FeatureTests(unittest.TestCase):
    def test_lag_and_rolling_use_only_past_values(self) -> None:
        df = pd.DataFrame(
            {
                "sigungu_key": ["A"] * 8,
                "date": pd.date_range("2021-01-01", periods=8, freq="D"),
                "waste_amount": [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )
        out = add_lag_features(df)
        self.assertTrue(pd.isna(out.loc[0, "lag_1"]))
        self.assertEqual(out.loc[1, "lag_1"], 10)
        self.assertTrue(pd.isna(out.loc[6, "lag_7"]))
        self.assertEqual(out.loc[7, "lag_7"], 10)
        self.assertEqual(out.loc[7, "rolling_mean_7"], sum([10, 20, 30, 40, 50, 60, 70]) / 7)
        self.assertEqual(out.loc[7, "rolling_mean_14"], sum([10, 20, 30, 40, 50, 60, 70]) / 7)


    def test_date_features(self) -> None:
        df = pd.DataFrame({"date": pd.to_datetime(["2021-01-02", "2021-07-05"])})
        out = add_date_features(df)
        self.assertEqual(out.loc[0, "is_weekend"], 1)
        self.assertEqual(out.loc[0, "season"], "winter")
        self.assertEqual(out.loc[1, "day_of_week"], 0)
        self.assertEqual(out.loc[1, "season"], "summer")
        self.assertEqual(out.loc[1, "year_month"], "2021-07")
        self.assertEqual(out.loc[1, "year"], 2021)
        self.assertEqual(out.loc[1, "month"], 7)
        self.assertEqual(out.loc[1, "day"], 5)


    def test_parse_population_period_from_korean_month_text(self) -> None:
        period = parse_population_period(pd.Series(["2021.01 시군구별 주민등록 인구"]), "fallback.csv")
        self.assertEqual(period.iloc[0], "2021-01")

    def test_parse_population_wide_extracts_monthly_population_and_households(self) -> None:
        df = pd.DataFrame(
            {
                "행정구역": ["서울특별시 종로구 (1111000000)"],
                "2021년01월_총인구수": ["149,125"],
                "2021년01월_세대수": ["75,060"],
            }
        )
        out = parse_population_wide(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "year_month"], "2021-01")
        self.assertEqual(out.loc[0, "sido"], "서울특별시")
        self.assertEqual(out.loc[0, "sigungu"], "종로구")
        self.assertEqual(out.loc[0, "sigungu_key"], "서울특별시 종로구")
        self.assertEqual(out.loc[0, "total_population"], 149125)
        self.assertEqual(out.loc[0, "households"], 75060)

    def test_merge_population_adds_monthly_population_features(self) -> None:
        waste = pd.DataFrame(
            {
                "date": pd.to_datetime(["2021-01-02"]),
                "year_month": ["2021-01"],
                "sigungu_key": ["서울특별시 종로구"],
                "waste_amount": [100],
            }
        )
        population = pd.DataFrame(
            {
                "year_month": ["2021-01"],
                "sigungu_key": ["서울특별시 종로구"],
                "total_population": [149125],
                "households": [75060],
                "population_per_household": [149125 / 75060],
            }
        )
        out = merge_population(waste, population)
        self.assertEqual(out.loc[0, "total_population"], 149125)
        self.assertAlmostEqual(out.loc[0, "population_per_household"], 149125 / 75060)

    def test_merge_weather_mapping_keeps_review_status_and_missing_station(self) -> None:
        df = pd.DataFrame({"sigungu_key": ["A", "B"], "date": pd.to_datetime(["2021-01-01", "2021-01-01"])})
        mapping = pd.DataFrame(
            {
                "standard_sigungu_key": ["A", "B"],
                "station_id": ["108", ""],
                "station_name": ["서울", ""],
                "mapping_status": ["validated", "review_required"],
                "mapping_method": ["inside_municipality", "unresolved_no_verified_station"],
                "confidence": ["high", "review_required"],
                "municipality": ["A", "B"],
                "evidence": ["official", "unresolved"],
                "source_url": ["url", "url"],
                "notes": ["", ""],
            }
        )
        out = merge_weather_mapping(df, mapping)
        self.assertEqual(out.loc[0, "station_id"], "108")
        self.assertEqual(out.loc[0, "weather_mapping_status"], "validated")
        self.assertTrue(pd.isna(out.loc[1, "station_id"]))
        self.assertEqual(out.loc[1, "weather_mapping_status"], "review_required")

    def test_merge_weather_uses_station_and_date_without_cross_station_fill(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
                "station_id": ["108", "159"],
                "sigungu_key": ["A", "B"],
            }
        )
        weather = pd.DataFrame(
            {
                "date": pd.to_datetime(["2021-01-01"]),
                "station_id": ["108"],
                "station_name": ["서울"],
                "avgTa": [-4.2],
                "minTa": [-9.8],
                "maxTa": [1.6],
                "sumRn": [pd.NA],
                "avgRhm": [58.1],
            }
        )
        out = merge_weather(df, weather)
        self.assertEqual(out.loc[0, "avgTa"], -4.2)
        self.assertTrue(pd.isna(out.loc[1, "avgTa"]))

    def test_deduplicate_daily_authority_prevents_duplicate_training_rows(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02"]),
                "sigungu_key": ["A", "A", "A"],
                "waste_amount": [1, 1, 2],
            }
        )
        out = deduplicate_daily_authority(df)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.duplicated(["date", "sigungu_key"]).sum(), 0)

    def test_standardize_authority_name_adds_missing_city_district_space(self) -> None:
        self.assertEqual(
            standardize_authority_name("경기도", "성남시분당구"),
            ("경기도", "성남시 분당구", "경기도 성남시 분당구"),
        )

    def test_standardize_authority_name_normalizes_special_province_names(self) -> None:
        self.assertEqual(
            standardize_authority_name("강원특별자치도", "강릉시"),
            ("강원도", "강릉시", "강원도 강릉시"),
        )
        self.assertEqual(
            standardize_authority_name("전북특별자치도", "전주시"),
            ("전라북도", "전주시", "전라북도 전주시"),
        )


if __name__ == "__main__":
    unittest.main()
