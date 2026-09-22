from __future__ import annotations

import unittest

import pandas as pd

from src.foodzero.modeling_prep import add_calendar_lag_features, date_continuity_report, validate_lag_features


class ModelingPrepTests(unittest.TestCase):
    def test_calendar_lag_does_not_use_previous_observed_day_for_missing_calendar_day(self) -> None:
        df = pd.DataFrame(
            {
                "sigungu_key": ["A", "A", "A"],
                "date": pd.to_datetime(["2021-01-01", "2021-01-03", "2021-01-04"]),
                "waste_amount": [10, 30, 40],
            }
        )
        out = add_calendar_lag_features(df)
        jan3 = out[out["date"].eq(pd.Timestamp("2021-01-03"))].iloc[0]
        jan4 = out[out["date"].eq(pd.Timestamp("2021-01-04"))].iloc[0]
        self.assertTrue(pd.isna(jan3["lag_1"]))
        self.assertEqual(jan4["lag_1"], 30)

    def test_calendar_lag_7_uses_exact_calendar_day(self) -> None:
        df = pd.DataFrame(
            {
                "sigungu_key": ["A"] * 8,
                "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08"]),
                "waste_amount": [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )
        out = add_calendar_lag_features(df)
        self.assertEqual(out.loc[out["date"].eq(pd.Timestamp("2021-01-08")), "lag_7"].iloc[0], 10)

    def test_strict_rolling_requires_full_previous_window_and_excludes_current_target(self) -> None:
        df = pd.DataFrame(
            {
                "sigungu_key": ["A"] * 8,
                "date": pd.date_range("2021-01-01", periods=8, freq="D"),
                "waste_amount": [10, 20, 30, 40, 50, 60, 70, 8000],
            }
        )
        out = add_calendar_lag_features(df)
        self.assertTrue(pd.isna(out.loc[6, "rolling_mean_7"]))
        self.assertEqual(out.loc[7, "rolling_mean_7"], 40)

    def test_date_continuity_report_counts_missing_calendar_days(self) -> None:
        df = pd.DataFrame(
            {
                "sigungu_key": ["A", "A", "B", "B"],
                "date": pd.to_datetime(["2021-01-01", "2021-01-03", "2021-01-01", "2021-01-02"]),
                "waste_amount": [1, 3, 1, 2],
            }
        )
        report = date_continuity_report(df)
        self.assertEqual(report["municipalities_with_gaps"], 1)
        self.assertEqual(report["total_missing_calendar_days"], 1)

    def test_validate_lag_features_detects_mismatches(self) -> None:
        df = pd.DataFrame(
            {
                "sigungu_key": ["A", "A"],
                "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                "waste_amount": [10, 20],
                "lag_1": [pd.NA, 999],
                "lag_7": [pd.NA, pd.NA],
                "rolling_mean_7": [pd.NA, pd.NA],
                "rolling_mean_14": [pd.NA, pd.NA],
            }
        )
        report = validate_lag_features(df)
        self.assertEqual(report["mismatches_against_calendar_lag"]["lag_1"], 1)


if __name__ == "__main__":
    unittest.main()
