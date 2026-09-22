from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.foodzero.next_day_modeling import (
    FEATURES,
    TARGET,
    assert_no_forbidden_features,
    assert_target_date_splits_do_not_overlap,
    build_next_day_dataset,
    classify_surge_ratio,
    fit_pipeline,
    predict_next_day,
    split_by_target_date,
)


class NextDayModelingTests(unittest.TestCase):
    def sample_nationwide(self) -> pd.DataFrame:
        dates = pd.date_range("2021-01-01", periods=40, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "sido": ["A"] * len(dates),
                "sigungu": ["B"] * len(dates),
                "sigungu_key": ["A B"] * len(dates),
                "waste_amount": np.arange(len(dates), dtype=float) + 100,
                "discharge_count": np.arange(len(dates), dtype=float),
                "total_population": [1000] * len(dates),
                "households": [400] * len(dates),
                "population_per_household": [2.5] * len(dates),
                "lag_1": np.arange(len(dates), dtype=float) + 90,
                "lag_7": np.arange(len(dates), dtype=float) + 80,
                "rolling_mean_7": np.arange(len(dates), dtype=float) + 70,
                "rolling_mean_14": np.arange(len(dates), dtype=float) + 60,
            }
        )
        return df

    def test_target_next_day_uses_exact_calendar_day_plus_one(self) -> None:
        dataset, _ = build_next_day_dataset(self.sample_nationwide(), write_files=False)
        row = dataset[dataset["date"].eq(pd.Timestamp("2021-01-15"))].iloc[0]
        self.assertEqual(row["target_date"], pd.Timestamp("2021-01-16"))
        self.assertEqual(row[TARGET], 115.0)

    def test_date_gap_does_not_connect_next_observed_day_as_target(self) -> None:
        df = self.sample_nationwide()
        df = df[~df["date"].eq(pd.Timestamp("2021-01-16"))].copy()
        dataset, _ = build_next_day_dataset(df, write_files=False)
        self.assertFalse(dataset["date"].eq(pd.Timestamp("2021-01-15")).any())

    def test_target_date_split_has_no_overlap(self) -> None:
        dates = pd.date_range("2021-01-01", periods=1126, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "target_date": dates + pd.Timedelta(days=1),
                "sigungu_key": ["A"] * len(dates),
            }
        )
        splits = split_by_target_date(df)
        assert_target_date_splits_do_not_overlap(splits)

    def test_forbidden_future_features_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_forbidden_features([*FEATURES, "target_next_day"])
        with self.assertRaises(ValueError):
            assert_no_forbidden_features(["avgTa"])

    def test_surge_level_calculation(self) -> None:
        self.assertEqual(classify_surge_ratio(0.05, 0.1, 0.25), "normal")
        self.assertEqual(classify_surge_ratio(0.15, 0.1, 0.25), "attention")
        self.assertEqual(classify_surge_ratio(0.30, 0.1, 0.25), "high")

    def test_prediction_function_uses_service_features_without_target(self) -> None:
        dataset, service = build_next_day_dataset(self.sample_nationwide(), write_files=False)
        model = fit_pipeline(FEATURES, dataset, "ridge")
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "next_day.joblib"
            service_path = Path(tmpdir) / "service.csv"
            service_no_target = service.drop(columns=[col for col in [TARGET] if col in service.columns])
            service_no_target.to_csv(service_path, index=False, encoding="utf-8-sig")
            from joblib import dump, load

            dump(
                {
                    "model": model,
                    "model_name": "ridge",
                    "model_version": "test",
                    "features": FEATURES,
                    "target": TARGET,
                    "thresholds": {"attention": 0.1, "high": 0.25},
                    "service_features_path": str(service_path),
                },
                model_path,
            )
            loaded = load(model_path)
            self.assertEqual(loaded["features"], FEATURES)
            result = predict_next_day("A B", "2021-01-15", model_path=model_path, service_features_path=service_path)
        self.assertEqual(result["prediction_date"], "2021-01-16")
        self.assertIn(result["surge_level"], {"normal", "attention", "high"})


if __name__ == "__main__":
    unittest.main()
