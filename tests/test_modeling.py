from __future__ import annotations

import tempfile
import unittest

import numpy as np
import pandas as pd

from src.foodzero.modeling import (
    TARGET,
    assert_no_leakage_features,
    assert_no_split_overlap,
    fit_pipeline,
    metrics,
    predict_original_units,
    split_data,
)


class ModelingTests(unittest.TestCase):
    def sample_modeling_frame(self) -> pd.DataFrame:
        dates = pd.date_range("2021-01-01", periods=1126, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "sigungu_key": ["A"] * len(dates),
                "month": dates.month,
                "day_of_week": dates.dayofweek,
                "is_weekend": dates.dayofweek.isin([5, 6]).astype(int),
                "season": ["winter"] * len(dates),
                "total_population": [1000] * len(dates),
                "households": [400] * len(dates),
                "population_per_household": [2.5] * len(dates),
                "lag_1": np.arange(len(dates), dtype=float) + 1,
                "lag_7": np.arange(len(dates), dtype=float) + 2,
                "rolling_mean_7": np.arange(len(dates), dtype=float) + 3,
                "rolling_mean_14": np.arange(len(dates), dtype=float) + 4,
                TARGET: np.arange(len(dates), dtype=float) + 100,
            }
        )

    def test_train_validation_test_dates_do_not_overlap(self) -> None:
        splits = split_data(self.sample_modeling_frame())
        assert_no_split_overlap(splits)
        self.assertLess(splits["train"]["date"].max(), splits["validation"]["date"].min())
        self.assertLess(splits["validation"]["date"].max(), splits["test"]["date"].min())

    def test_leakage_features_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_leakage_features(["lag_1", "discharge_count"])
        with self.assertRaises(ValueError):
            assert_no_leakage_features(["sumRn"])

    def test_log1p_prediction_restores_original_units(self) -> None:
        df = self.sample_modeling_frame().head(80)
        features = ["sigungu_key", "month", "day_of_week", "season", "lag_1", "rolling_mean_7"]
        spec = {
            "name": "ridge",
            "family": "linear",
            "estimator": __import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge(alpha=1.0),
        }
        model = fit_pipeline(spec, features, df, df[TARGET], "log1p")
        pred = predict_original_units(model, df.head(5), features, "log1p")
        self.assertEqual(len(pred), 5)
        self.assertTrue(np.all(pred >= 0))
        self.assertTrue(np.all(pred > 1))

    def test_fit_pipeline_clones_estimator_between_target_transforms(self) -> None:
        from sklearn.ensemble import RandomForestRegressor

        df = self.sample_modeling_frame().head(120)
        features = ["sigungu_key", "month", "day_of_week", "season", "lag_1", "rolling_mean_7"]
        spec = {
            "name": "random_forest",
            "family": "tree",
            "estimator": RandomForestRegressor(n_estimators=5, random_state=42),
        }
        raw_model = fit_pipeline(spec, features, df, df[TARGET], "none")
        log_model = fit_pipeline(spec, features, df, df[TARGET], "log1p")
        raw_pred = raw_model.predict(df.head(3)[features])
        log_pred = log_model.predict(df.head(3)[features])
        self.assertGreater(raw_pred.mean(), 50)
        self.assertLess(log_pred.mean(), 10)

    def test_metrics_calculation(self) -> None:
        out = metrics(pd.Series([10, 20, 30]), np.array([10, 22, 28]))
        self.assertAlmostEqual(out["MAE"], 4 / 3)
        self.assertGreater(out["R2"], 0.9)
        self.assertGreater(out["RMSE"], 0)

    def test_saved_model_can_load_and_predict(self) -> None:
        from joblib import dump, load
        from sklearn.linear_model import Ridge

        df = self.sample_modeling_frame().head(80)
        features = ["sigungu_key", "month", "day_of_week", "season", "lag_1", "rolling_mean_7"]
        spec = {"name": "ridge", "family": "linear", "estimator": Ridge(alpha=1.0)}
        model = fit_pipeline(spec, features, df, df[TARGET], "none")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model.joblib"
            dump({"model": model, "features": features, "target_transform": "none"}, path)
            payload = load(path)
            pred = predict_original_units(payload["model"], df.head(3), payload["features"], payload["target_transform"])
        self.assertEqual(len(pred), 3)


if __name__ == "__main__":
    unittest.main()
