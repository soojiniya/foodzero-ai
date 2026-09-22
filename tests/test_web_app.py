from __future__ import annotations

import importlib
import unittest

import pandas as pd


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.web_app = importlib.import_module("src.web_app")

    def test_web_app_module_imports(self) -> None:
        self.assertTrue(hasattr(self.web_app, "main"))

    def test_model_loads(self) -> None:
        payload = self.web_app.load_next_day_model()
        self.assertIn("model", payload)
        self.assertIn("features", payload)

    def test_actual_municipality_list_loads(self) -> None:
        municipalities = self.web_app.get_municipalities()
        self.assertGreater(len(municipalities), 100)
        self.assertIn("강원도 강릉시", municipalities)

    def test_prediction_function_connection(self) -> None:
        dates = self.web_app.get_available_reference_dates("강원도 강릉시")
        result = self.web_app.run_prediction("강원도 강릉시", pd.Timestamp(max(dates)))
        self.assertEqual(result["municipality"], "강원도 강릉시")
        self.assertGreater(result["predicted_waste_g"], 0)
        self.assertIn(result["surge_level"], {"normal", "attention", "high"})

    def test_missing_date_returns_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            self.web_app.run_prediction("강원도 강릉시", pd.Timestamp("2099-01-01"))

    def test_unit_conversion(self) -> None:
        self.assertEqual(self.web_app.format_weight(1_250_000), "1.25 t")
        self.assertEqual(self.web_app.format_weight(12_500), "12.5 kg")
        self.assertEqual(self.web_app.format_weight(950), "950 g")

    def test_surge_status_mapping(self) -> None:
        self.assertEqual(self.web_app.normalize_status("normal"), "평소 수준")
        self.assertEqual(self.web_app.normalize_status("attention"), "확인 필요")
        self.assertEqual(self.web_app.normalize_status("high"), "우선 확인")

    def test_dashboard_kpis_are_computed_from_input(self) -> None:
        df = pd.DataFrame(
            {
                "sigungu_key": ["A", "B", "C"],
                "predicted_waste_g": [1000, 2000, 3000],
                "surge_level": ["normal", "attention", "high"],
            }
        )
        kpis = self.web_app.dashboard_kpis(df)
        self.assertEqual(kpis["municipalities"], 3)
        self.assertEqual(kpis["predicted_total_g"], 6000)
        self.assertEqual(kpis["surge_regions"], 2)
        self.assertEqual(kpis["priority_regions"], 1)

    def test_cache_data_function_available(self) -> None:
        features = self.web_app.load_service_features()
        self.assertGreater(len(features), 1000)
        self.assertIn("target_date", features.columns)

    def test_no_placeholder_fake_kpi_labels(self) -> None:
        source = self.web_app.Path(self.web_app.__file__).read_text(encoding="utf-8")
        forbidden = ["1,234", "999,999", "샘플 KPI", "테스트 지자체"]
        self.assertFalse(any(token in source for token in forbidden))


if __name__ == "__main__":
    unittest.main()
