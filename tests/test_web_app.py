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

    def test_management_status_mapping(self) -> None:
        self.assertEqual(self.web_app.normalize_management_status("normal"), "안정")
        self.assertEqual(self.web_app.normalize_management_status("attention"), "주의")
        self.assertEqual(self.web_app.normalize_management_status("high"), "집중관리")

    def test_change_vs_recent_average_handles_zero_denominator(self) -> None:
        value = self.web_app.calculate_change_vs_recent_average_pct(1000, 0)
        self.assertTrue(pd.isna(value))

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

    def test_management_kpis_are_computed_from_input(self) -> None:
        df = pd.DataFrame(
            {
                "sigungu_key": ["A", "B", "C"],
                "surge_level": ["normal", "attention", "high"],
            }
        )
        kpis = self.web_app.management_kpis(df)
        self.assertEqual(kpis["municipalities"], 3)
        self.assertEqual(kpis["normal"], 1)
        self.assertEqual(kpis["attention"], 1)
        self.assertEqual(kpis["high"], 1)

    def test_management_predictions_use_single_reference_date_and_rank_order(self) -> None:
        dates = self.web_app.get_available_reference_dates()
        reference_date = pd.Timestamp(max(dates))
        predictions = self.web_app.prepare_management_predictions(reference_date)
        self.assertGreater(len(predictions), 100)
        self.assertEqual(predictions["date"].nunique(), 1)
        self.assertEqual(pd.Timestamp(predictions["date"].iloc[0]), reference_date)
        self.assertIn("mae_to_mean_ratio", predictions.columns)
        self.assertEqual(predictions["management_rank"].tolist(), list(range(1, len(predictions) + 1)))
        order_tuples = list(
            zip(
                predictions["risk_order"].tolist(),
                (-predictions["change_vs_recent_average_pct"].fillna(float("-inf"))).tolist(),
                (-predictions["predicted_waste_g"]).tolist(),
            )
        )
        self.assertEqual(order_tuples, sorted(order_tuples))

    def test_priority_management_predictions_exclude_normal_regions(self) -> None:
        dates = self.web_app.get_available_reference_dates()
        reference_date = pd.Timestamp(max(dates))
        predictions = self.web_app.prepare_management_predictions(reference_date)
        priority = self.web_app.priority_management_predictions(predictions)
        kpis = self.web_app.management_kpis(predictions)
        self.assertEqual(len(priority), 6)
        self.assertEqual(kpis["high"] + kpis["attention"], 6)
        self.assertEqual((kpis["high"], kpis["attention"], kpis["normal"], kpis["municipalities"]), (2, 4, 172, 178))
        self.assertSetEqual(set(priority["surge_level"]), {"high", "attention"})
        self.assertFalse(priority["surge_level"].eq("normal").any())

    def test_cache_data_function_available(self) -> None:
        features = self.web_app.load_service_features()
        self.assertGreater(len(features), 1000)
        self.assertIn("target_date", features.columns)

    def test_no_placeholder_fake_kpi_labels(self) -> None:
        source = self.web_app.Path(self.web_app.__file__).read_text(encoding="utf-8")
        forbidden = ["1,234", "999,999", "샘플 KPI", "테스트 지자체"]
        self.assertFalse(any(token in source for token in forbidden))

    def test_management_page_is_in_navigation(self) -> None:
        source = self.web_app.Path(self.web_app.__file__).read_text(encoding="utf-8")
        self.assertIn('"배출량 예측", "수거·관리 지원", "데이터 인사이트"', source)
        self.assertIn('elif page == "수거·관리 지원":', source)
        self.assertIn("평상시보다 배출량 증가가 예상되어 우선 확인이 필요한 지역입니다.", source)
        self.assertIn("과거 분포 기준 높은 증가 수준", source)
        self.assertIn("관리 확인 필요", source)
        self.assertIn('class="fz-table management-table"', source)
        self.assertIn(".management-table th", source)
        self.assertIn("지역별 모델 예측 오차 참고 지표입니다.", source)
        self.assertNotIn("train q90 이상", source)
        self.assertNotIn("train q75 이상", source)


if __name__ == "__main__":
    unittest.main()
