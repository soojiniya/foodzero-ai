from __future__ import annotations

import unittest

import pandas as pd

from src.foodzero.weather import parse_weather_items, valid_station_ids


class WeatherTests(unittest.TestCase):
    def test_parse_weather_items_normalizes_numbers(self) -> None:
        payload = {
            "response": {
                "body": {
                    "items": {
                        "item": [
                            {
                                "tm": "2021-01-01",
                                "stnId": "108",
                                "stnNm": "서울",
                                "avgTa": "-4.2",
                                "maxTa": "1.6",
                                "minTa": "-9.8",
                                "sumRn": "",
                                "avgRhm": "58.1",
                            }
                        ]
                    }
                }
            }
        }
        df = parse_weather_items(payload)
        self.assertEqual(df.loc[0, "date"], pd.Timestamp("2021-01-01"))
        self.assertEqual(df.loc[0, "avgTa"], -4.2)
        self.assertTrue(pd.isna(df.loc[0, "sumRn"]))


    def test_valid_station_ids_uses_current_and_legacy_station_columns(self) -> None:
        mapping = pd.DataFrame(
            {
                "stn_id": ["108", "159", "", "999"],
                "mapping_status": ["validated", "review_required", "validated", "unusable"],
            }
        )
        self.assertEqual(valid_station_ids(mapping), ["108", "159"])


if __name__ == "__main__":
    unittest.main()
