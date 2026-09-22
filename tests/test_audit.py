from __future__ import annotations

import unittest

import pandas as pd

from src.foodzero.audit import profile_dataframe


class AuditTests(unittest.TestCase):
    def test_profile_dataframe_records_columns_missing_and_samples(self) -> None:
        df = pd.DataFrame({"시군구": ["서울 종로구", ""], "배출량": [1, None]})
        profile = profile_dataframe(df)
        self.assertEqual(profile["rows_loaded"], 2)
        self.assertEqual(profile["columns"], ["시군구", "배출량"])
        self.assertEqual(profile["missing_by_column"]["배출량"], 1)
        self.assertEqual(profile["blank_string_by_column"]["시군구"], 1)
        self.assertIn("서울 종로구", profile["sample_values"]["시군구"])


if __name__ == "__main__":
    unittest.main()
