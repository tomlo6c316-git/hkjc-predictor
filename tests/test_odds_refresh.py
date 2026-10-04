import unittest
from unittest.mock import patch

import pandas as pd

from hkjc_pipeline import refresh_market_odds


class OddsRefreshTests(unittest.TestCase):
    def test_refresh_updates_quotes_and_market_features_only(self):
        card = pd.DataFrame([
            {"賽事編號": "20261004-01", "馬號": 1, "馬名": "甲", "獨贏賠率": 3.0, "位置賠率": 1.5, "market_implied_prob": 0.5714, "odds_rank": 1.0, "is_favorite": 1, "racecourse_code": "ST", "jockey_win_rate": 0.22},
            {"賽事編號": "20261004-01", "馬號": 2, "馬名": "乙", "獨贏賠率": 4.0, "位置賠率": 1.8, "market_implied_prob": 0.4286, "odds_rank": 2.0, "is_favorite": 0, "racecourse_code": "ST", "jockey_win_rate": 0.13},
        ])
        odds = {
            (1, 1): {"獨贏賠率": 6.0, "位置賠率": 2.0},
            (1, 2): {"獨贏賠率": 2.0, "位置賠率": 1.2},
        }
        with patch("hkjc_pipeline.fetch_current_odds", return_value=(odds, "16:10")):
            refreshed, updated_at = refresh_market_odds(card)
        self.assertEqual(updated_at, "16:10")
        self.assertEqual(refreshed["獨贏賠率"].tolist(), [6.0, 2.0])
        self.assertEqual(refreshed["位置賠率"].tolist(), [2.0, 1.2])
        self.assertEqual(refreshed["odds_rank"].tolist(), [2.0, 1.0])
        self.assertEqual(refreshed["is_favorite"].tolist(), [0, 1])
        self.assertAlmostEqual(refreshed["market_implied_prob"].sum(), 1.0)
        self.assertEqual(refreshed["jockey_win_rate"].tolist(), card["jockey_win_rate"].tolist())
        self.assertEqual(card["獨贏賠率"].tolist(), [3.0, 4.0], "input racecard must remain unchanged")


if __name__ == "__main__":
    unittest.main()
