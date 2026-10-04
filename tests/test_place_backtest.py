import unittest

import pandas as pd

from hkjc_pipeline import place_top3_backtest


class PlaceBacktestTests(unittest.TestCase):
    def test_local_4_to_6_starters_have_two_places(self):
        frame = pd.DataFrame([
            {"賽事編號": "R1", "馬號": i, "馬名": f"Horse{i}", "model_win_probability": 1.0 / i,
             "market_implied_prob": 1.0 / 6, "名次": str(i), "位置賠率": 2.0}
            for i in range(1, 6)
        ])
        summary, details = place_top3_backtest(frame, stake_per_bet=5.0)
        self.assertEqual(summary["PLACE有效賽事數"], 1)
        self.assertEqual(summary["PLACE模型命中注數"], 2)
        self.assertAlmostEqual(summary["PLACE模型總投注"], 15.0)
        self.assertAlmostEqual(summary["PLACE模型估算總派彩"], 20.0)
        self.assertAlmostEqual(summary["PLACE模型ROI"], 1 / 3)
        third_pick = details.loc[(details["策略"] == "PLACE-模型前三") & (details["選擇次序"] == 3)].iloc[0]
        self.assertEqual(third_pick["是否得位"], 0)

    def test_local_seven_starters_have_three_places_and_dead_heat_counts(self):
        ranks = ["1 平頭馬", "1 平頭馬", "3", "4", "5", "6", "7"]
        frame = pd.DataFrame([
            {"賽事編號": "R2", "馬號": i, "model_win_probability": 1.0 / i,
             "market_implied_prob": 1.0 / 7, "名次": rank, "位置賠率": 2.0}
            for i, rank in enumerate(ranks, start=1)
        ])
        summary, details = place_top3_backtest(frame)
        self.assertEqual(summary["PLACE有效賽事數"], 1)
        self.assertEqual(summary["PLACE模型命中注數"], 3)
        self.assertEqual(summary["PLACE模型選中注數"], 3)
        self.assertEqual(summary["PLACE模型ROI"], 1.0)

    def test_roi_is_withheld_if_any_ticket_lacks_place_odds(self):
        frame = pd.DataFrame([
            {"賽事編號": "R3", "馬號": i, "model_win_probability": 1.0 / i,
             "market_implied_prob": 1.0 / 5, "名次": str(i), "位置賠率": None if i == 1 else 2.0}
            for i in range(1, 6)
        ])
        summary, _ = place_top3_backtest(frame)
        self.assertGreater(summary["PLACE模型命中注數"], 0)
        self.assertIsNone(summary["PLACE模型ROI"])
        self.assertEqual(summary["PLACE缺少有效位置賠率注數"], 1)

    def test_race_with_fewer_than_four_starters_is_excluded(self):
        frame = pd.DataFrame([
            {"賽事編號": "R4", "馬號": i, "model_win_probability": 1.0 / i,
             "market_implied_prob": 0.5, "名次": str(i), "位置賠率": 2.0}
            for i in range(1, 4)
        ])
        summary, details = place_top3_backtest(frame)
        self.assertEqual(summary["PLACE有效賽事數"], 0)
        self.assertEqual(summary["PLACE略過賽事數"], 1)
        self.assertTrue(details.empty)


if __name__ == "__main__":
    unittest.main()
