import unittest

import pandas as pd

from hkjc_pipeline import backtest_prediction_snapshot, place_top3_backtest


class SnapshotBacktestTests(unittest.TestCase):
    def setUp(self):
        self.snapshots = pd.DataFrame([
            {"賽事編號": "20261004-01", "馬號": 1, "馬名": "頭馬甲", "model_win_probability": 0.80, "market_implied_prob": 0.30, "獨贏賠率": 2.0, "位置賠率": 2.0, "賽前快照ID": "S1"},
            {"賽事編號": "20261004-01", "馬號": 2, "馬名": "第三名", "model_win_probability": 0.70, "market_implied_prob": 0.05, "獨贏賠率": 5.0, "位置賠率": 3.0, "賽前快照ID": "S1"},
            {"賽事編號": "20261004-01", "馬號": 3, "馬名": "第七名", "model_win_probability": 0.50, "market_implied_prob": 0.10, "獨贏賠率": 8.0, "位置賠率": 1.5, "賽前快照ID": "S1"},
            {"賽事編號": "20261004-01", "馬號": 4, "馬名": "第四名", "model_win_probability": 0.40, "market_implied_prob": 0.08, "獨贏賠率": 9.0, "位置賠率": 2.5, "賽前快照ID": "S1"},
            {"賽事編號": "20261004-01", "馬號": 5, "馬名": "第五名", "model_win_probability": 0.30, "market_implied_prob": 0.07, "獨贏賠率": 10.0, "位置賠率": 3.0, "賽前快照ID": "S1"},
            {"賽事編號": "20261004-01", "馬號": 6, "馬名": "第六名", "model_win_probability": 0.20, "market_implied_prob": 0.06, "獨贏賠率": 12.0, "位置賠率": 3.5, "賽前快照ID": "S1"},
            {"賽事編號": "20261004-01", "馬號": 7, "馬名": "第二名", "model_win_probability": 0.10, "market_implied_prob": 0.25, "獨贏賠率": 4.0, "位置賠率": 2.5, "賽前快照ID": "S1"},
            {"賽事編號": "20261004-01", "馬號": 8, "馬名": "退出馬", "model_win_probability": 0.60, "market_implied_prob": 0.04, "獨贏賠率": 40.0, "位置賠率": 8.0, "賽前快照ID": "S1"},
            {"賽事編號": "20261004-02", "馬號": 1, "馬名": "五匹賽頭馬", "model_win_probability": 0.90, "market_implied_prob": 0.10, "獨贏賠率": 2.0, "位置賠率": 2.0, "賽前快照ID": "S1"},
            {"賽事編號": "20261004-02", "馬號": 2, "馬名": "五匹賽次名", "model_win_probability": 0.80, "market_implied_prob": 0.15, "獨贏賠率": 3.0, "位置賠率": 3.0, "賽前快照ID": "S1"},
            {"賽事編號": "20261004-02", "馬號": 3, "馬名": "五匹賽第三名", "model_win_probability": 0.70, "market_implied_prob": 0.40, "獨贏賠率": 5.0, "位置賠率": 1.2, "賽前快照ID": "S1"},
            {"賽事編號": "20261004-02", "馬號": 4, "馬名": "五匹賽第四名", "model_win_probability": 0.20, "market_implied_prob": 0.20, "獨贏賠率": 7.0, "位置賠率": 2.0, "賽前快照ID": "S1"},
            {"賽事編號": "20261004-02", "馬號": 5, "馬名": "五匹賽第五名", "model_win_probability": 0.10, "market_implied_prob": 0.15, "獨贏賠率": 12.0, "位置賠率": 3.0, "賽前快照ID": "S1"},
        ])
        self.results = pd.DataFrame([
            *[{"賽事編號": "20261004-01", "馬號": n, "名次": rank, "馬名": name}
              for n, rank, name in [
                  (1, "1 平頭馬", "頭馬甲"), (2, "3", "第三名"), (3, "7", "第七名"),
                  (4, "4", "第四名"), (5, "5", "第五名"), (6, "6", "第六名"),
                  (7, "1 平頭馬", "第二名"), (8, "退出", "退出馬")]],
            *[{"賽事編號": "20261004-02", "馬號": n, "名次": str(n), "馬名": f"五匹賽{n}"}
              for n in range(1, 6)],
        ])

    def test_place_field_size_rules_and_withdrawal(self):
        summary, details, selected, place_details = backtest_prediction_snapshot(
            self.snapshots, self.results, snapshot_id="S1", stake_per_race=10.0, stake_per_place_bet=10.0
        )
        place = summary["PLACE"]
        self.assertEqual(place["PLACE有效賽事數"], 2)
        # Race 1 has seven actual starters (three Place positions); race 2 has five (two positions).
        # The scratched top-three selection is void and is not replaced.
        self.assertEqual(place["PLACE模型選中注數"], 5)
        self.assertEqual(place["PLACE模型命中注數"], 4)
        self.assertEqual(place["PLACE模型至少中一匹場數"], 2)
        self.assertAlmostEqual(place["PLACE模型總投注"], 50.0)
        self.assertAlmostEqual(place["PLACE模型估算總派彩"], 100.0)
        self.assertAlmostEqual(place["PLACE模型ROI"], 1.0)
        race1_model = place_details.loc[(place_details["策略"] == "PLACE-模型前三") & (place_details["賽事編號"] == "20261004-01")]
        self.assertTrue(race1_model["是否退出／作廢"].any())
        self.assertEqual(len(selected), 12)

    def test_missing_place_odds_keeps_hits_but_disables_roi(self):
        data = self.snapshots.copy()
        data.loc[data["馬號"].eq(1) & data["賽事編號"].eq("20261004-01"), "位置賠率"] = None
        summary, _, _, _ = backtest_prediction_snapshot(data, self.results, snapshot_id="S1")
        place = summary["PLACE"]
        self.assertGreater(place["PLACE模型命中注數"], 0)
        self.assertIsNone(place["PLACE模型ROI"])
        self.assertGreater(place["PLACE缺少有效位置賠率注數"], 0)

    def test_multiple_snapshot_ids_require_selection(self):
        both = pd.concat([self.snapshots, self.snapshots.assign(**{"賽前快照ID": "S2"})], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "多個賽前時間"):
            backtest_prediction_snapshot(both, self.results)

    def test_missing_result_row_fails_closed(self):
        incomplete = self.results.loc[~((self.results["賽事編號"] == "20261004-01") & (self.results["馬號"] == 8))]
        with self.assertRaisesRegex(ValueError, "缺少快照內"):
            backtest_prediction_snapshot(self.snapshots, incomplete, snapshot_id="S1")

    def test_target_win_only_cannot_claim_place_hits(self):
        summary, details = place_top3_backtest(pd.DataFrame([
            {"賽事編號": "R1", "馬號": 1, "model_win_probability": 0.9, "target_win": 1}
        ]))
        self.assertIsNone(summary["PLACE模型ROI"])
        self.assertEqual(summary["PLACE模型命中注數"], 0)
        self.assertTrue(details.empty)


if __name__ == "__main__":
    unittest.main()
