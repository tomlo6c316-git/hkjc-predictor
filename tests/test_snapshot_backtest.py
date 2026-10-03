import unittest

import pandas as pd

from hkjc_pipeline import backtest_prediction_snapshot


class SnapshotBacktestTests(unittest.TestCase):
    def setUp(self):
        self.snapshots = pd.DataFrame([
            {"賽事編號": "20261004-01", "馬號": 1, "馬名": "市場熱門", "model_win_probability": 0.10, "獨贏賠率": 2.0, "market_implied_prob": 0.60, "賽前快照ID": "S1", "賽前快照時間(HKT)": "2026-10-04T12:00:00+08:00"},
            {"賽事編號": "20261004-01", "馬號": 2, "馬名": "模型首選", "model_win_probability": 0.80, "獨贏賠率": 5.0, "market_implied_prob": 0.25, "賽前快照ID": "S1", "賽前快照時間(HKT)": "2026-10-04T12:00:00+08:00"},
            {"賽事編號": "20261004-01", "馬號": 3, "馬名": "平頭馬乙", "model_win_probability": 0.05, "獨贏賠率": 8.0, "market_implied_prob": 0.10, "賽前快照ID": "S1", "賽前快照時間(HKT)": "2026-10-04T12:00:00+08:00"},
            {"賽事編號": "20261004-01", "馬號": 4, "馬名": "退出馬", "model_win_probability": 0.03, "獨贏賠率": 40.0, "market_implied_prob": 0.05, "賽前快照ID": "S1", "賽前快照時間(HKT)": "2026-10-04T12:00:00+08:00"},
            {"賽事編號": "20261004-02", "馬號": 1, "馬名": "模型首選乙", "model_win_probability": 0.90, "獨贏賠率": 5.0, "market_implied_prob": 0.20, "賽前快照ID": "S1", "賽前快照時間(HKT)": "2026-10-04T12:00:00+08:00"},
            {"賽事編號": "20261004-02", "馬號": 2, "馬名": "熱門乙", "model_win_probability": 0.10, "獨贏賠率": 2.0, "market_implied_prob": 0.60, "賽前快照ID": "S1", "賽前快照時間(HKT)": "2026-10-04T12:00:00+08:00"},
            {"賽事編號": "20261004-03", "馬號": 1, "馬名": "取消賽首選", "model_win_probability": 0.80, "獨贏賠率": 3.0, "market_implied_prob": 0.50, "賽前快照ID": "S1", "賽前快照時間(HKT)": "2026-10-04T12:00:00+08:00"},
            {"賽事編號": "20261004-03", "馬號": 2, "馬名": "取消賽另一馬", "model_win_probability": 0.20, "獨贏賠率": 4.0, "market_implied_prob": 0.40, "賽前快照ID": "S1", "賽前快照時間(HKT)": "2026-10-04T12:00:00+08:00"},
        ])
        self.results = pd.DataFrame([
            {"賽事編號": "20261004-01", "馬號": 1, "名次": "5", "馬名": "市場熱門"},
            {"賽事編號": "20261004-01", "馬號": 2, "名次": "1 平頭馬", "馬名": "模型首選"},
            {"賽事編號": "20261004-01", "馬號": 3, "名次": "1 平頭馬", "馬名": "平頭馬乙"},
            {"賽事編號": "20261004-01", "馬號": 4, "名次": "退出", "馬名": "退出馬"},
            {"賽事編號": "20261004-02", "馬號": 1, "名次": "1", "馬名": "模型首選乙"},
            {"賽事編號": "20261004-02", "馬號": 2, "名次": "2", "馬名": "熱門乙"},
            {"賽事編號": "20261004-03", "馬號": 1, "名次": "取消", "馬名": "取消賽首選"},
            {"賽事編號": "20261004-03", "馬號": 2, "名次": "取消", "馬名": "取消賽另一馬"},
        ])

    def test_dead_heat_withdrawal_and_void_race(self):
        summary, details, selected = backtest_prediction_snapshot(
            self.snapshots, self.results, snapshot_id="S1", stake_per_race=10.0
        )
        self.assertEqual(summary["可結算賽事數"], 2)
        self.assertEqual(summary["無頭馬／未結算賽事略過數"], 1)
        self.assertEqual(summary["退出／退賽馬匹略過數"], 1)
        self.assertEqual(summary["模型首選命中場數"], 2)
        # R1 dead-heat return: 10 * 5 / 2 = 25; R2: 10 * 5 = 50.
        self.assertAlmostEqual(summary["模型首選估算總派彩"], 75.0)
        self.assertAlmostEqual(summary["模型首選淨盈虧"], 55.0)
        self.assertAlmostEqual(summary["模型首選ROI"], 2.75)
        self.assertEqual(len(selected), 7)
        self.assertEqual(len(details), 2)

    def test_multiple_snapshot_ids_require_selection(self):
        both = pd.concat([self.snapshots, self.snapshots.assign(**{"賽前快照ID": "S2"})], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "多個賽前時間"):
            backtest_prediction_snapshot(both, self.results)

    def test_missing_result_row_fails_closed(self):
        incomplete = self.results.loc[~((self.results["賽事編號"] == "20261004-01") & (self.results["馬號"] == 4))]
        with self.assertRaisesRegex(ValueError, "缺少快照內"):
            backtest_prediction_snapshot(self.snapshots, incomplete, snapshot_id="S1")


if __name__ == "__main__":
    unittest.main()
