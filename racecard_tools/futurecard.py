"""Create a future HKJC racecard CSV with the model's 19 training features.

Usage:
  python futurecard.py --date 2026-10-01 --venue ST --history seasons_hkjc_enriched.csv
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hkjc_get_racecard import download_racecard  # noqa: E402
from hkjc_extra_features import add_horse_rest_and_course_features  # noqa: E402

FEATURE_COLS = [
    "market_implied_prob", "獨贏賠率", "odds_rank", "is_favorite",
    "排位檔位", "weight_diff", "weight_rank", "jockey_win_rate",
    "trainer_win_rate", "combo_win_rate", "horse_win_rate", "horse_last_rank",
    "距離", "horse_surface_win_rate", "horse_dist_win_rate",
    "days_since_last_race", "horse_course_starts", "horse_course_win_rate",
    "horse_course_top3_rate",
]
BASE_WIN_RATE = 0.10
PRIOR_STRENGTH = 10.0


def read_csv(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp950", on_bad_lines="skip")


def date_from_race_id(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.extract(r"(20\d{6})", expand=False)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def entity_keys(frame: pd.DataFrame, cols: list[str]) -> list[tuple[str, ...]]:
    values = []
    for col in cols:
        if col not in frame:
            values.append(["__UNKNOWN__"] * len(frame))
            continue
        s = frame[col].astype("string").str.strip()
        values.append(s.mask(s.isna() | s.eq(""), "__UNKNOWN__").astype(str).tolist())
    return list(zip(*values))


def prior_win_rate(history: pd.DataFrame, target: pd.DataFrame, cols: list[str], base: float) -> pd.Series:
    out = pd.Series(base, index=target.index, dtype=float)
    if history.empty or target.empty:
        return out
    h = history.loc[history["_date"].notna()].copy()
    t = target.loc[target["_date"].notna()].copy()
    if h.empty or t.empty:
        return out
    h["_key"] = entity_keys(h, cols)
    t["_key"] = entity_keys(t, cols)
    daily = (
        h.groupby(["_key", "_date"], sort=True, observed=True)
        .agg(wins=("_win", "sum"), starts=("_win", "size"))
        .reset_index()
        .sort_values(["_key", "_date"])
    )
    daily["cum_wins"] = daily.groupby("_key", sort=False)["wins"].cumsum()
    daily["cum_starts"] = daily.groupby("_key", sort=False)["starts"].cumsum()
    lookup = {key: group for key, group in daily.groupby("_key", sort=False)}
    for key, positions in t.groupby("_key", sort=False).groups.items():
        g = lookup.get(key)
        if g is None or g.empty:
            continue
        pos = np.searchsorted(
            g["_date"].to_numpy(dtype="datetime64[ns]"),
            t.loc[positions, "_date"].to_numpy(dtype="datetime64[ns]"),
            side="left",
        ) - 1
        valid = pos >= 0
        if valid.any():
            row_idx = np.asarray(positions)[valid]
            wins = g["cum_wins"].to_numpy(dtype=float)[pos[valid]]
            starts = g["cum_starts"].to_numpy(dtype=float)[pos[valid]]
            out.loc[row_idx] = (wins + base * PRIOR_STRENGTH) / (starts + PRIOR_STRENGTH)
    return out


def prior_last_rank(history: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    out = pd.Series(6.0, index=target.index, dtype=float)
    if history.empty or target.empty:
        return out
    h = history.loc[history["_date"].notna()].copy()
    t = target.loc[target["_date"].notna()].copy()
    h["_key"] = entity_keys(h, ["_horse_key"])
    t["_key"] = entity_keys(t, ["_horse_key"])
    daily = (
        h.groupby(["_key", "_date"], observed=True, as_index=False)
        .agg(day_rank=("_rank", "min"))
        .sort_values(["_key", "_date"])
    )
    lookup = {key: group for key, group in daily.groupby("_key", sort=False)}
    for key, positions in t.groupby("_key", sort=False).groups.items():
        g = lookup.get(key)
        if g is None or g.empty:
            continue
        pos = np.searchsorted(
            g["_date"].to_numpy(dtype="datetime64[ns]"),
            t.loc[positions, "_date"].to_numpy(dtype="datetime64[ns]"),
            side="left",
        ) - 1
        valid = pos >= 0
        if valid.any():
            out.loc[np.asarray(positions)[valid]] = g["day_rank"].to_numpy(dtype=float)[pos[valid]]
    return out


def build_future_features(history_raw: pd.DataFrame, card_raw: pd.DataFrame) -> pd.DataFrame:
    required_history = ["賽事編號", "馬號", "馬名", "名次", "獨贏賠率", "騎師", "練馬師", "場地", "距離", "racecourse_code", "official_distance_m"]
    missing = [c for c in required_history if c not in history_raw.columns]
    if missing:
        raise ValueError(f"歷史 enriched CSV 缺少欄位：{missing}")
    required_card = ["賽事編號", "馬號", "馬名", "騎師", "練馬師", "排位檔位", "實際負磅", "距離", "場地", "獨贏賠率", "racecourse_code", "official_distance_m", "名次"]
    missing = [c for c in required_card if c not in card_raw.columns]
    if missing:
        raise ValueError(f"新賽卡缺少欄位：{missing}")

    card = card_raw.copy().reset_index(drop=True)
    card["_is_futurecard"] = True
    hist = history_raw.copy().reset_index(drop=True)
    hist["_is_futurecard"] = False
    all_rows = pd.concat([hist, card], ignore_index=True, sort=False)
    all_rows["賽事編號"] = all_rows["賽事編號"].astype(str).str.strip()
    all_rows["_date"] = date_from_race_id(all_rows["賽事編號"])
    if all_rows["_date"].isna().any():
        raise ValueError("有賽事編號不能解析 YYYYMMDD 日期，請檢查 CSV。")

    # Apply the same stable-entity fallbacks as training: horse name when no
    # permanent horse ID is supplied, and raw jockey/trainer names otherwise.
    horse_id_col = next((c for c in ["馬匹編號", "馬匹代號", "horse_id", "horse_code"] if c in all_rows.columns), "馬名")
    all_rows["_horse_key"] = all_rows[horse_id_col].astype("string").str.strip().fillna("__UNKNOWN__")
    all_rows["_jockey_key"] = all_rows["騎師"].astype("string").str.strip().fillna("__UNKNOWN__")
    all_rows["_trainer_key"] = all_rows["練馬師"].astype("string").str.strip().fillna("__UNKNOWN__")
    all_rows["_venue_key"] = all_rows["場地"].astype("string").str.strip().fillna("__UNKNOWN__")

    rank_text = all_rows["名次"].astype("string").str.strip()
    all_rows["_has_result"] = rank_text.notna() & rank_text.ne("")
    all_rows["_rank"] = pd.to_numeric(all_rows["名次"], errors="coerce").fillna(99.0)
    all_rows["_win"] = (all_rows["_rank"] == 1).astype(int)
    odds = pd.to_numeric(all_rows["獨贏賠率"], errors="coerce")
    valid_history = all_rows.loc[all_rows["_has_result"] & odds.gt(1.0)].copy()
    valid_history = valid_history.drop_duplicates(["賽事編號", "馬號"], keep="last")
    targets = all_rows.loc[all_rows["_is_futurecard"]].copy()
    if targets.empty:
        raise ValueError("沒有賽卡預測列。")

    target_odds = pd.to_numeric(targets["獨贏賠率"], errors="coerce")
    missing_odds = target_odds.isna() | target_odds.le(1.0)
    if missing_odds.any():
        raise ValueError(
            f"有 {int(missing_odds.sum())} 匹馬沒有有效獨贏賠率。"
            "請等待 HKJC 開盤後再執行；不要用虛構賠率填補。"
        )

    targets["獨贏賠率"] = target_odds
    targets["market_prob_raw"] = 1.0 / target_odds
    prob_sum = targets.groupby("賽事編號", observed=True)["market_prob_raw"].transform("sum")
    targets["market_implied_prob"] = (targets["market_prob_raw"] / prob_sum.replace(0, np.nan)).fillna(0.0)
    targets["odds_rank"] = targets.groupby("賽事編號", observed=True)["獨贏賠率"].rank(method="min", ascending=True)
    targets["is_favorite"] = (targets["odds_rank"] == 1).astype(int)
    targets["排位檔位"] = pd.to_numeric(targets["排位檔位"], errors="coerce").fillna(7.0)
    targets["距離"] = pd.to_numeric(targets["距離"], errors="coerce").fillna(1200.0)
    targets["實際負磅"] = pd.to_numeric(targets["實際負磅"], errors="coerce")
    targets["實際負磅"] = targets["實際負磅"].fillna(targets.groupby("賽事編號")["實際負磅"].transform("mean")).fillna(120.0)
    mean_weight = targets.groupby("賽事編號", observed=True)["實際負磅"].transform("mean")
    targets["weight_diff"] = targets["實際負磅"] - mean_weight
    targets["weight_rank"] = targets.groupby("賽事編號", observed=True)["實際負磅"].rank(ascending=False, method="min")

    targets["jockey_win_rate"] = prior_win_rate(valid_history, targets, ["_jockey_key"], 0.10)
    targets["trainer_win_rate"] = prior_win_rate(valid_history, targets, ["_trainer_key"], 0.10)
    targets["combo_win_rate"] = (targets["jockey_win_rate"] + targets["trainer_win_rate"]) / 2.0
    targets["horse_win_rate"] = prior_win_rate(valid_history, targets, ["_horse_key"], 0.10)
    targets["horse_last_rank"] = prior_last_rank(valid_history, targets)
    targets["horse_surface_win_rate"] = prior_win_rate(valid_history, targets, ["_horse_key", "_venue_key"], 0.08)

    distance_group = pd.cut(
        targets["距離"], bins=[0, 1200, 1600, 2000, 2400, 10000],
        labels=["<=1200", "1201-1600", "1601-2000", "2001-2400", ">2400"], include_lowest=True,
    ).astype("string").fillna("unknown")
    targets["_dist_group"] = distance_group
    hist_distance = pd.to_numeric(valid_history.get("距離", pd.Series(1200.0, index=valid_history.index)), errors="coerce").fillna(1200.0)
    valid_history = valid_history.copy()
    valid_history["_dist_group"] = pd.cut(
        hist_distance, bins=[0, 1200, 1600, 2000, 2400, 10000],
        labels=["<=1200", "1201-1600", "1601-2000", "2001-2400", ">2400"], include_lowest=True,
    ).astype("string").fillna("unknown")
    targets["horse_dist_win_rate"] = prior_win_rate(valid_history, targets, ["_horse_key", "_dist_group"], 0.08)

    # Use the same as-of helper that generated the historical enriched data.
    extras, _ = add_horse_rest_and_course_features(all_rows, verbose=False)
    extra_cols = ["days_since_last_race", "horse_course_starts", "horse_course_win_rate", "horse_course_top3_rate"]
    future_extras = extras.loc[extras["_is_futurecard"], extra_cols].reset_index(drop=True)
    targets = targets.reset_index(drop=True)
    for col in extra_cols:
        targets[col] = pd.to_numeric(future_extras[col], errors="coerce")

    for col in FEATURE_COLS:
        if col not in targets:
            raise ValueError(f"計算後缺少模型特徵：{col}")
        targets[col] = pd.to_numeric(targets[col], errors="coerce")
    defaults = {
        "days_since_last_race": 999.0,
        "horse_course_starts": 0.0,
        "horse_course_win_rate": 0.10,
        "horse_course_top3_rate": 0.30,
    }
    for col, default in defaults.items():
        targets[col] = targets[col].fillna(default)
    targets[FEATURE_COLS] = targets[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    targets["名次"] = ""
    return targets.drop(columns=["_is_futurecard", "_date", "_horse_key", "_jockey_key", "_trainer_key", "_venue_key", "_has_result", "_rank", "_win", "market_prob_raw", "_dist_group"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="下載 HKJC 未開跑賽卡並計算19項模型特徵")
    parser.add_argument("--date", required=True, help="賽日 YYYY-MM-DD")
    parser.add_argument("--venue", required=True, choices=["ST", "HV"], help="ST 沙田 / HV 跑馬地")
    parser.add_argument("--history", required=True, help="過往賽果 enriched CSV；只在本機用於算歷史特徵")
    parser.add_argument("--output", default=None, help="輸出 CSV；預設 futurecard_日期_場地_19features.csv")
    args = parser.parse_args()
    date_text = pd.to_datetime(args.date).strftime("%Y-%m-%d")
    out_path = Path(args.output or f"futurecard_{date_text.replace('-', '')}_{args.venue}_19features.csv")
    history = read_csv(args.history)
    with tempfile.TemporaryDirectory(prefix="futurecard-") as tmp:
        raw_path = Path(tmp) / "racecard_raw.csv"
        card = download_racecard(date_text, args.venue, raw_path)
        result = build_future_features(history, card)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"輸出：{out_path.resolve()}")
    print(f"場次：{result['賽事編號'].nunique()}；馬匹：{len(result)}；特徵數：{len(FEATURE_COLS)}")
    print("此 CSV 可直接上傳至已更新的 Streamlit app；不必把歷史 enriched CSV 上傳至 Streamlit。")


if __name__ == "__main__":
    main()
