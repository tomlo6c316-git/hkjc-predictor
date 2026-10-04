"""Shared data/model operations for the HKJC 19-feature Streamlit workbench."""
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

CORE_PATH = Path(__file__).resolve().parent / "train_hkjc_19feature_time_split.py"
spec = importlib.util.spec_from_file_location("hkjc_19feature_training_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"無法載入19大特徵核心：{CORE_PATH}")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

FEATURE_COLS = list(trainer.FEATURE_COLS)
RACECARD_TOOLS_DIR = Path(__file__).resolve().parent / "racecard_tools"
sys.path.insert(0, str(RACECARD_TOOLS_DIR))
from futurecard import FEATURE_COLS as FUTURECARD_FEATURE_COLS, build_future_features
from hkjc_get_racecard import download_racecard, fetch_current_odds

if FEATURE_COLS != FUTURECARD_FEATURE_COLS:
    raise ValueError("訓練器與 HKJC futurecard 的19項特徵或欄位次序不一致。")

MODEL_PARAMS = {
    "objective": "binary",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read uploaded CSV with the common HKJC encodings."""
    try:
        return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), encoding="cp950", low_memory=False)


def model_bytes(model: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def load_model_bytes(data: bytes) -> Any:
    model = joblib.load(io.BytesIO(data))
    model_features = getattr(model, "feature_name_", None)
    if model_features is None and hasattr(model, "booster_"):
        model_features = model.booster_.feature_name()
    if model_features is not None and list(model_features) != FEATURE_COLS:
        raise ValueError("模型的欄位名稱／順序與此工作台的19大特徵不一致。")
    if not hasattr(model, "predict_proba"):
        raise ValueError("模型不是支援 predict_proba 的分類模型。")
    return model


def prepare_training_data(raw: pd.DataFrame) -> pd.DataFrame:
    return trainer.build_features(raw)


def fetch_and_build_futurecard(
    history: pd.DataFrame,
    target_date: str,
    venue: str,
) -> tuple[pd.DataFrame, str | None, pd.DataFrame]:
    """Download the official HKJC card and create a complete 19-feature CSV."""
    target_day = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    venue = str(venue).upper()
    if venue not in {"ST", "HV"}:
        raise ValueError("馬場只可選 ST（沙田）或 HV（跑馬地）。")
    with tempfile.TemporaryDirectory(prefix="hkjc-streamlit-futurecard-") as tmp:
        raw_path = Path(tmp) / "official_racecard.csv"
        raw_card = download_racecard(target_day, venue, raw_path)
        odds_update_time = raw_card.attrs.get("odds_update_time")
        raw_win_odds = pd.to_numeric(
            raw_card.get("獨贏賠率", pd.Series(np.nan, index=raw_card.index)), errors="coerce"
        )
        missing_raw_odds = raw_win_odds.isna() | raw_win_odds.le(1.0)
        if missing_raw_odds.any():
            columns = [col for col in ["賽事編號", "馬號", "馬名"] if col in raw_card.columns]
            affected = raw_card.loc[missing_raw_odds, columns].head(12)
            affected_text = "；".join(
                f"{row.get('賽事編號', '?')} 馬號{row.get('馬號', '?')} {row.get('馬名', '')}"
                for _, row in affected.iterrows()
            )
            suffix = "" if int(missing_raw_odds.sum()) <= len(affected) else f"；另有 {int(missing_raw_odds.sum()) - len(affected)} 匹未列出"
            raise ValueError(
                f"有 {int(missing_raw_odds.sum())} 匹馬沒有有效獨贏賠率。缺漏：{affected_text}{suffix}。"
                "請到 HKJC 官方獨贏賠率頁確認該馬仍在排位且WIN池已開，稍後再按一次重新抓取。"
                "系統不會用位置賠率、猜測值或其他馬匹賠率代替；本次不會輸出不完整CSV。"
            )
        futurecard = build_future_features(history, raw_card)

    missing = [col for col in FEATURE_COLS if col not in futurecard.columns]
    if missing:
        raise ValueError(f"產生賽卡時缺少19大欄位：{missing}")
    feature_frame = futurecard[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(feature_frame.to_numpy(dtype=float)).all():
        bad = [col for col in FEATURE_COLS if not np.isfinite(feature_frame[col].to_numpy(dtype=float)).all()]
        raise ValueError(f"產生的賽卡有空白或非有限19大特徵：{bad}")
    if futurecard["名次"].astype("string").str.strip().replace("<NA>", "").ne("").any():
        raise ValueError("官方頁回傳資料含已填賽果，為防止賽後資料混入預測已停止。")
    return futurecard, odds_update_time, raw_card


def refresh_market_odds(card: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Refresh only current odds and market-derived features for a saved racecard."""
    required = ["賽事編號", "馬號", "獨贏賠率", "位置賠率", "market_implied_prob", "odds_rank", "is_favorite"]
    missing = [col for col in required if col not in card.columns]
    if missing:
        raise ValueError(f"目前 prediction CSV 缺少賠率更新欄位：{missing}。請重新抓取完整官方賽卡。")
    if card.empty:
        raise ValueError("prediction CSV 沒有馬匹資料，無法更新賠率。")

    dates = card["賽事編號"].astype("string").str.extract(r"(20\d{6})", expand=False).dropna().unique()
    venues = card["racecourse_code"].astype("string").str.upper().dropna().unique() if "racecourse_code" in card.columns else []
    if len(dates) != 1 or len(venues) != 1 or venues[0] not in {"ST", "HV"}:
        raise ValueError("prediction CSV 的賽日或馬場資料不明確，為避免抓錯場次，請重新取得官方排位表。")
    date_text = pd.to_datetime(dates[0], format="%Y%m%d").strftime("%Y-%m-%d")
    venue = str(venues[0])

    session = requests.Session()
    try:
        odds_map, update_time = fetch_current_odds(session, date_text, venue)
    except Exception as exc:
        raise RuntimeError(f"HKJC 即時賠率 API 暫時無法更新：{exc}") from exc

    refreshed = card.copy()
    race_numbers = refreshed["賽事編號"].astype(str).str.rsplit("-", n=1).str[-1]
    race_numbers = pd.to_numeric(race_numbers, errors="coerce")
    horse_numbers = pd.to_numeric(refreshed["馬號"], errors="coerce")
    new_win = []
    new_place = []
    missing_runners = []
    for index, (race_no, horse_no) in enumerate(zip(race_numbers, horse_numbers)):
        values = None
        if pd.notna(race_no) and pd.notna(horse_no):
            values = odds_map.get((int(race_no), int(float(horse_no))))
        win = pd.to_numeric(values.get("獨贏賠率"), errors="coerce") if values else np.nan
        place = pd.to_numeric(values.get("位置賠率"), errors="coerce") if values else np.nan
        if pd.isna(win) or win <= 1.0:
            row = refreshed.iloc[index]
            missing_runners.append(f"{row['賽事編號']} 馬號{row['馬號']} {row.get('馬名', '')}")
        new_win.append(float(win) if pd.notna(win) and win > 1.0 else np.nan)
        new_place.append(float(place) if pd.notna(place) and place > 1.0 else np.nan)

    if missing_runners:
        examples = "；".join(missing_runners[:12])
        extra = f"；另有 {len(missing_runners)-12} 匹未列出" if len(missing_runners) > 12 else ""
        raise ValueError(
            f"本次 HKJC 最新WIN賠率仍未涵蓋 {len(missing_runners)} 匹馬：{examples}{extra}。"
            "沒有更新任何欄位；請稍後再刷新。"
        )

    refreshed["獨贏賠率"] = new_win
    # PLACE quotes are useful UI context, but are not one of the 19 model inputs.
    refreshed["位置賠率"] = [place if pd.notna(place) else old for place, old in zip(new_place, refreshed["位置賠率"])]
    raw_probability = 1.0 / refreshed["獨贏賠率"]
    sums = raw_probability.groupby(refreshed["賽事編號"], observed=True).transform("sum")
    refreshed["market_implied_prob"] = raw_probability / sums
    refreshed["odds_rank"] = refreshed.groupby("賽事編號", observed=True)["獨贏賠率"].rank(
        method="min", ascending=True
    )
    refreshed["is_favorite"] = refreshed["odds_rank"].eq(1).astype(int)
    if "賠率快照時間" in refreshed.columns:
        refreshed["賠率快照時間"] = update_time or "HKJC API 未提供"
    return refreshed, update_time


def _fit_iteration(train: pd.DataFrame, validation: pd.DataFrame) -> int:
    if train["target_win"].nunique() < 2 or validation["target_win"].nunique() < 2:
        raise ValueError("Train 和 Validation 都必須同時包含勝出與未勝出樣本。")
    model = lgb.LGBMClassifier(**MODEL_PARAMS)
    model.fit(
        train[FEATURE_COLS],
        train["target_win"],
        eval_set=[(validation[FEATURE_COLS], validation["target_win"])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)],
    )
    return int(model.best_iteration_ or MODEL_PARAMS["n_estimators"])


def _fit_model(rows: pd.DataFrame, n_estimators: int) -> Any:
    params = dict(MODEL_PARAMS)
    params["n_estimators"] = int(n_estimators)
    model = lgb.LGBMClassifier(**params)
    model.fit(rows[FEATURE_COLS], rows["target_win"])
    return model


def _prob_for_win(model: Any, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("模型未訓練出勝出類別（target_win=1）。")
    p = np.asarray(model.predict_proba(X), dtype=float)[:, classes.index(1)]
    if not np.isfinite(p).all():
        raise ValueError("模型輸出有無效機率值。")
    return p


def _probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
    return {
        "AUC": auc,
        "Brier score": float(brier_score_loss(y, p)),
        "Log loss": float(log_loss(y, p, labels=[0, 1])),
    }


def top1_roi(predictions: pd.DataFrame, stake_per_race: float = 10.0) -> tuple[dict[str, float], pd.DataFrame]:
    """Indicative top-1 win-bet ROI using the CSV's win odds, not official dividends."""
    required = ["賽事編號", "馬號", "target_win", "獨贏賠率", "model_win_probability", "market_implied_prob"]
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise ValueError(f"回測預測結果缺少 ROI 欄位：{missing}")
    if stake_per_race <= 0:
        raise ValueError("每場投注金額必須大於0。")

    frame = predictions.copy()
    frame["model_win_probability"] = pd.to_numeric(frame["model_win_probability"], errors="coerce")
    frame["market_implied_prob"] = pd.to_numeric(frame["market_implied_prob"], errors="coerce")
    frame["獨贏賠率"] = pd.to_numeric(frame["獨贏賠率"], errors="coerce")
    if frame[["model_win_probability", "market_implied_prob", "獨贏賠率"]].isna().any().any():
        raise ValueError("ROI 欄位含缺值，無法完成回測。")
    if (frame["獨贏賠率"] <= 1.0).any():
        raise ValueError("回測 CSV 包含無效獨贏賠率（需大於1.0）。")

    frame["target_win"] = pd.to_numeric(frame["target_win"], errors="coerce").fillna(0).astype(int)
    if not frame["target_win"].isin([0, 1]).all():
        raise ValueError("target_win 只能是0或1。")
    winner_counts = frame.groupby("賽事編號", observed=True)["target_win"].sum()
    no_winner_races = int(winner_counts.eq(0).sum())
    frame["race_winner_count"] = frame["賽事編號"].map(winner_counts)
    # Races with no official winner (e.g. abandoned/cancelled) have no valid
    # win bet and are excluded from stake/ROI. Dead-heat winnings are split.
    frame = frame.loc[frame["race_winner_count"] > 0].copy()
    if frame.empty:
        raise ValueError("所有賽事都沒有有效頭馬，無法計算獨贏 ROI。")

    frame["market_rank"] = frame.groupby("賽事編號", observed=True)["market_implied_prob"].rank(
        ascending=False, method="min"
    )
    model_picks = frame.sort_values(
        ["賽事編號", "model_win_probability", "馬號"],
        ascending=[True, False, True], kind="stable",
    ).groupby("賽事編號", observed=True).head(1).copy()
    market_picks = frame.sort_values(
        ["賽事編號", "market_implied_prob", "馬號"],
        ascending=[True, False, True], kind="stable",
    ).groupby("賽事編號", observed=True).head(1).copy()

    def settle(picks: pd.DataFrame, prefix: str) -> dict[str, float]:
        wins = pd.to_numeric(picks["target_win"], errors="coerce").fillna(0).astype(int)
        split_count = pd.to_numeric(picks["race_winner_count"], errors="coerce").clip(lower=1).to_numpy(float)
        gross = np.where(wins.eq(1), stake_per_race * picks["獨贏賠率"].to_numpy(float) / split_count, 0.0)
        total_stake = float(stake_per_race * len(picks))
        total_return = float(gross.sum())
        profit = total_return - total_stake
        return {
            f"{prefix}命中場數": int(wins.sum()),
            f"{prefix}命中率": float(wins.mean()) if len(wins) else 0.0,
            f"{prefix}總投注": total_stake,
            f"{prefix}估算總派彩": total_return,
            f"{prefix}淨盈虧": profit,
            f"{prefix}ROI": profit / total_stake if total_stake else 0.0,
        }

    model_picks["是否命中"] = pd.to_numeric(model_picks["target_win"]).astype(int)
    market_picks["是否命中"] = pd.to_numeric(market_picks["target_win"]).astype(int)
    model_split = pd.to_numeric(model_picks["race_winner_count"], errors="coerce").clip(lower=1)
    market_split = pd.to_numeric(market_picks["race_winner_count"], errors="coerce").clip(lower=1)
    model_picks["估算淨盈虧"] = np.where(
        model_picks["是否命中"].eq(1), stake_per_race * model_picks["獨贏賠率"] / model_split - stake_per_race, -stake_per_race
    )
    market_picks["估算淨盈虧"] = np.where(
        market_picks["是否命中"].eq(1), stake_per_race * market_picks["獨贏賠率"] / market_split - stake_per_race, -stake_per_race
    )

    for picks in (model_picks, market_picks):
        if "馬名" not in picks.columns:
            picks["馬名"] = ""

    summary = {
        "可結算賽事數": int(frame["賽事編號"].nunique()),
        "無頭馬／未結算賽事略過數": no_winner_races,
    }
    summary.update(settle(model_picks, "模型首選"))
    summary.update(settle(market_picks, "市場熱門"))
    model_report = model_picks[["賽事編號", "馬號", "馬名", "target_win", "獨贏賠率", "model_win_probability", "估算淨盈虧"]].copy()
    model_report = model_report.rename(columns={"馬號": "模型首選馬號", "馬名": "模型首選馬名", "target_win": "模型首選命中"})
    market_report = market_picks[["賽事編號", "馬號", "馬名", "target_win", "獨贏賠率", "market_implied_prob", "估算淨盈虧"]].copy()
    market_report = market_report.rename(columns={"馬號": "市場熱門馬號", "馬名": "市場熱門馬名", "target_win": "市場熱門命中", "market_implied_prob": "市場熱門概率"})
    race_report = model_report.merge(market_report, on="賽事編號", how="outer", suffixes=("_模型", "_市場"))
    winner_rows = frame.loc[frame["target_win"].eq(1)].copy()
    winner_rows["model_rank"] = winner_rows.groupby("賽事編號", observed=True)["model_win_probability"].rank(
        ascending=False, method="min"
    )
    winners = winner_rows.groupby("賽事編號", observed=True).agg(
        實際頭馬馬號=("馬號", lambda x: ", ".join(x.astype(str))),
        實際頭馬馬名=("馬名", lambda x: ", ".join(x.fillna("").astype(str))),
        頭馬模型排名=("model_rank", "min"),
        頭馬模型排名分數=("model_win_probability", "max"),
    ).reset_index()
    race_report = race_report.merge(winners, on="賽事編號", how="left", validate="one_to_one")
    return summary, race_report


def place_top3_backtest(
    predictions: pd.DataFrame,
    stake_per_bet: float = 10.0,
    rank_col: str | None = None,
    withdrawn_col: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Backtest three model/market PLACE selections per eligible local race.

    HKJC local Place qualification: 1st/2nd for 4-6 starters and 1st-3rd
    for 7+ starters. Each selected horse is treated as a separate equal stake.
    Withdrawn selections are refunded and are not replaced by the next-ranked
    runner. Place ROI is only estimated when every non-void ticket has a valid
    saved PLACE quote; it is not official pari-mutuel settlement.
    """
    required = ["賽事編號", "馬號", "model_win_probability"]
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        raise ValueError(f"PLACE回測缺少必要欄位：{missing}")
    if stake_per_bet <= 0:
        raise ValueError("每注PLACE投注金額必須大於0。")

    frame = predictions.copy()
    if rank_col is None:
        rank_col = next((c for c in ["官方名次", "名次", "numeric_rank", "official_rank"] if c in frame.columns), None)
    if rank_col is None:
        # A target_win-only file cannot establish 2nd/3rd-place outcomes.
        empty = {
            "PLACE有效賽事數": 0, "PLACE略過賽事數": int(frame["賽事編號"].nunique()),
            "PLACE模型選中注數": 0, "PLACE模型命中注數": 0, "PLACE模型注命中率": None,
            "PLACE模型至少中一匹場數": 0, "PLACE模型場命中率": None,
            "PLACE模型總投注": 0.0, "PLACE模型估算總派彩": None,
            "PLACE模型淨盈虧": None, "PLACE模型ROI": None,
            "PLACE市場選中注數": 0, "PLACE市場命中注數": 0, "PLACE市場注命中率": None,
            "PLACE市場至少中一匹場數": 0, "PLACE市場場命中率": None,
            "PLACE市場總投注": 0.0, "PLACE市場估算總派彩": None,
            "PLACE市場淨盈虧": None, "PLACE市場ROI": None,
            "PLACE缺少有效位置賠率注數": 0,
            "PLACE狀態": "CSV沒有官方名次；只有target_win無法計算PLACE名次命中。",
        }
        return empty, pd.DataFrame()

    place_odds_col = next((c for c in ["位置賠率", "PLACE賠率", "place_odds", "place_odds_decimal"] if c in frame.columns), None)
    if withdrawn_col is None:
        withdrawn_col = next((c for c in ["__withdrawn", "退出", "退賽", "狀態", "馬匹狀態", "賽果狀態"] if c in frame.columns), None)

    rank_text = frame[rank_col].astype("string").fillna("").str.strip()
    frame["__place_rank"] = pd.to_numeric(rank_text.str.extract(r"^\s*(\d+)", expand=False), errors="coerce")
    if withdrawn_col and withdrawn_col in frame.columns:
        status = frame[withdrawn_col]
        if pd.api.types.is_bool_dtype(status):
            frame["__place_withdrawn"] = status.fillna(False).astype(bool)
        else:
            status_text = status.astype("string").fillna("")
            frame["__place_withdrawn"] = status_text.str.contains(r"退出|退賽|scratched|withdrawn", case=False, regex=True, na=False)
    else:
        frame["__place_withdrawn"] = False

    frame["__place_score_model"] = pd.to_numeric(frame["model_win_probability"], errors="coerce")
    if "market_implied_prob" in frame.columns:
        frame["__place_score_market"] = pd.to_numeric(frame["market_implied_prob"], errors="coerce")
    else:
        odds = pd.to_numeric(frame.get("獨贏賠率", pd.Series(np.nan, index=frame.index)), errors="coerce")
        inverse = 1.0 / odds.where(odds.gt(1.0))
        denominator = inverse.groupby(frame["賽事編號"], observed=True).transform("sum")
        frame["__place_score_market"] = inverse / denominator

    if frame[["__place_score_model", "__place_score_market"]].isna().any().any():
        raise ValueError("PLACE回測的模型／市場排序分數有缺值，無法公平選取前三匹。")
    if place_odds_col:
        frame["__place_odds"] = pd.to_numeric(frame[place_odds_col], errors="coerce")
    else:
        frame["__place_odds"] = np.nan

    ticket_rows: list[dict[str, Any]] = []
    race_count = 0
    skipped = 0
    for race_id, race in frame.groupby("賽事編號", observed=True, sort=False):
        starters = race.loc[~race["__place_withdrawn"]]
        starter_count = int(len(starters))
        if starter_count < 4 or starters["__place_rank"].notna().sum() == 0:
            skipped += 1
            continue
        qualifying_places = 2 if starter_count <= 6 else 3
        race_count += 1
        for strategy, score_col in [("模型", "__place_score_model"), ("市場", "__place_score_market")]:
            # Choose from the recorded card first. If a selected horse was later
            # scratched, its own ticket is void; do not silently replace it.
            picks = race.sort_values([score_col, "馬號"], ascending=[False, True], kind="stable").head(3)
            for selection_order, (_, row) in enumerate(picks.iterrows(), start=1):
                is_void = bool(row["__place_withdrawn"])
                rank = row["__place_rank"]
                hit = bool(pd.notna(rank) and int(rank) <= qualifying_places) and not is_void
                ticket_rows.append({
                    "賽事編號": race_id, "策略": f"PLACE-{strategy}前三",
                    "選擇次序": selection_order, "馬號": row["馬號"], "馬名": row.get("馬名", ""),
                    "官方名次": rank if pd.notna(rank) else row.get(rank_col, ""),
                    "該場合資格得位數": qualifying_places, "是否得位": int(hit),
                    "是否退出／作廢": is_void,
                    "位置賠率": row["__place_odds"],
                    "估算投注": 0.0 if is_void else float(stake_per_bet),
                    "估算派彩": (
                        0.0 if is_void or not hit else
                        (float(stake_per_bet) * float(row["__place_odds"])
                         if pd.notna(row["__place_odds"]) and row["__place_odds"] > 1 else np.nan)
                    ),
                    "模型分數": row["__place_score_model"], "市場分數": row["__place_score_market"],
                })

    details = pd.DataFrame(ticket_rows)
    results: dict[str, Any] = {"PLACE有效賽事數": race_count, "PLACE略過賽事數": skipped}
    for strategy, prefix in [("模型", "PLACE模型"), ("市場", "PLACE市場")]:
        part = details.loc[details["策略"].eq(f"PLACE-{strategy}前三")].copy() if not details.empty else pd.DataFrame()
        if part.empty:
            results.update({f"{prefix}選中注數": 0, f"{prefix}命中注數": 0, f"{prefix}注命中率": None,
                            f"{prefix}至少中一匹場數": 0, f"{prefix}場命中率": None,
                            f"{prefix}總投注": 0.0, f"{prefix}估算總派彩": None,
                            f"{prefix}淨盈虧": None, f"{prefix}ROI": None})
            continue
        valid_tickets = part.loc[~part["是否退出／作廢"]]
        hit_count = int(valid_tickets["是否得位"].sum())
        race_hits = int(valid_tickets.groupby("賽事編號")["是否得位"].max().sum())
        missing_quotes = int((valid_tickets["位置賠率"].isna() | valid_tickets["位置賠率"].le(1.0)).sum())
        total_stake = float(valid_tickets["估算投注"].sum())
        if missing_quotes:
            total_return = profit = roi = None
            state = f"缺少{missing_quotes}注有效位置賠率；保留命中統計，但不計部分ROI。"
        else:
            total_return = float(valid_tickets["估算派彩"].sum())
            profit = total_return - total_stake
            roi = profit / total_stake if total_stake else 0.0
            state = "位置賠率估算派彩；非HKJC官方結算派彩。"
        results.update({
            f"{prefix}選中注數": int(len(valid_tickets)),
            f"{prefix}命中注數": hit_count,
            f"{prefix}注命中率": hit_count / len(valid_tickets) if len(valid_tickets) else 0.0,
            f"{prefix}至少中一匹場數": race_hits,
            f"{prefix}場命中率": race_hits / race_count if race_count else 0.0,
            f"{prefix}總投注": total_stake,
            f"{prefix}估算總派彩": total_return,
            f"{prefix}淨盈虧": profit,
            f"{prefix}ROI": roi,
        })
        if strategy == "模型":
            results["PLACE缺少有效位置賠率注數"] = missing_quotes
            results["PLACE狀態"] = state
        else:
            results["PLACE市場缺少有效位置賠率注數"] = missing_quotes
            results["PLACE市場狀態"] = state
    return results, details


def train_full_model(raw: pd.DataFrame) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Choose iteration on validation data, then refit on all labelled rows."""
    frame = prepare_training_data(raw)
    masks, train_cut, val_cut = trainer.chronological_masks(frame)
    train = frame.loc[masks["train"]].copy()
    validation = frame.loc[masks["validation"]].copy()
    best_iteration = _fit_iteration(train, validation)
    model = _fit_model(frame, best_iteration)
    info = {
        "最佳迭代次數": best_iteration,
        "訓練馬匹數": len(frame),
        "賽事數": int(frame["賽事編號"].nunique()),
        "賽日數": int(frame["_race_date"].nunique()),
        "Train日期截止（不含）": train_cut.strftime("%Y-%m-%d"),
        "Validation日期截止（不含）": val_cut.strftime("%Y-%m-%d"),
        "訓練資料最後日期": frame["_race_date"].max().strftime("%Y-%m-%d"),
    }
    return model, frame, info


def chronological_backtest(
    raw: pd.DataFrame,
    stake_per_race: float = 10.0,
    stake_per_place_bet: float | None = None,
) -> dict[str, Any]:
    """Evaluate an untouched chronological test block for WIN and PLACE."""
    frame = prepare_training_data(raw)
    masks, train_cut, val_cut = trainer.chronological_masks(frame)
    train = frame.loc[masks["train"]].copy()
    validation = frame.loc[masks["validation"]].copy()
    test = frame.loc[masks["test"]].copy()
    best_iteration = _fit_iteration(train, validation)
    model = _fit_model(pd.concat([train, validation], axis=0).sort_values(["_race_date", "賽事編號", "馬號"]), best_iteration)
    probabilities = _prob_for_win(model, test[FEATURE_COLS])
    predicted = test[[c for c in ["賽事編號", "馬季", "馬號", "馬名", "名次", "target_win", "獨贏賠率", "位置賠率", "market_implied_prob"] if c in test.columns]].copy()
    predicted["model_win_probability"] = probabilities
    predicted["model_rank"] = predicted.groupby("賽事編號", observed=True)["model_win_probability"].rank(ascending=False, method="min")
    predicted["market_rank"] = predicted.groupby("賽事編號", observed=True)["market_implied_prob"].rank(ascending=False, method="min")
    roi_summary, race_details = top1_roi(predicted, stake_per_race)
    place_summary, place_details = place_top3_backtest(
        predicted, stake_per_bet=stake_per_place_bet if stake_per_place_bet is not None else stake_per_race
    )
    metrics = {
        "切分": {
            "Train賽日數": int(train["_race_date"].nunique()),
            "Validation賽日數": int(validation["_race_date"].nunique()),
            "Test賽日數": int(test["_race_date"].nunique()),
            "Train截止（不含）": train_cut.strftime("%Y-%m-%d"),
            "Validation截止／Test開始": val_cut.strftime("%Y-%m-%d"),
            "Test最後日期": test["_race_date"].max().strftime("%Y-%m-%d"),
            "Test場數": int(test["賽事編號"].nunique()),
            "Test馬匹數": len(test),
        },
        "模型概率指標": _probability_metrics(test["target_win"].to_numpy(), probabilities),
        "市場概率指標": _probability_metrics(test["target_win"].to_numpy(), test["market_implied_prob"].to_numpy()),
        "最佳迭代次數": best_iteration,
        "ROI": roi_summary,
        "PLACE": place_summary,
    }
    return {"metrics": metrics, "predictions": predicted, "race_details": race_details,
            "place_details": place_details, "model": model}


def predict_racecard(raw: pd.DataFrame, model: Any) -> pd.DataFrame:
    missing = [c for c in ["賽事編號", "馬號", "獨贏賠率", *FEATURE_COLS] if c not in raw.columns]
    if missing:
        raise ValueError(f"賽卡缺少欄位：{missing}。請先準備完整19大特徵 CSV。")
    if raw.empty:
        raise ValueError("預測賽卡沒有馬匹資料。")
    if raw.duplicated(["賽事編號", "馬號"]).any():
        raise ValueError("賽卡有重複的『賽事編號＋馬號』。")
    if "名次" in raw and raw["名次"].astype("string").str.strip().replace("<NA>", "").ne("").any():
        raise ValueError("預測賽卡含已填名次。已完賽資料請使用回測頁。")
    frame = raw.copy()
    X = frame[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        missing = [c for c in FEATURE_COLS if not np.isfinite(X[c].to_numpy(dtype=float)).all()]
        raise ValueError(f"19大特徵含缺值或無限值：{missing}")
    odds = pd.to_numeric(frame["獨贏賠率"], errors="coerce")
    if odds.isna().any() or (odds <= 1).any():
        raise ValueError("賽卡必須有真實有效的獨贏賠率（>1），不會使用假賠率。")
    p = _prob_for_win(model, X)
    frame["model_win_probability"] = p
    totals = frame.groupby("賽事編號", observed=True)["model_win_probability"].transform("sum")
    counts = frame.groupby("賽事編號", observed=True)["賽事編號"].transform("size")
    frame["model_race_probability"] = np.where(totals > 0, frame["model_win_probability"] / totals, 1 / counts)
    frame["model_rank"] = frame.groupby("賽事編號", observed=True)["model_win_probability"].rank(ascending=False, method="min")
    if "market_implied_prob" in frame:
        frame["market_rank"] = frame.groupby("賽事編號", observed=True)["market_implied_prob"].rank(ascending=False, method="min")
    return frame.sort_values(["賽事編號", "model_rank", "馬號"], kind="stable").reset_index(drop=True)


def single_date_backtest(
    saved_predictions: pd.DataFrame,
    target_date: str,
    stake_per_race: float = 10.0,
    stake_per_place_bet: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate one meeting from saved out-of-sample predictions, without refitting."""
    required = ["賽事編號", "馬號", "model_win_probability", "獨贏賠率"]
    missing = [col for col in required if col not in saved_predictions.columns]
    if missing:
        raise ValueError(f"預測明細CSV缺少欄位：{missing}。請上傳包含逐馬模型預測分數的回測輸出。")

    date_key = pd.Timestamp(target_date).strftime("%Y%m%d")
    race_ids = saved_predictions["賽事編號"].astype("string").str.strip()
    selected = saved_predictions.loc[race_ids.str.match(rf"^{date_key}(?:-|$)", na=False)].copy()
    if selected.empty:
        raise ValueError(f"此預測CSV沒有 {target_date} 的賽事。請確認日期在未見測試期內，並使用逐馬預測CSV。")
    if "target_win" not in selected.columns:
        if "名次" not in selected.columns:
            raise ValueError("CSV 必須包含 target_win 或官方名次欄，才能核對賽果。")
        rank_number = pd.to_numeric(
            selected["名次"].astype("string").str.extract(r"^\s*(\d+)", expand=False),
            errors="coerce",
        )
        selected["target_win"] = rank_number.eq(1).astype(int)
    if "market_implied_prob" not in selected.columns:
        odds = pd.to_numeric(selected["獨贏賠率"], errors="coerce")
        if odds.isna().any() or (odds <= 1).any():
            raise ValueError("缺少市場概率，且 CSV 獨贏賠率無效，無法建立市場熱門比較。")
        raw_prob = 1.0 / odds
        denom = raw_prob.groupby(selected["賽事編號"], observed=True).transform("sum")
        selected["market_implied_prob"] = raw_prob / denom
    if "馬名" not in selected.columns:
        selected["馬名"] = ""
    selected["賽事編號"] = selected["賽事編號"].astype(str).str.strip()

    summary_roi, details = top1_roi(selected, stake_per_race)
    place_summary, place_details = place_top3_backtest(
        selected, stake_per_bet=stake_per_place_bet if stake_per_place_bet is not None else stake_per_race
    )
    y = pd.to_numeric(selected["target_win"], errors="coerce").fillna(0).to_numpy(dtype=int)
    model_p = pd.to_numeric(selected["model_win_probability"], errors="coerce").to_numpy(dtype=float)
    market_p = pd.to_numeric(selected["market_implied_prob"], errors="coerce").to_numpy(dtype=float)
    metrics = {
        "日期": target_date,
        "賽事數": int(selected["賽事編號"].nunique()),
        "馬匹數": int(len(selected)),
        "ROI": summary_roi,
        "PLACE": place_summary,
        "模型概率指標": _probability_metrics(y, model_p),
        "市場概率指標": _probability_metrics(y, market_p),
    }
    return metrics, details, selected, place_details


def backtest_prediction_snapshot(
    snapshots: pd.DataFrame,
    official_results: pd.DataFrame,
    snapshot_id: str | None = None,
    stake_per_race: float = 10.0,
    stake_per_place_bet: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join one saved pre-race prediction/odds snapshot to official results.

    Snapshot prices are the quotes recorded at snapshot time, not official
    dividends. A race entry marked withdrawn/scratched in the result file is
    removed before picking; races with no winning finisher are excluded by
    ``top1_roi``.
    """
    required_snapshot = ["賽事編號", "馬號", "model_win_probability", "獨贏賠率"]
    missing = [c for c in required_snapshot if c not in snapshots.columns]
    if missing:
        raise ValueError(f"賽前快照缺少必要欄位：{missing}")
    required_results = ["賽事編號", "馬號"]
    missing = [c for c in required_results if c not in official_results.columns]
    if missing:
        raise ValueError(f"官方賽果 CSV 缺少必要欄位：{missing}")
    if "名次" not in official_results.columns and "target_win" not in official_results.columns:
        raise ValueError("官方賽果 CSV 需包含『名次』或 target_win 欄。")

    frame = snapshots.copy()
    if "賽前快照ID" in frame.columns:
        available = frame["賽前快照ID"].dropna().astype(str).unique().tolist()
        if snapshot_id is None:
            if len(available) != 1:
                raise ValueError("快照檔含多個賽前時間；請指定要回測的『賽前快照ID』。")
            snapshot_id = available[0]
        frame = frame.loc[frame["賽前快照ID"].astype(str).eq(str(snapshot_id))].copy()
    elif snapshot_id is not None:
        raise ValueError("此快照檔沒有『賽前快照ID』欄，不能按快照ID篩選。")
    if frame.empty:
        raise ValueError(f"快照檔找不到指定 ID：{snapshot_id}")

    def normalize_horse_number(values: pd.Series) -> pd.Series:
        text = values.astype("string").str.strip()
        numeric = pd.to_numeric(text, errors="coerce")
        return text.mask(numeric.notna(), numeric.round().astype("Int64").astype("string"))

    results = official_results.copy()
    for data in (frame, results):
        data["賽事編號"] = data["賽事編號"].astype("string").str.strip()
        data["馬號"] = normalize_horse_number(data["馬號"])
    key = ["賽事編號", "馬號"]
    if frame.duplicated(key).any():
        raise ValueError("選取的賽前快照有重複『賽事編號＋馬號』。")
    if results.duplicated(key).any():
        raise ValueError("官方賽果 CSV 有重複『賽事編號＋馬號』，請先去重。")

    result_cols = key + [c for c in ["名次", "target_win", "馬名", "狀態", "馬匹狀態", "賽果狀態"] if c in results.columns and c not in key]
    results = results[result_cols].copy()
    if "名次" in results.columns:
        placing = results["名次"].astype("string").fillna("").str.strip()
        status_text = placing.copy()
        for status_col in ["狀態", "馬匹狀態", "賽果狀態", "馬名"]:
            if status_col in results.columns:
                status_text = status_text.str.cat(results[status_col].astype("string").fillna(""), sep=" ")
        withdrawn = status_text.str.contains(r"退出|退賽|scratched|withdrawn", case=False, regex=True, na=False)
        leading_number = placing.str.extract(r"^\s*(\d+)", expand=False)
        result_win = pd.to_numeric(leading_number, errors="coerce").eq(1).fillna(False).astype(int)
        results["__result_win"] = result_win
        results["__withdrawn"] = withdrawn
    else:
        results["__result_win"] = pd.to_numeric(results["target_win"], errors="coerce").fillna(0).astype(int)
        results["__withdrawn"] = False

    result_join = results[key + ["__result_win", "__withdrawn"]].copy()
    if "名次" in results.columns:
        result_join["官方名次"] = results["名次"]
    if "馬名" in results.columns:
        result_join["官方賽果馬名"] = results["馬名"]
    merged = frame.merge(result_join, on=key, how="left", indicator=True, validate="one_to_one")
    missing_results = int(merged["_merge"].ne("both").sum())
    if missing_results:
        examples = merged.loc[merged["_merge"].ne("both"), key].head(8).astype(str).agg(" ".join, axis=1).tolist()
        raise ValueError(f"官方賽果缺少快照內 {missing_results} 匹馬的對應紀錄；例：{'；'.join(examples)}。請使用包含退出馬狀態的完整賽果 CSV，避免把退出馬錯算成落敗。")
    merged = merged.drop(columns="_merge").rename(columns={"__result_win": "target_win"})
    withdrawn_count = int(merged["__withdrawn"].sum())
    active = merged.loc[~merged["__withdrawn"]].drop(columns="__withdrawn").copy()
    if active.empty:
        raise ValueError("選取的快照內所有馬匹均標記退出／退賽，無可回測對象。")
    if "market_implied_prob" not in active.columns:
        odds = pd.to_numeric(active["獨贏賠率"], errors="coerce")
        if odds.isna().any() or odds.le(1).any():
            raise ValueError("快照缺少 market_implied_prob，且 WIN 賠率無效，不能建立市場熱門基準。")
        raw_prob = 1.0 / odds
        active["market_implied_prob"] = raw_prob / raw_prob.groupby(active["賽事編號"], observed=True).transform("sum")
    if "馬名" not in active.columns:
        active["馬名"] = ""
    summary, details = top1_roi(active, stake_per_race)
    # Preserve original selections before scratches are removed from WIN picks;
    # a PLACE bet on a scratched runner is void, not replaced by rank four.
    place_summary, place_details = place_top3_backtest(
        merged,
        stake_per_bet=stake_per_place_bet if stake_per_place_bet is not None else stake_per_race,
        rank_col="官方名次" if "官方名次" in merged.columns else "名次",
        withdrawn_col="__withdrawn",
    )
    summary["PLACE"] = place_summary
    details["賽前快照ID"] = snapshot_id or "單一快照"
    if "賽前快照時間(HKT)" in active.columns:
        times = active["賽前快照時間(HKT)"].dropna().astype(str).unique().tolist()
        if len(times) == 1:
            details["賽前快照時間(HKT)"] = times[0]
    summary["退出／退賽馬匹略過數"] = withdrawn_count
    summary["選取快照ID"] = snapshot_id or "單一快照"
    selected = active.drop(columns="__withdrawn", errors="ignore")
    return summary, details, selected, place_details
