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
from hkjc_get_racecard import download_racecard

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
        futurecard = build_future_features(history, raw_card)

    missing = [col for col in FEATURE_COLS if col not in futurecard.columns]
    if missing:
        raise ValueError(f"產生賽卡時缺少19大欄位：{missing}")
    feature_frame = futurecard[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(feature_frame.to_numpy(dtype=float)).all():
        bad = [col for col in FEATURE_COLS if not np.isfinite(feature_frame[col].to_numpy(dtype=float)).all()]
        raise ValueError(f"產生的賽卡有空白或非有限19大特徵：{bad}")
    if futurecard["獨贏賠率"].isna().any() or (pd.to_numeric(futurecard["獨贏賠率"], errors="coerce") <= 1).any():
        raise ValueError("HKJC 尚未提供全部有效獨贏賠率，未輸出不完整的19大預測CSV。")
    if futurecard["名次"].astype("string").str.strip().replace("<NA>", "").ne("").any():
        raise ValueError("官方頁回傳資料含已填賽果，為防止賽後資料混入預測已停止。")
    return futurecard, odds_update_time, raw_card


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


def chronological_backtest(raw: pd.DataFrame, stake_per_race: float = 10.0) -> dict[str, Any]:
    """Evaluate a model on the untouched final date block and calculate top-1 ROI."""
    frame = prepare_training_data(raw)
    masks, train_cut, val_cut = trainer.chronological_masks(frame)
    train = frame.loc[masks["train"]].copy()
    validation = frame.loc[masks["validation"]].copy()
    test = frame.loc[masks["test"]].copy()
    best_iteration = _fit_iteration(train, validation)
    model = _fit_model(pd.concat([train, validation], axis=0).sort_values(["_race_date", "賽事編號", "馬號"]), best_iteration)
    probabilities = _prob_for_win(model, test[FEATURE_COLS])
    predicted = test[[c for c in ["賽事編號", "馬季", "馬號", "馬名", "名次", "target_win", "獨贏賠率", "market_implied_prob"] if c in test.columns]].copy()
    predicted["model_win_probability"] = probabilities
    predicted["model_rank"] = predicted.groupby("賽事編號", observed=True)["model_win_probability"].rank(ascending=False, method="min")
    predicted["market_rank"] = predicted.groupby("賽事編號", observed=True)["market_implied_prob"].rank(ascending=False, method="min")
    roi_summary, race_details = top1_roi(predicted, stake_per_race)
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
    }
    return {"metrics": metrics, "predictions": predicted, "race_details": race_details, "model": model}


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
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
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
        selected["target_win"] = (
            pd.to_numeric(selected["名次"], errors="coerce").eq(1).astype(int)
        )
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
    y = pd.to_numeric(selected["target_win"], errors="coerce").fillna(0).to_numpy(dtype=int)
    model_p = pd.to_numeric(selected["model_win_probability"], errors="coerce").to_numpy(dtype=float)
    market_p = pd.to_numeric(selected["market_implied_prob"], errors="coerce").to_numpy(dtype=float)
    metrics = {
        "日期": target_date,
        "賽事數": int(selected["賽事編號"].nunique()),
        "馬匹數": int(len(selected)),
        "ROI": summary_roi,
        "模型概率指標": _probability_metrics(y, model_p),
        "市場概率指標": _probability_metrics(y, market_p),
    }
    return metrics, details, selected
