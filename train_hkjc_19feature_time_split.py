"""HKJC LightGBM training with chronological splits and leakage-safe features.

Expected race id example: 20260927-01. The script is designed for Google Colab,
but can also run locally after changing DATA_DIR and removing the Drive mount.
"""
from __future__ import annotations

import glob
import json
import os
import re
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# =========================
# 1. Colab / data settings
# =========================
USE_GOOGLE_DRIVE = True
DRIVE_DIR = "/content/drive/MyDrive/Jmanus"
LOCAL_DIR = "."  # Used when USE_GOOGLE_DRIVE = False

FEATURE_INPUT_FILENAMES = [
    "Predict/hkjc_19feature_training_20261001.csv",
]
SEARCH_PATTERNS = ["hkjc_data_*.csv", "hkjc_full_season_*.csv"]
MODEL_FILENAME = "my_hkjc_model_19feature_candidate.pkl"
METRICS_FILENAME = "training_metrics_19feature.json"
TEST_PREDICTIONS_FILENAME = "test_predictions_19feature.csv"
RANDOM_STATE = 42

# Keep this order aligned with the Streamlit app's feature list.
FEATURE_COLS = [
    "market_implied_prob",
    "獨贏賠率",
    "odds_rank",
    "is_favorite",
    "排位檔位",
    "weight_diff",
    "weight_rank",
    "jockey_win_rate",
    "trainer_win_rate",
    "combo_win_rate",
    "horse_win_rate",
    "horse_last_rank",
    "距離",
    "horse_surface_win_rate",
    "horse_dist_win_rate",
    "days_since_last_race",
    "horse_course_starts",
    "horse_course_win_rate",
    "horse_course_top3_rate",
]

# Smoothed fallback values for runners with little/no history.
BASE_WIN_RATE = 0.10
PRIOR_STRENGTH = 10.0


def setup_data_dir() -> Path:
    """Mount Drive in Colab, or use the configured local folder."""
    if USE_GOOGLE_DRIVE:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
        data_dir = Path(DRIVE_DIR)
    else:
        data_dir = Path(LOCAL_DIR).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"資料目錄不存在：{data_dir}")
    os.chdir(data_dir)
    return data_dir


def read_csv_robust(file_path: str) -> pd.DataFrame:
    """Read a CSV with common HKJC encodings."""
    try:
        return pd.read_csv(file_path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="cp950", on_bad_lines="skip")


def race_date_from_id(race_id: object) -> pd.Timestamp:
    """Extract YYYYMMDD from a race id such as 20260927-01."""
    match = re.search(r"(20\d{6})", str(race_id))
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")


def text_key(series: pd.Series, fallback: str = "__UNKNOWN__") -> pd.Series:
    """Normalize entity names/ids for historical statistics."""
    values = series.astype("string").str.strip()
    values = values.mask(values.isna() | values.eq(""), fallback)
    return values.astype(str)


def add_prior_rate_feature(
    df: pd.DataFrame,
    entity_cols: list[str],
    feature_name: str,
    base_rate: float = BASE_WIN_RATE,
    prior_strength: float = PRIOR_STRENGTH,
) -> pd.DataFrame:
    """Add a smoothed entity win rate using outcomes from earlier dates only.

    First aggregate at entity + date, then shift via cumulative totals. This
    excludes every result from the current date, including other races on that
    date, so same-day race ordering cannot leak outcomes into features.
    """
    group_cols = entity_cols + ["_race_date"]
    daily = (
        df.groupby(group_cols, observed=True, dropna=False, as_index=False)
        .agg(
            _daily_wins=("target_win", "sum"),
            _daily_starts=("target_win", "size"),
        )
        .sort_values(group_cols)
    )

    daily["_cum_wins"] = (
        daily.groupby(entity_cols, observed=True, sort=False)["_daily_wins"].cumsum()
        - daily["_daily_wins"]
    )
    daily["_cum_starts"] = (
        daily.groupby(entity_cols, observed=True, sort=False)["_daily_starts"].cumsum()
        - daily["_daily_starts"]
    )
    daily[feature_name] = (
        daily["_cum_wins"] + base_rate * prior_strength
    ) / (daily["_cum_starts"] + prior_strength)

    lookup = daily[group_cols + [feature_name]]
    # Input CSVs may already contain stale precomputed feature columns. Drop
    # them so merge does not rename the fresh column to feature_name_x/y.
    df_for_merge = df.drop(columns=[feature_name], errors="ignore")
    result = df_for_merge.merge(lookup, on=group_cols, how="left", validate="many_to_one")
    result[feature_name] = result[feature_name].fillna(base_rate)
    return result


def add_prior_last_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Add each horse's most recent finishing position before the current date."""
    daily = (
        df.groupby(["_horse_key", "_race_date"], observed=True, as_index=False)
        .agg(_day_rank=("numeric_rank", "min"))
        .sort_values(["_horse_key", "_race_date"])
    )
    daily["horse_last_rank"] = daily.groupby(
        "_horse_key", observed=True, sort=False
    )["_day_rank"].shift(1)

    # Ignore a stale value from a source CSV; calculate the as-of feature anew.
    df_for_merge = df.drop(columns=["horse_last_rank"], errors="ignore")
    result = df_for_merge.merge(
        daily[["_horse_key", "_race_date", "horse_last_rank"]],
        on=["_horse_key", "_race_date"],
        how="left",
        validate="many_to_one",
    )
    # Six is a neutral/unknown-history fallback, not a future-derived value.
    result["horse_last_rank"] = result["horse_last_rank"].fillna(6.0)
    return result


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean rows and build features that are available without future results."""
    required = ["賽事編號", "馬號", "名次", "獨贏賠率"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要欄位：{missing}")

    extra_features = [
        "days_since_last_race",
        "horse_course_starts",
        "horse_course_win_rate",
        "horse_course_top3_rate",
    ]
    missing_extra = [col for col in extra_features if col not in raw.columns]
    required_official = ["racecourse_code", "official_distance_m"]
    missing_official = [col for col in required_official if col not in raw.columns]
    if missing_extra:
        raise ValueError(
            f"輸入 CSV 缺少新增歷史特徵：{missing_extra}。"
            "請先將所有賽季 CSV 合併，執行 hkjc_extra_features.py，"
            "再使用其輸出的 *_features.csv 進行訓練。"
        )
    if missing_official:
        raise ValueError(
            f"輸入 CSV 缺少 HKJC 官方補抓欄位：{missing_official}。"
            "請先執行 hkjc_fetch_missing_features.py，並使用 *_hkjc_enriched.csv。"
        )

    df = raw.copy()
    df["賽事編號"] = df["賽事編號"].astype(str).str.strip()
    df["_race_date"] = df["賽事編號"].map(race_date_from_id)
    if df["_race_date"].isna().any():
        date_col = next(
            (col for col in ["賽事日期", "日期", "race_date", "date"] if col in df.columns),
            None,
        )
        if date_col:
            fallback_dates = pd.to_datetime(df[date_col], errors="coerce")
            df["_race_date"] = df["_race_date"].fillna(fallback_dates)

    # Keep recorded non-numeric outcomes (e.g. DNF/withdrawn statuses) as
    # non-winners; exclude only blank outcomes that may be unrun races.
    rank_text = df["名次"].astype("string").str.strip()
    has_result = rank_text.notna() & rank_text.ne("")
    df = df.loc[has_result].copy()
    df["numeric_rank"] = pd.to_numeric(df["名次"], errors="coerce").fillna(99.0)
    df["獨贏賠率"] = pd.to_numeric(df["獨贏賠率"], errors="coerce")
    df["馬號"] = df["馬號"].astype(str).str.strip()
    df = df.dropna(subset=["_race_date", "獨贏賠率"]).copy()
    df = df[df["獨贏賠率"] > 1.0].copy()
    df["target_win"] = (df["numeric_rank"] == 1).astype(int)

    # Avoid duplicated rows when overlapping season files were concatenated.
    df = df.drop_duplicates(subset=["賽事編號", "馬號"], keep="last").copy()
    if df.empty:
        raise ValueError("清洗後沒有可訓練資料，請檢查日期、名次、獨贏賠率與馬號欄位。")

    # Stable entity keys. Prefer a permanent horse id/code when present.
    horse_id_col = next(
        (col for col in ["馬匹編號", "馬匹代號", "horse_id", "horse_code"] if col in df.columns),
        "馬名" if "馬名" in df.columns else "馬號",
    )
    df["_horse_key"] = text_key(df[horse_id_col])
    jockey_col = "騎師" if "騎師" in df.columns else None
    trainer_col = "練馬師" if "練馬師" in df.columns else None
    df["_jockey_key"] = text_key(df[jockey_col]) if jockey_col else "__UNKNOWN__"
    df["_trainer_key"] = text_key(df[trainer_col]) if trainer_col else "__UNKNOWN__"
    venue_col = "場地" if "場地" in df.columns else None
    df["_venue_key"] = text_key(df[venue_col]) if venue_col else "__UNKNOWN__"

    # Market features available for all runners in the same race.
    df["market_prob_raw"] = 1.0 / df["獨贏賠率"]
    prob_sum = df.groupby("賽事編號", observed=True)["market_prob_raw"].transform("sum")
    df["market_implied_prob"] = df["market_prob_raw"] / prob_sum.replace(0, np.nan)
    df["market_implied_prob"] = df["market_implied_prob"].fillna(0.0)
    df["odds_rank"] = df.groupby("賽事編號", observed=True)["獨贏賠率"].rank(
        method="min", ascending=True
    )
    df["is_favorite"] = (df["odds_rank"] == 1).astype(int)

    draw_col = "排位檔位" if "排位檔位" in df.columns else None
    df["排位檔位"] = (
        pd.to_numeric(df[draw_col], errors="coerce").fillna(7.0)
        if draw_col
        else 7.0
    )
    distance_col = "距離" if "距離" in df.columns else None
    df["距離"] = (
        pd.to_numeric(df[distance_col], errors="coerce").fillna(1200.0)
        if distance_col
        else 1200.0
    )

    weight_col = "實際負磅" if "實際負磅" in df.columns else None
    if weight_col:
        df["實際負磅"] = pd.to_numeric(df[weight_col], errors="coerce")
        df["實際負磅"] = df["實際負磅"].fillna(df.groupby("賽事編號")["實際負磅"].transform("mean"))
        df["實際負磅"] = df["實際負磅"].fillna(120.0)
        race_mean_weight = df.groupby("賽事編號", observed=True)["實際負磅"].transform("mean")
        df["weight_diff"] = df["實際負磅"] - race_mean_weight
        df["weight_rank"] = df.groupby("賽事編號", observed=True)["實際負磅"].rank(
            ascending=False, method="min"
        )
    else:
        df["weight_diff"] = 0.0
        df["weight_rank"] = 6.0

    # All historical rate features are calculated from dates strictly earlier
    # than the current race date. Current-day and future outcomes are excluded.
    df = add_prior_rate_feature(df, ["_jockey_key"], "jockey_win_rate")
    df = add_prior_rate_feature(df, ["_trainer_key"], "trainer_win_rate")
    df["combo_win_rate"] = (df["jockey_win_rate"] + df["trainer_win_rate"]) / 2.0
    df = add_prior_rate_feature(df, ["_horse_key"], "horse_win_rate")
    df = add_prior_last_rank(df)
    df = add_prior_rate_feature(
        df, ["_horse_key", "_venue_key"], "horse_surface_win_rate", base_rate=0.08
    )

    # Distance groups are defined from race distance, then use only prior dates.
    df["_dist_group"] = pd.cut(
        df["距離"],
        bins=[0, 1200, 1600, 2000, 2400, 10000],
        labels=["<=1200", "1201-1600", "1601-2000", "2001-2400", ">2400"],
        include_lowest=True,
    ).astype("string").fillna("unknown")
    df = add_prior_rate_feature(
        df,
        ["_horse_key", "_dist_group"],
        "horse_dist_win_rate",
        base_rate=0.08,
    )

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["days_since_last_race"] = df["days_since_last_race"].fillna(999.0)
    df["horse_course_starts"] = df["horse_course_starts"].fillna(0.0)
    df["horse_course_win_rate"] = df["horse_course_win_rate"].fillna(0.10)
    df["horse_course_top3_rate"] = df["horse_course_top3_rate"].fillna(0.30)
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df.sort_values(["_race_date", "賽事編號", "馬號"]).reset_index(drop=True)


def chronological_masks(df: pd.DataFrame):
    """Make 80/10/10 date-based train/validation/test splits."""
    dates = np.array(sorted(df["_race_date"].dropna().unique()))
    if len(dates) < 10:
        warnings.warn(
            f"只有 {len(dates)} 個不同賽日；時間切分的驗證／測試結果可能不穩。建議使用更多賽季資料。"
        )
    if len(dates) < 3:
        raise ValueError("至少需要 3 個不同賽日才能進行時間切分。")
    train_cut = dates[max(1, int(len(dates) * 0.80))]
    val_cut = dates[max(2, int(len(dates) * 0.90))]
    if val_cut <= train_cut:
        val_cut = dates[-1]
    masks = {
        "train": df["_race_date"] < train_cut,
        "validation": (df["_race_date"] >= train_cut) & (df["_race_date"] < val_cut),
        "test": df["_race_date"] >= val_cut,
    }
    if not masks["train"].any() or not masks["validation"].any() or not masks["test"].any():
        raise ValueError("時間切分後某一資料集為空；請增加賽日資料或調整切分比例。")
    return masks, pd.Timestamp(train_cut), pd.Timestamp(val_cut)


def safe_metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, float | None]:
    """Calculate probability and ranking metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    auc = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) > 1 else None
    return {
        "auc": auc,
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def top_pick_accuracy(eval_df: pd.DataFrame, score_col: str) -> float | None:
    """Fraction of races whose highest-scored runner actually won."""
    if eval_df.empty:
        return None
    top = (
        eval_df.sort_values(["賽事編號", score_col], ascending=[True, False])
        .groupby("賽事編號", observed=True, as_index=False)
        .head(1)
    )
    return float(top["target_win"].mean()) if len(top) else None


def main() -> None:
    data_dir = setup_data_dir()
    prepared_files = [str(data_dir / name) for name in FEATURE_INPUT_FILENAMES if (data_dir / name).exists()]
    if prepared_files:
        # Use the consolidated feature export only; don't append its source CSVs
        # a second time and accidentally duplicate rows.
        files = prepared_files[:1]
    else:
        files = sorted(
            set(
                path
                for pattern in SEARCH_PATTERNS
                for path in glob.glob(str(data_dir / pattern))
                if not Path(path).stem.endswith("_features")
            )
        )
    if not files:
        raise FileNotFoundError(
            f"在 {data_dir} 找不到符合檔名的 CSV：{SEARCH_PATTERNS}"
        )
    print(f"找到 {len(files)} 個 CSV：")
    for file_path in files:
        print(" -", Path(file_path).name)

    frames = [read_csv_robust(file_path) for file_path in files]
    raw = pd.concat(frames, ignore_index=True)
    print(f"合併後原始資料：{len(raw):,} 列")

    df = build_features(raw)
    print(f"清理／去重後資料：{len(df):,} 列、{df['賽事編號'].nunique():,} 場、{df['_race_date'].nunique():,} 個賽日")

    masks, train_cut, val_cut = chronological_masks(df)
    train_df = df.loc[masks["train"]].copy()
    val_df = df.loc[masks["validation"]].copy()
    test_df = df.loc[masks["test"]].copy()
    print("時間切分（依日期，不隨機拆散同一賽日）：")
    for name, subset in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(
            f"  {name}: {subset['_race_date'].min().date()} 至 {subset['_race_date'].max().date()}"
            f" | {len(subset):,} 匹 | {subset['賽事編號'].nunique():,} 場"
        )

    X_train, y_train = train_df[FEATURE_COLS], train_df["target_win"]
    X_val, y_val = val_df[FEATURE_COLS], val_df["target_win"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target_win"]
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        raise ValueError("Train 和 Validation 都必須同時包含勝出與未勝出樣本。")

    params = {
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
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    tuning_model = lgb.LGBMClassifier(**params)
    tuning_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=True)],
    )

    best_iteration = int(tuning_model.best_iteration_ or params["n_estimators"])
    print(f"Validation 選出的最佳迭代次數：{best_iteration}")

    # Refit on all pre-test rows, preserving the untouched future test period.
    final_params = dict(params)
    final_params["n_estimators"] = best_iteration
    final_model = lgb.LGBMClassifier(**final_params)
    train_val_df = pd.concat([train_df, val_df], axis=0).sort_values(
        ["_race_date", "賽事編號", "馬號"]
    )
    final_model.fit(train_val_df[FEATURE_COLS], train_val_df["target_win"])

    test_probability = final_model.predict_proba(X_test)[:, 1]
    model_scores = safe_metrics(y_test, test_probability)
    market_scores = safe_metrics(y_test, test_df["market_implied_prob"].to_numpy())

    test_eval = test_df[["賽事編號", "馬號", "target_win", "odds_rank", "market_implied_prob"]].copy()
    test_eval["model_win_probability"] = test_probability
    model_top1 = top_pick_accuracy(test_eval, "model_win_probability")
    market_top1 = top_pick_accuracy(test_eval, "market_implied_prob")

    print("\n最終未見未來測試期成績：")
    print(f"  Model AUC:        {model_scores['auc']}")
    print(f"  Model Brier:      {model_scores['brier_score']:.5f}（越低越好）")
    print(f"  Model Log loss:   {model_scores['log_loss']:.5f}（越低越好）")
    print(f"  Model 每場最高分馬命中率: {model_top1:.4f}" if model_top1 is not None else "  Model Top-1: N/A")
    print(f"  Market AUC:       {market_scores['auc']}")
    print(f"  Market Brier:     {market_scores['brier_score']:.5f}")
    print(f"  Market Log loss:  {market_scores['log_loss']:.5f}")
    print(f"  Market 每場最低賠率馬命中率: {market_top1:.4f}" if market_top1 is not None else "  Market Top-1: N/A")

    # Save only the estimator object at the expected path so the existing app can load it.
    joblib.dump(final_model, MODEL_FILENAME)
    print(f"\n模型已儲存：{data_dir / MODEL_FILENAME}")

    test_export = test_eval.copy()
    test_export["日期"] = test_df["_race_date"].dt.strftime("%Y-%m-%d").to_numpy()
    test_export = test_export.rename(
        columns={
            "target_win": "actual_win",
            "odds_rank": "market_odds_rank",
            "market_implied_prob": "market_probability",
        }
    )
    test_export.to_csv(TEST_PREDICTIONS_FILENAME, index=False, encoding="utf-8-sig")

    metrics = {
        "feature_columns": FEATURE_COLS,
        "best_iteration": best_iteration,
        "split": {
            "train_end_exclusive": train_cut.strftime("%Y-%m-%d"),
            "validation_end_exclusive": val_cut.strftime("%Y-%m-%d"),
            "test_start": test_df["_race_date"].min().strftime("%Y-%m-%d"),
            "test_end": test_df["_race_date"].max().strftime("%Y-%m-%d"),
        },
        "rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "races": {"train": int(train_df["賽事編號"].nunique()), "validation": int(val_df["賽事編號"].nunique()), "test": int(test_df["賽事編號"].nunique())},
        "test_model_metrics": model_scores,
        "test_market_metrics": market_scores,
        "test_top1_accuracy": {"model": model_top1, "market": market_top1},
        "notes": [
            "History rate and previous-rank features use records from strictly earlier calendar dates only.",
            "All races on the same date are excluded from each other's historical features.",
            "The test period is not used for early stopping or parameter selection.",
        ],
    }
    Path(METRICS_FILENAME).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"測試預測已儲存：{data_dir / TEST_PREDICTIONS_FILENAME}")
    print(f"評估摘要已儲存：{data_dir / METRICS_FILENAME}")
    print("\n提醒：部署時必須以相同的過去資料邏輯產生騎師／馬匹歷史率特徵，避免訓練與推論特徵不一致。")


if __name__ == "__main__":
    main()
