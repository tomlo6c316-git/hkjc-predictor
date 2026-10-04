from __future__ import annotations

import json
import hashlib
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from hkjc_pipeline import (
    FEATURE_COLS,
    MODEL_PARAMS,
    backtest_prediction_snapshot,
    chronological_backtest,
    fetch_and_build_futurecard,
    load_model_bytes,
    model_bytes,
    predict_racecard,
    read_csv_bytes,
    refresh_market_odds,
    single_date_backtest,
    train_full_model,
)

st.set_page_config(page_title="HKJC 19大模型工作台", page_icon="🐎", layout="wide")
st.title("HKJC 19大模型工作台")
st.caption("訓練模型｜預測未開跑賽事｜時間切分回測。沿用既有19大特徵和訓練參數，不會自動改參數或覆蓋 GitHub 檔案。")

with st.expander("模型設定及資料規則", expanded=False):
    st.markdown(
        "**固定 LightGBM 設定**：" + ", ".join(
            f"`{k}={v}`" for k, v in MODEL_PARAMS.items() if k not in {"objective", "random_state", "n_jobs", "verbosity"}
        ) + "; `early_stopping=75`。"
    )
    st.markdown("**19項特徵欄位次序**：" + "、".join(f"`{c}`" for c in FEATURE_COLS))
    st.warning("模型檔是 joblib/pickle 格式，只載入你自己或可信來源的模型。")

train_tab, predict_tab, backtest_tab = st.tabs(["一、訓練模型", "二、預測賽事", "三、回測與 ROI"])


def _refresh_saved_card(active_model) -> None:
    card = st.session_state.get("hkjc19_futurecard_csv")
    if card is None:
        raise ValueError("請先抓取並產生一份19大 prediction CSV。")
    refreshed, odds_time = refresh_market_odds(card)
    scored = predict_racecard(refreshed, active_model) if active_model is not None else None
    checked_at = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds")
    st.session_state["hkjc19_futurecard_csv"] = refreshed
    st.session_state["hkjc19_futurecard_prediction"] = scored
    metadata = dict(st.session_state.get("hkjc19_futurecard_meta", {}))
    metadata.update({"odds_time": odds_time, "last_refresh_hkt": checked_at})
    st.session_state["hkjc19_futurecard_meta"] = metadata
    st.session_state["hkjc19_odds_refresh_error"] = None
    st.session_state["hkjc19_odds_refresh_last_attempt"] = time.time()


def _append_pre_race_snapshot(prediction: pd.DataFrame, metadata: dict) -> str:
    snapshot_time = datetime.now(ZoneInfo("Asia/Hong_Kong"))
    inferred_date = ""
    if "賽事編號" in prediction.columns:
        date_keys = prediction["賽事編號"].astype("string").str.extract(r"(20\d{6})", expand=False).dropna().unique()
        if len(date_keys) == 1:
            inferred_date = pd.to_datetime(date_keys[0], format="%Y%m%d").strftime("%Y-%m-%d")
    inferred_venue = ""
    if "racecourse_code" in prediction.columns:
        venue_keys = prediction["racecourse_code"].astype("string").str.upper().dropna().unique()
        if len(venue_keys) == 1:
            inferred_venue = venue_keys[0]
    day_text = str(metadata.get("date") or inferred_date)
    venue = str(metadata.get("venue") or inferred_venue or "unknown")
    day = day_text.replace("-", "") or "unknown"
    snapshot_id = f"{day}_{venue}_{snapshot_time:%Y%m%dT%H%M%S%f}"
    snapshot = prediction.copy()
    snapshot["賽前快照ID"] = snapshot_id
    snapshot["賽前快照時間(HKT)"] = snapshot_time.isoformat(timespec="seconds")
    snapshot["預測賽日"] = day_text
    snapshot["預測馬場"] = metadata.get("venue", "")
    snapshot["模型來源"] = st.session_state.get("hkjc19_model_name", "本工作階段模型")
    model_artifact = st.session_state.get("hkjc19_model_bytes", b"")
    snapshot["模型SHA256"] = hashlib.sha256(model_artifact).hexdigest() if model_artifact else ""
    all_snapshots = list(st.session_state.get("hkjc19_saved_snapshots", []))
    all_snapshots.append(snapshot)
    st.session_state["hkjc19_saved_snapshots"] = all_snapshots
    st.session_state["hkjc19_latest_snapshot_id"] = snapshot_id
    return snapshot_id


def _render_place_results(summary: dict, details: pd.DataFrame, key: str) -> None:
    """Show model/market top-three Place backtest results and detailed tickets."""
    st.markdown("#### 位置（PLACE）前三選馬回測")
    if summary and "沒有官方名次" in str(summary.get("PLACE狀態", "")):
        st.warning(summary["PLACE狀態"])
        return
    if not summary or summary.get("PLACE有效賽事數", 0) == 0:
        st.info("沒有可結算的PLACE賽事（HKJC本地賽4匹起跑馬以上才符合位置派彩規則）。")
        return
    model_roi = summary.get("PLACE模型ROI")
    market_roi = summary.get("PLACE市場ROI")
    table = pd.DataFrame([
        {"策略": "模型排序前三匹分別買PLACE", "賽事數": summary.get("PLACE有效賽事數"),
         "選中注數": summary.get("PLACE模型選中注數"), "命中注數": summary.get("PLACE模型命中注數"),
         "注命中率": summary.get("PLACE模型注命中率"), "至少中一匹場數": summary.get("PLACE模型至少中一匹場數"),
         "場命中率": summary.get("PLACE模型場命中率"), "總投注(HK$)": summary.get("PLACE模型總投注"),
         "估算總派彩(HK$)": summary.get("PLACE模型估算總派彩"), "淨盈虧(HK$)": summary.get("PLACE模型淨盈虧"), "ROI": model_roi},
        {"策略": "市場排序前三匹分別買PLACE", "賽事數": summary.get("PLACE有效賽事數"),
         "選中注數": summary.get("PLACE市場選中注數"), "命中注數": summary.get("PLACE市場命中注數"),
         "注命中率": summary.get("PLACE市場注命中率"), "至少中一匹場數": summary.get("PLACE市場至少中一匹場數"),
         "場命中率": summary.get("PLACE市場場命中率"), "總投注(HK$)": summary.get("PLACE市場總投注"),
         "估算總派彩(HK$)": summary.get("PLACE市場估算總派彩"), "淨盈虧(HK$)": summary.get("PLACE市場淨盈虧"), "ROI": market_roi},
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)
    missing_model_odds = summary.get("PLACE缺少有效位置賠率注數", 0)
    missing_market_odds = summary.get("PLACE市場缺少有效位置賠率注數", 0)
    if missing_model_odds or missing_market_odds:
        st.warning(f"模型前三缺 {missing_model_odds} 注、 市場前三缺 {missing_market_odds} 注有效位置賠率；仍可看得位命中，但不會用不完整金額計ROI。請保存HKJC PLACE賠率。")
    elif summary.get("PLACE狀態"):
        st.caption(summary["PLACE狀態"])
    if details is not None and not details.empty:
        st.dataframe(details, use_container_width=True, hide_index=True)
        st.download_button(
            "下載PLACE逐注回測明細 CSV",
            data=details.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{key}_place_top3_details.csv",
            mime="text/csv",
            key=f"download_{key}_place_details",
        )


def _refresh_uploaded_card(active_model) -> str | None:
    card = st.session_state.get("hkjc19_last_prediction_card")
    if card is None:
        raise ValueError("請先上傳19大賽卡並完成一次預測。")
    refreshed, odds_time = refresh_market_odds(card)
    refreshed["賠率快照時間"] = odds_time or "HKJC API 未提供"
    refreshed["CSV產生時間(HKT)"] = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds")
    scored = predict_racecard(refreshed, active_model)
    st.session_state["hkjc19_last_prediction_card"] = refreshed
    st.session_state["hkjc19_last_prediction"] = scored
    return odds_time

with train_tab:
    st.subheader("訓練19大模型")
    st.write("上傳已完成賽事的 enriched CSV。系統使用過去日期計算特徵、以時間驗證期選最佳迭代次數，再用全部已標籤賽事訓練供未來預測使用的模型。")
    training_file = st.file_uploader("訓練資料 CSV（需含官方賽果、19大來源欄位）", type=["csv"], key="training_csv")
    st.info("此頁產生供未來預測使用的模型，不會將留出測試期指標冒充訓練成績。請先到「回測與 ROI」用時間切分測試評估資料。")
    if training_file and st.button("開始訓練模型", type="primary", key="train_model_button"):
        try:
            with st.spinner("正在整理資料、選擇迭代次數並訓練模型；資料較大時需稍候…"):
                raw = read_csv_bytes(training_file.getvalue())
                model, prepared, info = train_full_model(raw)
                artifact = model_bytes(model)
            st.session_state["hkjc19_model"] = model
            st.session_state["hkjc19_model_bytes"] = artifact
            st.session_state["hkjc19_model_name"] = "my_hkjc_model_19feature.pkl"
            st.success("訓練完成。模型已暫存在目前 Streamlit 工作階段；下載後可在預測分頁重新載入。")
            c1, c2, c3 = st.columns(3)
            c1.metric("完成賽事", f"{info['賽事數']:,} 場")
            c2.metric("訓練馬匹", f"{info['訓練馬匹數']:,} 匹")
            c3.metric("最佳迭代次數", str(info["最佳迭代次數"]))
            st.json(info)
        except Exception as exc:
            st.error(f"訓練失敗：{exc}")
    if st.session_state.get("hkjc19_model_bytes"):
        st.download_button(
            "下載候選19大模型（joblib）",
            data=st.session_state["hkjc19_model_bytes"],
            file_name=st.session_state.get("hkjc19_model_name", "my_hkjc_model_19feature.pkl"),
            mime="application/octet-stream",
            key="download_trained_model",
        )

with predict_tab:
    st.subheader("預測未開跑賽事")
    st.write("可直接選日期抓取 HKJC 官方排位與當前WIN／PLACE賠率，搭配歷史 enriched CSV 計算19大特徵，再下載單一 prediction CSV。")
    active_model = st.session_state.get("hkjc19_model")
    model_file = st.file_uploader("可選：載入已訓練的19大模型（joblib/pkl）", type=["pkl", "joblib"], key="prediction_model")
    if model_file is not None:
        try:
            active_model = load_model_bytes(model_file.getvalue())
            st.session_state["hkjc19_model"] = active_model
            st.session_state["hkjc19_model_bytes"] = model_file.getvalue()
            st.session_state["hkjc19_model_name"] = model_file.name
            st.success(f"已載入模型：{model_file.name}")
        except Exception as exc:
            active_model = None
            st.error(f"模型載入失敗：{exc}")

    st.markdown("#### 一鍵取得官方排位＋19大預測資料 CSV")
    st.caption("需要當前有效的歷史 enriched CSV（只在這次運算中使用，不會存入 GitHub）。輸出是賽前特徵，不含預測賽果；賠率必須已由 HKJC 開盤提供。")
    history_file = st.file_uploader("上傳歷史 enriched CSV（例如 hkjc_enriched_season_fixed.csv）", type=["csv"], key="futurecard_history_csv")
    today_hkt = datetime.now(ZoneInfo("Asia/Hong_Kong")).date()
    future_date = st.date_input(
        "選擇未來賽日",
        value=today_hkt + timedelta(days=1),
        min_value=today_hkt + timedelta(days=1),
        key="futurecard_target_date",
    )
    future_venue = st.selectbox("馬場", ["ST", "HV"], format_func=lambda value: "沙田（ST）" if value == "ST" else "跑馬地（HV）", key="futurecard_venue")
    if st.button("抓取排位並產生19大 prediction CSV", type="primary", key="fetch_futurecard_button"):
        for state_key in ["hkjc19_futurecard_csv", "hkjc19_futurecard_prediction", "hkjc19_futurecard_meta", "hkjc19_futurecard_file_name"]:
            st.session_state.pop(state_key, None)
        if history_file is None:
            st.error("請先上傳歷史 enriched CSV，否則不能安全計算19大過去賽績特徵。")
        else:
            try:
                with st.spinner("正在連線 HKJC 下載整日排位與最新賠率，並以賽日前歷史計算19大特徵；通常需數十秒…"):
                    history_raw = read_csv_bytes(history_file.getvalue())
                    futurecard, odds_time, raw_card = fetch_and_build_futurecard(
                        history_raw, future_date.isoformat(), future_venue
                    )
                    generated_at = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds")
                    futurecard["賠率快照時間"] = odds_time or "HKJC API 未提供"
                    futurecard["CSV產生時間(HKT)"] = generated_at
                    feature_card = futurecard[[c for c in futurecard.columns if c not in FEATURE_COLS] + FEATURE_COLS]
                    st.session_state["hkjc19_futurecard_csv"] = feature_card
                    st.session_state["hkjc19_futurecard_file_name"] = f"prediction_{future_date:%Y%m%d}_{future_venue}_19features.csv"
                    st.session_state["hkjc19_futurecard_meta"] = {
                        "date": future_date.isoformat(),
                        "venue": future_venue,
                        "odds_time": odds_time,
                        "generated_at": generated_at,
                    }
                    if active_model is not None:
                        st.session_state["hkjc19_futurecard_prediction"] = predict_racecard(feature_card, active_model)
                    else:
                        st.session_state.pop("hkjc19_futurecard_prediction", None)
            except Exception as exc:
                st.error(f"取得／計算失敗：{exc}")

    generated_card = st.session_state.get("hkjc19_futurecard_csv")
    if generated_card is not None:
        st.markdown("#### 即時賠率更新")
        st.caption("刷新時只查 HKJC 最新 WIN／PLACE 賠率，不重新下載排位或重算歷史特徵；成功後會重算4項市場特徵與模型排序。")
        auto_col, interval_col = st.columns([1.0, 1.0])
        with auto_col:
            auto_refresh = st.checkbox("自動更新", key="auto_refresh_odds")
        with interval_col:
            refresh_interval = st.selectbox("間隔（秒）", [10, 30, 60, 120, 300], index=2, key="odds_refresh_interval")

        @st.fragment(run_every=refresh_interval if auto_refresh else None)
        def live_odds_panel():
            refresh_col, _ = st.columns([1.2, 4.0])
            with refresh_col:
                manual_refresh = st.button("立即更新賠率", key="refresh_odds_now")
            last_attempt = st.session_state.get("hkjc19_odds_refresh_last_attempt")
            due = auto_refresh and (last_attempt is None or time.time() - last_attempt >= refresh_interval)
            if manual_refresh or due:
                try:
                    with st.spinner("只更新 HKJC 賠率，並重算市場特徵／排序…"):
                        _refresh_saved_card(active_model)
                except Exception as exc:
                    st.session_state["hkjc19_odds_refresh_error"] = str(exc)
                    st.session_state["hkjc19_odds_refresh_last_attempt"] = time.time()

            current_card = st.session_state.get("hkjc19_futurecard_csv")
            metadata = st.session_state.get("hkjc19_futurecard_meta", {})
            odds_label = metadata.get("odds_time") or "HKJC 未提供更新時間"
            st.success(f"已取得 {metadata.get('date')} {metadata.get('venue')}：{current_card['賽事編號'].nunique()}場、{len(current_card)}匹、19項特徵。")
            last_success = metadata.get("last_refresh_hkt") or metadata.get("generated_at")
            st.caption(f"最近成功賠率快照：{odds_label}；本機檢查時間（香港）：{last_success}。賠率可能繼續變動。")
            refresh_error = st.session_state.get("hkjc19_odds_refresh_error")
            if refresh_error:
                st.warning(f"最新刷新未成功，畫面仍保留上一份有效快照：{refresh_error}")
            st.download_button(
                "下載最新19大 prediction CSV",
                data=current_card.to_csv(index=False).encode("utf-8-sig"),
                file_name=st.session_state.get("hkjc19_futurecard_file_name", "prediction_19features.csv"),
                mime="text/csv",
                key="download_futurecard_csv",
            )
            no_prior_race = pd.to_numeric(current_card["days_since_last_race"], errors="coerce").eq(999.0)
            no_same_course = pd.to_numeric(current_card["horse_course_starts"], errors="coerce").eq(0.0)
            st.caption(f"歷史資料檢查：沒有可用舊賽紀錄 {int(no_prior_race.sum())} 匹；沒有同場地／路程歷史出賽 {int(no_same_course.sum())} 匹。這兩項使用訓練流程的預設值。")
            saved_prediction = st.session_state.get("hkjc19_futurecard_prediction")
            if saved_prediction is not None:
                st.markdown("#### 目前模型排序（如有載入19大模型）")
                prediction_cols = [c for c in ["賽事編號", "馬號", "馬名", "獨贏賠率", "model_win_probability", "model_race_probability", "model_rank"] if c in saved_prediction.columns]
                st.dataframe(saved_prediction[prediction_cols], use_container_width=True, hide_index=True)
                st.download_button(
                    "下載含最新賠率及模型排序的 CSV",
                    data=saved_prediction.to_csv(index=False).encode("utf-8-sig"),
                    file_name=st.session_state.get("hkjc19_futurecard_file_name", "prediction_19features.csv").replace(".csv", "_scored.csv"),
                    mime="text/csv",
                    key="download_scored_futurecard_csv",
                )
                st.caption("每次按保存都會記錄此刻的模型分數及可用的WIN／PLACE賠率。快照只暫存在目前瀏覽器工作階段，請下載保存，賽後再上傳回測。")
                if st.button("保存本次賽前預測＋賠率快照", type="primary", key="save_pre_race_snapshot"):
                    snapshot_id = _append_pre_race_snapshot(saved_prediction, metadata)
                    st.success(f"已保存 {snapshot_id}：{saved_prediction['賽事編號'].nunique()}場、{len(saved_prediction)}匹。請下載快照CSV妥善保存。")
            saved_snapshots = st.session_state.get("hkjc19_saved_snapshots", [])
            if saved_snapshots:
                snapshot_export = pd.concat(saved_snapshots, ignore_index=True)
                snapshot_count = snapshot_export["賽前快照ID"].nunique()
                st.info(f"本工作階段已保存 {snapshot_count} 個快照，共 {len(snapshot_export):,} 匹次紀錄。工作階段結束後可能清除，請下載留存。")
                latest_id = st.session_state.get("hkjc19_latest_snapshot_id", "snapshot")
                st.download_button(
                    "下載所有已保存的賽前快照 CSV",
                    data=snapshot_export.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"hkjc_pre_race_snapshots_{latest_id}.csv",
                    mime="text/csv",
                    key="download_pre_race_snapshots",
                )

        live_odds_panel()

    st.markdown("#### 或上傳已準備好的19大賽卡 CSV")
    st.caption("如已使用 futurecard skill 建好含19項特徵的 CSV，可在這裡直接載入模型預測。")
    race_file = st.file_uploader("19大未開跑賽卡 CSV", type=["csv"], key="racecard_csv")
    if active_model is None:
        st.warning("請先在「訓練模型」頁訓練，或上傳可信的19大模型檔。")
    elif race_file and st.button("開始預測", type="primary", key="predict_button"):
        try:
            race_raw = read_csv_bytes(race_file.getvalue())
            predicted = predict_racecard(race_raw, active_model)
            st.session_state["hkjc19_last_prediction_card"] = race_raw.copy()
            st.session_state["hkjc19_last_prediction"] = predicted
            st.session_state["hkjc19_last_prediction_name"] = f"{race_file.name.rsplit('.', 1)[0]}_predicted.csv"
            st.session_state["hkjc19_last_prediction_meta"] = {
                "date": "",
                "venue": "",
            }
        except Exception as exc:
            st.error(f"預測失敗：{exc}")
    uploaded_prediction = st.session_state.get("hkjc19_last_prediction")
    if uploaded_prediction is not None:
        st.success(f"預測完成：{uploaded_prediction['賽事編號'].nunique()}場、{len(uploaded_prediction)}匹。")
        st.markdown("#### 上傳賽卡的即時賠率更新")
        st.caption("此刷新器適用於上方『上傳已準備好的19大賽卡 CSV』流程；只更新HKJC WIN／PLACE賠率及市場特徵，不重抓排位或重算19大歷史特徵。")
        uploaded_auto_col, uploaded_interval_col = st.columns([1.0, 1.0])
        with uploaded_auto_col:
            uploaded_auto_refresh = st.checkbox("上傳賽卡自動更新", key="uploaded_auto_refresh_odds")
        with uploaded_interval_col:
            uploaded_refresh_interval = st.selectbox("上傳賽卡更新間隔（秒）", [10, 30, 60, 120, 300], index=2, key="uploaded_odds_refresh_interval")

        @st.fragment(run_every=uploaded_refresh_interval if uploaded_auto_refresh else None)
        def uploaded_prediction_panel():
            if st.button("立即更新賠率", key="refresh_uploaded_odds_now"):
                st.session_state["hkjc19_uploaded_refresh_requested"] = True
            last_attempt = st.session_state.get("hkjc19_uploaded_odds_last_attempt")
            due = uploaded_auto_refresh and (last_attempt is None or time.time() - last_attempt >= uploaded_refresh_interval)
            requested = st.session_state.pop("hkjc19_uploaded_refresh_requested", False)
            if requested or due:
                try:
                    with st.spinner("正在向HKJC更新WIN／PLACE賠率並重算模型排序…"):
                        odds_time = _refresh_uploaded_card(active_model)
                    st.session_state["hkjc19_uploaded_odds_error"] = None
                    st.caption(f"最近成功更新的HKJC賠率時間：{odds_time or 'API未提供'}")
                except Exception as exc:
                    st.session_state["hkjc19_uploaded_odds_error"] = str(exc)
                finally:
                    st.session_state["hkjc19_uploaded_odds_last_attempt"] = time.time()
            error = st.session_state.get("hkjc19_uploaded_odds_error")
            if error:
                st.warning(f"最新賠率更新未成功，仍保留上一份有效快照：{error}")
            current_prediction = st.session_state.get("hkjc19_last_prediction")
            if current_prediction is None:
                st.error("目前沒有已預測賽卡；請重新上傳並按開始預測。")
                return
            show_cols = [c for c in ["賽事編號", "馬號", "馬名", "獨贏賠率", "model_win_probability", "model_race_probability", "model_rank"] if c in current_prediction.columns]
            st.dataframe(current_prediction[show_cols], use_container_width=True, hide_index=True)
            st.download_button(
                "下載預測結果 CSV",
                data=current_prediction.to_csv(index=False).encode("utf-8-sig"),
                file_name=st.session_state.get("hkjc19_last_prediction_name", "predictions.csv"),
                mime="text/csv",
                key="download_predictions",
            )
            if st.button("保存這份賽前預測＋賠率快照", key="save_uploaded_pre_race_snapshot"):
                snapshot_id = _append_pre_race_snapshot(
                    current_prediction, st.session_state.get("hkjc19_last_prediction_meta", {})
                )
                st.success(f"已保存快照 {snapshot_id}。")
            saved_snapshots = st.session_state.get("hkjc19_saved_snapshots", [])
            if saved_snapshots:
                snapshot_export = pd.concat(saved_snapshots, ignore_index=True)
                st.info(f"目前保存 {snapshot_export['賽前快照ID'].nunique()} 個快照，共 {len(snapshot_export):,} 匹次紀錄。請在離開前下載。")
                latest_id = st.session_state.get("hkjc19_latest_snapshot_id", "snapshot")
                st.download_button(
                    "下載所有已保存的賽前快照 CSV",
                    data=snapshot_export.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"hkjc_pre_race_snapshots_{latest_id}.csv",
                    mime="text/csv",
                    key="download_pre_race_snapshots_uploaded_flow",
                )
            st.caption("`model_win_probability` 為模型二元分類分數；`model_race_probability` 是同場正規化供排名參考，不保證校準或賽果。保存快照可固定此刻模型分數與輸入賠率。")

        uploaded_prediction_panel()

with backtest_tab:
    st.subheader("賽事回測與 ROI")
    backtest_mode = st.radio(
        "選擇回測方式",
        ["賽前快照＋官方賽果回測", "單一賽日快速回測（用已保存的模型預測）", "完整時間切分回測（重新訓練並評估）"],
        horizontal=True,
        key="backtest_mode",
    )
    stake_col, place_stake_col = st.columns(2)
    stake = stake_col.number_input("每場WIN首選／熱門投注金額（HK$）", min_value=1.0, max_value=10000.0, value=10.0, step=1.0, key="roi_stake")
    place_stake = place_stake_col.number_input("每匹PLACE選馬投注金額（HK$）", min_value=1.0, max_value=10000.0, value=10.0, step=1.0, key="place_roi_stake")
    st.warning("回測同時估算WIN一注及模型／市場各自排序前三匹的PLACE（每匹各一注，共最多三注）。本地賽4–6匹出賽只計頭兩名得位；7匹或以上計頭三名。派彩按CSV賠率估算，非HKJC官方結算。")

    if backtest_mode.startswith("賽前快照"):
        st.write("上傳預先保存的賽前快照 CSV 和賽後官方賽果 CSV。系統按『賽事編號＋馬號』匹配，不重新訓練模型；ROI 按快照時記錄的WIN賠率估算。")
        st.warning("請用含完整出賽名單的官方賽果檔，退出／退賽馬也須有『退出／退賽』狀態列；缺少任何匹配記錄會停止回測。快照請保存當時的獨贏與位置賠率；缺少位置賠率仍可計得位命中，但不計PLACE ROI。快照賠率不是官方派彩。")
        snapshot_file = st.file_uploader("已保存的賽前預測快照 CSV", type=["csv"], key="pre_race_snapshot_csv")
        results_file = st.file_uploader("賽後官方賽果 CSV（至少含賽事編號、馬號、名次）", type=["csv"], key="official_results_for_snapshot_csv")
        selected_snapshot_id = None
        if snapshot_file is not None:
            try:
                snapshot_preview = read_csv_bytes(snapshot_file.getvalue())
                if "賽前快照ID" in snapshot_preview.columns:
                    snapshot_ids = snapshot_preview["賽前快照ID"].dropna().astype(str).unique().tolist()
                    if len(snapshot_ids) > 1:
                        selected_snapshot_id = st.selectbox("選擇要回測的快照時間", snapshot_ids, key="snapshot_id_to_backtest")
                    elif snapshot_ids:
                        selected_snapshot_id = snapshot_ids[0]
                        st.caption(f"快照ID：{selected_snapshot_id}")
            except Exception as exc:
                st.error(f"讀取快照失敗：{exc}")
        if snapshot_file and results_file and st.button("執行賽前快照回測", type="primary", key="snapshot_backtest_button"):
            try:
                with st.spinner("正在按賽事編號及馬號配對官方賽果…"):
                    snapshots = read_csv_bytes(snapshot_file.getvalue())
                    official_results = read_csv_bytes(results_file.getvalue())
                    summary, details, selected_rows, place_details = backtest_prediction_snapshot(
                        snapshots, official_results, snapshot_id=selected_snapshot_id,
                        stake_per_race=float(stake), stake_per_place_bet=float(place_stake)
                    )
                st.success(f"快照回測完成：{summary['可結算賽事數']}場可結算，略過{summary['無頭馬／未結算賽事略過數']}場無頭馬／未結算賽事。")
                cols = st.columns(4)
                cols[0].metric("模型首選命中", f"{summary['模型首選命中場數']} / {summary['可結算賽事數']}")
                cols[1].metric("模型首選命中率", f"{summary['模型首選命中率']:.1%}")
                cols[2].metric("模型首選 ROI", f"{summary['模型首選ROI']:.1%}")
                cols[3].metric("市場熱門 ROI", f"{summary['市場熱門ROI']:.1%}")
                st.caption(f"快照ID：{summary['選取快照ID']}；退出／退賽馬匹略過：{summary['退出／退賽馬匹略過數']}匹。快照賠率是記錄時的市場報價，不等於官方結算派彩。")
                comparison = pd.DataFrame([
                    {"策略": "模型每場最高分馬", "命中場數": summary["模型首選命中場數"], "命中率": summary["模型首選命中率"], "總投注(HK$)": summary["模型首選總投注"], "估算派彩(HK$)": summary["模型首選估算總派彩"], "淨盈虧(HK$)": summary["模型首選淨盈虧"], "ROI": summary["模型首選ROI"]},
                    {"策略": "市場最低賠率熱門", "命中場數": summary["市場熱門命中場數"], "命中率": summary["市場熱門命中率"], "總投注(HK$)": summary["市場熱門總投注"], "估算派彩(HK$)": summary["市場熱門估算總派彩"], "淨盈虧(HK$)": summary["市場熱門淨盈虧"], "ROI": summary["市場熱門ROI"]},
                ])
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                _render_place_results(summary["PLACE"], place_details, "snapshot")
                st.markdown("#### 逐場預測與官方頭馬")
                st.dataframe(details, use_container_width=True, hide_index=True)
                c1, c2 = st.columns(2)
                c1.download_button("下載逐場快照回測明細 CSV", details.to_csv(index=False).encode("utf-8-sig"), "pre_race_snapshot_backtest_details.csv", "text/csv", key="download_snapshot_backtest_details")
                c2.download_button("下載已配對快照與賽果 CSV", selected_rows.to_csv(index=False).encode("utf-8-sig"), "pre_race_snapshot_with_results.csv", "text/csv", key="download_snapshot_joined_rows")
            except Exception as exc:
                st.error(f"賽前快照回測失敗：{exc}")
    elif backtest_mode.startswith("單一賽日"):
        st.write("此模式只篩選已保存的測試預測，不訓練模型；請上傳含 `model_win_probability`、`target_win`／`名次` 的逐馬預測 CSV，例如 `backtest_predictions_19feature.csv`。")
        saved_prediction_file = st.file_uploader("已保存的逐馬模型預測 CSV", type=["csv"], key="single_day_predictions_csv")
        target_date = st.date_input("選擇賽日", value=date.today(), key="single_day_backtest_date")
        if saved_prediction_file and st.button("開始單日回測", type="primary", key="single_day_backtest_button"):
            try:
                with st.spinner("正在比對該日逐馬預測與實際賽果…"):
                    rows = read_csv_bytes(saved_prediction_file.getvalue())
                    metrics, details, selected, place_details = single_date_backtest(
                        rows, target_date.isoformat(), float(stake), stake_per_place_bet=float(place_stake)
                    )
                roi = metrics["ROI"]
                st.success(f"{metrics['日期']} 回測完成：{metrics['賽事數']}場、{metrics['馬匹數']}匹；未重新訓練模型。")
                cols = st.columns(4)
                cols[0].metric("模型命中", f"{roi['模型首選命中場數']} / {roi['可結算賽事數']}")
                cols[1].metric("模型 ROI", f"{roi['模型首選ROI']:.1%}")
                cols[2].metric("模型淨盈虧", f"HK${roi['模型首選淨盈虧']:,.0f}")
                cols[3].metric("市場熱門 ROI", f"{roi['市場熱門ROI']:.1%}")
                if roi["無頭馬／未結算賽事略過數"]:
                    st.caption(f"略過 {roi['無頭馬／未結算賽事略過數']} 場沒有頭馬／未結算賽事；平頭派彩按頭馬數分攤。")
                comparison = pd.DataFrame([
                    {"策略": "模型每場最高分馬", "命中場數": roi["模型首選命中場數"], "命中率": roi["模型首選命中率"], "總投注(HK$)": roi["模型首選總投注"], "估算派彩(HK$)": roi["模型首選估算總派彩"], "淨盈虧(HK$)": roi["模型首選淨盈虧"], "ROI": roi["模型首選ROI"]},
                    {"策略": "市場最低賠率熱門", "命中場數": roi["市場熱門命中場數"], "命中率": roi["市場熱門命中率"], "總投注(HK$)": roi["市場熱門總投注"], "估算派彩(HK$)": roi["市場熱門估算總派彩"], "淨盈虧(HK$)": roi["市場熱門淨盈虧"], "ROI": roi["市場熱門ROI"]},
                ])
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                _render_place_results(roi["PLACE"], place_details, f"single_day_{target_date:%Y%m%d}")
                st.markdown("#### 逐場：模型首選、熱門及實際頭馬")
                st.dataframe(details, use_container_width=True, hide_index=True)
                st.download_button(
                    "下載單日逐場回測 CSV",
                    details.to_csv(index=False).encode("utf-8-sig"),
                    f"backtest_{target_date:%Y%m%d}_race_details.csv",
                    "text/csv",
                    key="download_single_day_details",
                )
                st.caption("這是已保存預測的單日分析；輸入資料若使用終盤賠率，結果是回顧性估算，不代表較早時段能取得相同賠率。")
            except Exception as exc:
                st.error(f"單日回測失敗：{exc}")
    else:
        st.write("上傳已完成賽事的 enriched CSV。程式按賽日切分 Train／Validation／未見 Test；同一賽日不會拆到不同集合。")
        backtest_file = st.file_uploader("已完成賽事訓練／回測 CSV", type=["csv"], key="backtest_csv")
        if backtest_file and st.button("開始時間切分回測", type="primary", key="backtest_button"):
            try:
                with st.spinner("正在建構19大特徵、以 Validation 選迭代數並評估未見 Test…"):
                    raw = read_csv_bytes(backtest_file.getvalue())
                    result = chronological_backtest(
                        raw, stake_per_race=float(stake), stake_per_place_bet=float(place_stake)
                    )
                metrics = result["metrics"]
                roi = metrics["ROI"]
                st.success("回測完成。候選模型僅以 Train + Validation 訓練，Test 留作未見評估。")
                split = metrics["切分"]
                st.caption(f"Train 截止（不含）{split['Train截止（不含）']}；Validation 截止／Test 開始 {split['Validation截止／Test開始']}；Test 最後日期 {split['Test最後日期']}。Test {split['Test場數']}場／{split['Test馬匹數']:,}匹。")
                st.markdown("#### 固定每場首選的 ROI 比較")
                cols = st.columns(4)
                cols[0].metric("模型命中", f"{roi['模型首選命中場數']} / {roi['可結算賽事數']}")
                cols[1].metric("模型淨盈虧", f"HK${roi['模型首選淨盈虧']:,.0f}")
                cols[2].metric("模型 ROI", f"{roi['模型首選ROI']:.1%}")
                cols[3].metric("市場熱門 ROI", f"{roi['市場熱門ROI']:.1%}")
                if roi["無頭馬／未結算賽事略過數"]:
                    st.caption(f"另略過 {roi['無頭馬／未結算賽事略過數']} 場沒有官方頭馬的賽事，不納入投注金額與 ROI。平頭派彩按頭馬數平均分攤。")
                compare = pd.DataFrame([
                    {"策略": "模型每場最高分馬", "有效賽事": roi["可結算賽事數"], "命中場數": roi["模型首選命中場數"], "命中率": roi["模型首選命中率"], "總投注(HK$)": roi["模型首選總投注"], "估算總派彩(HK$)": roi["模型首選估算總派彩"], "淨盈虧(HK$)": roi["模型首選淨盈虧"], "ROI": roi["模型首選ROI"]},
                    {"策略": "市場最低賠率熱門", "有效賽事": roi["可結算賽事數"], "命中場數": roi["市場熱門命中場數"], "命中率": roi["市場熱門命中率"], "總投注(HK$)": roi["市場熱門總投注"], "估算總派彩(HK$)": roi["市場熱門估算總派彩"], "淨盈虧(HK$)": roi["市場熱門淨盈虧"], "ROI": roi["市場熱門ROI"]},
                ])
                st.dataframe(compare, use_container_width=True, hide_index=True)
                _render_place_results(metrics["PLACE"], result["place_details"], "chronological_test")
                metric_rows = []
                for label, values in [("模型", metrics["模型概率指標"]), ("市場", metrics["市場概率指標"])]:
                    metric_rows.append({"概率來源": label, **values})
                st.markdown("#### 機率預測指標（Brier／Log loss 越低越好）")
                st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
                st.markdown("#### Test 逐場預測與頭馬")
                st.dataframe(result["race_details"], use_container_width=True, hide_index=True)
                c1, c2, c3 = st.columns(3)
                c1.download_button("下載Test逐馬預測 CSV", result["predictions"].to_csv(index=False).encode("utf-8-sig"), "backtest_test_predictions.csv", "text/csv", key="download_bt_predictions")
                c2.download_button("下載逐場回測／ROI CSV", result["race_details"].to_csv(index=False).encode("utf-8-sig"), "backtest_race_roi_details.csv", "text/csv", key="download_bt_race_details")
                c3.download_button("下載回測指標 JSON", json.dumps(metrics, ensure_ascii=False, indent=2).encode("utf-8"), "backtest_metrics.json", "application/json", key="download_bt_metrics")
                st.session_state["hkjc19_model"] = result["model"]
                st.session_state["hkjc19_model_bytes"] = model_bytes(result["model"])
                st.session_state["hkjc19_model_name"] = "my_hkjc_model_19feature_candidate.pkl"
                st.info("回測後候選模型已暫存在本工作階段，可到「預測賽事」分頁試用；工作階段結束後請下載模型保管。")
            except Exception as exc:
                st.error(f"回測失敗：{exc}")
