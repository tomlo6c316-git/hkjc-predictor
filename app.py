from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from hkjc_pipeline import (
    FEATURE_COLS,
    MODEL_PARAMS,
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
    st.write("可直接選日期抓取 HKJC 官方排位與當前獨贏賠率，搭配歷史 enriched CSV 計算19大特徵，再下載單一 prediction CSV。")
    active_model = st.session_state.get("hkjc19_model")
    model_file = st.file_uploader("可選：載入已訓練的19大模型（joblib/pkl）", type=["pkl", "joblib"], key="prediction_model")
    if model_file is not None:
        try:
            active_model = load_model_bytes(model_file.getvalue())
            st.session_state["hkjc19_model"] = active_model
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
            refresh_interval = st.selectbox("間隔（秒）", [60, 120, 300], index=0, key="odds_refresh_interval")

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
            st.session_state["hkjc19_last_prediction"] = predicted
            st.success(f"預測完成：{predicted['賽事編號'].nunique()}場、{len(predicted)}匹。")
            show_cols = [c for c in ["賽事編號", "馬號", "馬名", "獨贏賠率", "model_win_probability", "model_race_probability", "model_rank"] if c in predicted.columns]
            st.dataframe(predicted[show_cols], use_container_width=True, hide_index=True)
            st.download_button(
                "下載預測結果 CSV",
                data=predicted.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{race_file.name.rsplit('.', 1)[0]}_predicted.csv",
                mime="text/csv",
                key="download_predictions",
            )
            st.caption("`model_win_probability` 為模型二元分類分數；`model_race_probability` 是同場正規化供排名參考，不保證校準或賽果。賠率會變動，請注意輸入 CSV 的賠率快照時間。")
        except Exception as exc:
            st.error(f"預測失敗：{exc}")

with backtest_tab:
    st.subheader("賽事回測與 ROI")
    backtest_mode = st.radio(
        "選擇回測方式",
        ["單一賽日快速回測（用已保存的模型預測）", "完整時間切分回測（重新訓練並評估）"],
        horizontal=True,
        key="backtest_mode",
    )
    stake = st.number_input("每場模型首選／市場熱門投注金額（HK$）", min_value=1.0, max_value=10000.0, value=10.0, step=1.0, key="roi_stake")
    st.warning("ROI 以 CSV 中的獨贏賠率估算每場買一注獨贏首選的派彩；不是香港賽馬會正式派彩結算。若賠率是終盤賠率，回測也不代表較早賽前可取得同一價格。")

    if backtest_mode.startswith("單一賽日"):
        st.write("此模式只篩選已保存的測試預測，不訓練模型；請上傳含 `model_win_probability`、`target_win`／`名次` 的逐馬預測 CSV，例如 `backtest_predictions_19feature.csv`。")
        saved_prediction_file = st.file_uploader("已保存的逐馬模型預測 CSV", type=["csv"], key="single_day_predictions_csv")
        target_date = st.date_input("選擇賽日", value=date.today(), key="single_day_backtest_date")
        if saved_prediction_file and st.button("開始單日回測", type="primary", key="single_day_backtest_button"):
            try:
                with st.spinner("正在比對該日逐馬預測與實際賽果…"):
                    rows = read_csv_bytes(saved_prediction_file.getvalue())
                    metrics, details, selected = single_date_backtest(rows, target_date.isoformat(), float(stake))
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
                    result = chronological_backtest(raw, stake_per_race=float(stake))
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
