from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from hkjc_pipeline import (
    FEATURE_COLS,
    MODEL_PARAMS,
    chronological_backtest,
    load_model_bytes,
    model_bytes,
    predict_racecard,
    read_csv_bytes,
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
    st.write("上傳已算好19項特徵的 racecard CSV。必須包含有效獨贏賠率，名次欄需留空。輸出會新增模型分數及同場排名。")
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
    st.subheader("時間切分回測與 ROI")
    st.write("上傳已完成賽事的 enriched CSV。程式按賽日切分 Train／Validation／未見 Test；同一賽日不會拆到不同集合。")
    backtest_file = st.file_uploader("已完成賽事訓練／回測 CSV", type=["csv"], key="backtest_csv")
    stake = st.number_input("每場模型首選／市場熱門投注金額（HK$）", min_value=1.0, max_value=10000.0, value=10.0, step=1.0, key="roi_stake")
    st.warning("ROI 以 CSV 中的獨贏賠率估算每場買一注獨贏首選的派彩；不是香港賽馬會正式派彩結算。若賠率是終盤賠率，回測也不代表較早賽前可取得同一價格。")
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
            st.info("回測後候選模型已暫存在本工作階段，可到「預測賽事」分頁試用；下載並自行保管，工作階段結束後不會自動永久儲存。")
        except Exception as exc:
            st.error(f"回測失敗：{exc}")
