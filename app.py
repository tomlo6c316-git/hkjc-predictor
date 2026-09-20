import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# 頁面基本設定
st.set_page_config(
    page_title="HKJC AI 智能賽馬預測系統",
    page_icon="🐎",
    layout="wide"
)

st.title("🐎 HKJC 旗艦 15 大特徵 AI 預測系統")
st.markdown("---")

# 設定模型路徑
MODEL_PATH = 'my_hkjc_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error(f"⚠️ 找不到 AI 模型檔 `{MODEL_PATH}`！請確認模型是否已上傳至正確目錄。")
    st.stop()

# 側邊欄：檔案上傳區與參數設定
st.sidebar.header("📂 資料載入")
uploaded_file = st.sidebar.file_uploader("請上傳賽事資料 (CSV)", type=['csv'])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 投注策略參數設定")
min_ev = st.sidebar.slider("最小期望值 (EV 門檻)", 0.0, 1.5, 0.0, 0.05)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=3.0)
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=20.0)

if uploaded_file is not None:
    # 雙編碼容錯讀取
    try:
        df_raw = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    except:
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, encoding='cp950')
        
    st.success("✅ 賽事資料載入成功！")
    
    df = df_raw.copy()
    
    # 1. 特徵工程
    df['獨贏賠率'] = pd.to_numeric(df['獨贏賠率'], errors='coerce').fillna(10.0)
    df['market_prob'] = 1 / df['獨贏賠率']
    prob_sum = df.groupby('賽事編號')['market_prob'].transform('sum')
    df['market_implied_prob'] = df['market_prob'] / prob_sum

    df['odds_rank'] = df.groupby('賽事編號')['獨贏賠率'].rank(method='min')
    df['is_favorite'] = (df['odds_rank'] == 1).astype(int)
    df['排位檔位'] = pd.to_numeric(df['排位檔位'], errors='coerce').fillna(7)

    if '實際負磅' in df.columns:
        df['實際負磅'] = pd.to_numeric(df['實際負磅'], errors='coerce').fillna(120)
        avg_weight = df.groupby('賽事編號')['實際負磅'].transform('mean')
        df['weight_diff'] = df['實際負磅'] - avg_weight
        df['weight_rank'] = df.groupby('賽事編號')['實際負磅'].rank(ascending=False, method='min')
    else:
        df['weight_diff'] = 0.0
        df['weight_rank'] = 6.0

    # 解析名次（用於回測判斷）
    df['numeric_rank'] = pd.to_numeric(df.get('名次', 99), errors='coerce').fillna(99)

    df['jockey_win_rate'] = df.get('jockey_win_rate', 0.12)
    df['trainer_win_rate'] = df.get('trainer_win_rate', 0.12)
    df['combo_win_rate'] = df.get('combo_win_rate', 0.10)
    df['horse_win_rate'] = df.get('horse_win_rate', 0.10)
    df['horse_last_rank'] = df.get('horse_last_rank', 6.0)

    if '距離' not in df.columns: df['距離'] = 1200
    if 'horse_surface_win_rate' not in df.columns: df['horse_surface_win_rate'] = 0.08
    if 'horse_dist_win_rate' not in df.columns: df['horse_dist_win_rate'] = 0.08

    # 位置賠率處理：若 CSV 無真實位置賠率，則依獨贏賠率自動轉換估算
    if '位置賠率' in df.columns:
        df['位置賠率'] = pd.to_numeric(df['位置賠率'], errors='coerce')
    else:
        df['位置賠率'] = 1.0 + (df['獨贏賠率'] - 1.0) / 3.2

    # 2. 提取 15 大特徵並預測勝率與 EV
    feature_cols = [
        'market_implied_prob', '獨贏賠率', 'odds_rank', 'is_favorite',
        '排位檔位', 'weight_diff', 'weight_rank',
        'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
        'horse_win_rate', 'horse_last_rank',
        '距離', 'horse_surface_win_rate', 'horse_dist_win_rate'
    ]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X_predict = df[feature_cols].fillna(0)

    df['pred_win_prob'] = model.predict_proba(X_predict)[:, 1]
    df['ev'] = df['pred_win_prob'] * df['獨贏賠率']

    # ==========================================
    # 建立雙分頁 UI (預測 vs 回測)
    # ==========================================
    tab1, tab2 = st.tabs(["🎯 各場次預測推薦", "📈 歷史回測 (獨贏/位置/位置Q)"])

    # ---------------- 分頁 1: 賽前預測 ----------------
    with tab1:
        st.subheader("🎯 各場次 AI 智慧投注推薦清單")

        recommendations = []
        for race_id, group in df.groupby('賽事編號'):
            filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
            sorted_group = filtered_group.sort_values(by='ev', ascending=False).reset_index(drop=True)

            if len(sorted_group) >= 2:
                top1 = sorted_group.iloc[0]
                top2 = sorted_group.iloc[1]
                top3 = sorted_group.iloc[2] if len(sorted_group) >= 3 else top2

                win_pick = f"馬號 {top1['馬號']} ({top1['馬名']}) [勝率:{top1['pred_win_prob']*100:.1f}%, EV:{top1['ev']:.2f}]"
                q_pick = f"{top1['馬號']} + {top2['馬號']} ({top1['馬名']} / {top2['馬名']})"
                qp_pick = f"{top1['馬號']} + {top2['馬號']} 或 {top1['馬號']} + {top3['馬號']}"

                recommendations.append({
                    '賽事編號': race_id,
                    '🎯 獨贏推薦': win_pick,
                    '🔗 連贏推薦 (Q)': q_pick,
                    '🔗 位置Q推薦 (QP)': qp_pick
                })

        rec_df = pd.DataFrame(recommendations)
        if rec_df.empty:
            st.warning("⚠️ 沒有符合當前篩選條件的馬匹，請試著放寬左側欄的 EV 或賠率限制。")
        else:
            st.dataframe(rec_df, use_container_width=True)

    # ---------------- 分頁 2: 賽後回測 ----------------
    with tab2:
        st.subheader("📊 多彩種策略回測總覽")
        
        # 防呆檢查：確認上傳的檔案裡有沒有真實的「名次」數據
        if (df['numeric_rank'] == 99).all():
            st.warning("⚠️ 系統偵測到目前的 CSV 中沒有真實的『名次』紀錄 (或全部為空值)。請上傳已完賽並包含名次結果的歷史檔案來執行回測。")
        else:
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🥇 獨贏 (Win)", "🥈 位置 (Place)", "🔗 位置Q (QP)"])
            BET_AMOUNT = 100 
            
            # 1. 獨贏 (Win) 回測
            with sub_tab1:
                st.markdown("#### 🥇 獨贏 (Win) 策略回測")
                win_invested, win_return, win_bets, win_hits = 0, 0, 0, 0
                win_records = []
                
                for race_id, group in df.groupby('賽事編號'):
                    filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                    sorted_group = filtered_group.sort_values(by='ev', ascending=False).reset_index(drop=True)
                    
                    if len(sorted_group) > 0:
                        pick = sorted_group.iloc[0]
                        win_bets += 1
                        win_invested += BET_AMOUNT
                        
                        is_hit = (pick['numeric_rank'] == 1)
                        if is_hit:
                            win_hits += 1
                            payout = BET_AMOUNT * pick['獨贏賠率']
                            win_return += payout
                            result_str = "✅ 贏"
                        else:
                            payout = 0
                            result_str = "❌ 輸"
                            
                        win_records.append({
                            '賽事編號': race_id,
                            '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                            '實際名次': str(pick['名次']).replace('.0', ''),
                            '獨贏賠率': pick['獨贏賠率'],
                            'EV': round(pick['ev'], 2),
                            '結果': result_str,
                            '派彩': f"${payout:.1f}",
                            '淨盈虧': f"${payout - BET_AMOUNT:.1f}"
                        })
                
                if win_bets > 0:
                    roi = ((win_return - win_invested) / win_invested) * 100
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("投注場數", f"{win_bets} 場")
                    col2.metric("命中場數", f"{win_hits} 場", f"勝率: {win_hits/win_bets*100:.1f}%")
                    col3.metric("總成本", f"${win_invested}")
                    col4.metric("總回收", f"${win_return:.1f}", f"ROI: {roi:.2f}%")
                    
                    st.markdown("##### 📝 獨贏明細")
                    st.dataframe(pd.DataFrame(win_records), use_container_width=True)
                else:
                    st.info("💡 目前設定下沒有符合獨贏出手的場次。")

            # 2. 位置 (Place) 回測
            with sub_tab2:
                st.markdown("#### 🥈 位置 (Place) 策略回測 (目標：跑入前三名)")
                place_invested, place_return, place_bets, place_hits = 0, 0, 0, 0
                place_records = []
                
                for race_id, group in df.groupby('賽事編號'):
                    filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                    sorted_group = filtered_group.sort_values(by='ev', ascending=False).reset_index(drop=True)
                    
                    if len(sorted_group) > 0:
                        pick = sorted_group.iloc[0]
                        place_bets += 1
                        place_invested += BET_AMOUNT
                        
                        is_hit = (pick['numeric_rank'] <= 3)
                        p_odds = float(pick['位置賠率']) if pd.notna(pick['位置賠率']) and float(pick['位置賠率']) > 1.0 else 1.5
                        
                        if is_hit:
                            place_hits += 1
                            payout = BET_AMOUNT * p_odds
                            place_return += payout
                            result_str = "✅ 命中位置"
                        else:
                            payout = 0
                            result_str = "❌ 未入前三"
                            
                        place_records.append({
                            '賽事編號': race_id,
                            '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                            '實際名次': str(pick['名次']).replace('.0', ''),
                            '位置賠率': round(p_odds, 2),
                            'EV': round(pick['ev'], 2),
                            '結果': result_str,
                            '派彩': f"${payout:.1f}",
                            '淨盈虧': f"${payout - BET_AMOUNT:.1f}"
                        })
                
                if place_bets > 0:
                    roi = ((place_return - place_invested) / place_invested) * 100
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("投注場數", f"{place_bets} 場")
                    col2.metric("命中位置場數", f"{place_hits} 場", f"位置勝率: {place_hits/place_bets*100:.1f}%")
                    col3.metric("總成本", f"${place_invested}")
                    col4.metric("總回收", f"${place_return:.1f}", f"ROI: {roi:.2f}%")
                    
                    st.markdown("##### 📝 位置明細")
                    st.dataframe(pd.DataFrame(place_records), use_container_width=True)
                else:
                    st.info("💡 目前設定下沒有符合位置出手的場次。")

            # 3. 位置Q (Quinella Place) 回測
            with sub_tab3:
                st.markdown("#### 🔗 位置Q (QP) 策略回測 (挑選 EV 最高前兩匹，雙雙跑入前三名)")
                qp_invested, qp_return, qp_bets, qp_hits = 0, 0, 0, 0
                qp_records = []
                
                for race_id, group in df.groupby('賽事編號'):
                    filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                    sorted_group = filtered_group.sort_values(by='ev', ascending=False).reset_index(drop=True)
                    
                    if len(sorted_group) >= 2:
                        top1 = sorted_group.iloc[0]
                        top2 = sorted_group.iloc[1]
                        
                        qp_bets += 1
                        qp_invested += BET_AMOUNT
                        
                        is_hit = (top1['numeric_rank'] <= 3) and (top2['numeric_rank'] <= 3)
                        
                        p1_odds = float(top1['位置賠率']) if pd.notna(top1['位置賠率']) else 1.5
                        p2_odds = float(top2['位置賠率']) if pd.notna(top2['位置賠率']) else 1.5
                        estimated_qp_odds = round(p1_odds * p2_odds * 1.8, 1)
                        
                        if is_hit:
                            qp_hits += 1
                            payout = BET_AMOUNT * estimated_qp_odds
                            qp_return += payout
                            result_str = "✅ 命中位置Q"
                        else:
                            payout = 0
                            result_str = "❌ 落空"
                            
                        qp_records.append({
                            '賽事編號': race_id,
                            'QP 組合': f"{top1['馬號']} + {top2['馬號']} ({top1['馬名']} / {top2['馬名']})",
                            '實際名次': f"首選:第{str(top1['名次']).replace('.0','')}名 | 次選:第{str(top2['名次']).replace('.0','')}名",
                            '估算QP賠率': estimated_qp_odds,
                            '結果': result_str,
                            '派彩': f"${payout:.1f}",
                            '淨盈虧': f"${payout - BET_AMOUNT:.1f}"
                        })
                
                if qp_bets > 0:
                    roi = ((qp_return - qp_invested) / qp_invested) * 100
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("投注場數", f"{qp_bets} 場")
                    col2.metric("命中 QP 場數", f"{qp_hits} 場", f"QP 命中率: {qp_hits/qp_bets*100:.1f}%")
                    col3.metric("總成本", f"${qp_invested}")
                    col4.metric("總回收", f"${qp_return:.1f}", f"ROI: {roi:.2f}%")
                    
                    st.markdown("##### 📝 位置Q (QP) 明細")
                    st.dataframe(pd.DataFrame(qp_records), use_container_width=True)
                else:
                    st.info("💡 目前設定下沒有符合位置 Q 出手的場次 (需同場至少有 2 匹馬符合 EV/賠率門檻)。")

else:
    st.info("👈 請在左側上傳賽事 CSV 檔案來啟動系統！")
