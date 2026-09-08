import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import joblib
import os

st.set_page_config(page_title="HKJC AI 智能量化預測", page_icon="🏇", layout="wide")

st.title("🏇 香港賽馬 AI 智能量化預測系統")
st.markdown("支援 **獨贏 (Win) / 連贏 (Q) / 位置Q (QP)** 多維度價值投注分析")

st.sidebar.header("⚙️ 賽事設定")
race_date_input = st.sidebar.text_input("輸入賽事日期 (格式: YYYY/MM/DD)", "2026/09/09")
ev_threshold = st.sidebar.slider("EV 期望值門檻", 1.0, 1.5, 1.15, 0.05)
min_odds = st.sidebar.slider("最低賠率門檻", 1.0, 20.0, 3.0, 0.5)

run_button = st.sidebar.button("🚀 開始分析預測")

@st.cache_resource
def load_model():
    model_path = 'my_hkjc_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

def fetch_and_predict(target_date, model):
    formatted_date_url = target_date.replace('/', '')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    all_races_data = []
    
    with st.spinner(f"正在連線馬會抓取 {target_date} 賽事數據與賠率..."):
        for race_no in range(1, 13):
            url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/LocalResults.aspx?RaceDate={target_date}&RaceNo={race_no}"
            try:
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code != 200: continue
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                tables = soup.find_all('table')
                if len(tables) < 3: continue
                
                odds_dict = {}
                for table in tables:
                    for row in table.find_all('tr'):
                        cols = [td.text.strip() for td in row.find_all(['td', 'th'])]
                        if len(cols) >= 3 and cols[0] in ['獨贏', 'Win']:
                            try: odds_dict[cols[1]] = float(cols[2].replace(',', ''))
                            except: continue

                main_table = None
                for table in tables:
                    if '名次' in table.text and '馬號' in table.text and '騎師' in table.text:
                        main_table = table
                        break
                
                if main_table:
                    for row in main_table.find_all('tr'):
                        cols = [td.text.strip() for td in row.find_all(['td', 'th'])]
                        if len(cols) >= 8 and cols[0].isdigit():
                            try:
                                fin_pos = int(cols[0])
                                horse_no = cols[1]
                                horse_name = cols[2]
                                jockey = cols[3]
                                trainer = cols[4]
                                weight = cols[5]
                                draw = cols[7] if len(cols) > 7 else "7"
                                odds = odds_dict.get(horse_no, 0.0)
                                if odds == 0.0:
                                    for col in cols[8:]:
                                        try:
                                            val = float(col.replace(',', ''))
                                            if 1.0 <= val <= 200.0:
                                                odds = val
                                                break
                                        except: continue
                                            
                                all_races_data.append({
                                    '賽事編號': f"{formatted_date_url}-{race_no:02d}",
                                    '場次': f"第 {race_no} 場",
                                    '名次': fin_pos, '馬號': horse_no, '馬名': horse_name,
                                    '騎師': jockey, '練馬師': trainer, '實際負磅': weight,
                                    '排位檔位': draw, '獨贏賠率': odds
                                })
                            except: continue
            except: continue
                
    if not all_races_data: return None
    df = pd.DataFrame(all_races_data)
    
    df['獨贏賠率'] = pd.to_numeric(df['獨贏賠率'], errors='coerce').fillna(10.0)
    df['market_prob'] = 1 / df['獨贏賠率']
    prob_sum = df.groupby('賽事編號')['market_prob'].transform('sum')
    df['market_implied_prob'] = df['market_prob'] / prob_sum

    df['odds_rank'] = df.groupby('賽事編號')['獨贏賠率'].rank(method='min')
    df['is_favorite'] = (df['odds_rank'] == 1).astype(int)
    df['排位檔位'] = pd.to_numeric(df['排位檔位'], errors='coerce').fillna(7)
    df['實際負磅'] = pd.to_numeric(df['實際負磅'], errors='coerce').fillna(120)
    avg_weight = df.groupby('賽事編號')['實際負磅'].transform('mean')
    df['weight_diff'] = df['實際負磅'] - avg_weight
    df['weight_rank'] = df.groupby('賽事編號')['實際負磅'].rank(ascending=False, method='min')

    df['jockey_win_rate'] = 0.12
    df['trainer_win_rate'] = 0.12
    df['combo_win_rate'] = 0.10
    df['horse_win_rate'] = 0.10
    df['horse_last_rank'] = 6.0

    feature_cols = [
        'market_implied_prob', '獨贏賠率', 'odds_rank', 'is_favorite',
        '排位檔位', 'weight_diff', 'weight_rank',
        'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
        'horse_win_rate', 'horse_last_rank'
    ]
    
    if model is not None:
        df['pred_win_prob'] = model.predict(df[feature_cols])
        df['ev'] = df['pred_win_prob'] * df['獨贏賠率']
    else:
        df['pred_win_prob'] = 0.1
        df['ev'] = 1.0

    return df

if run_button:
    if model is None:
        st.error("⚠️ 找不到模型檔案 `my_hkjc_model.pkl`！")
    else:
        result_df = fetch_and_predict(race_date_input, model)
        if result_df is not None and not result_df.empty:
            st.success(f"🎉 成功完成 {race_date_input} 賽事分析！")
            for race_id, group in result_df.groupby('賽事編號'):
                sorted_group = group.sort_values(by='ev', ascending=False).reset_index(drop=True)
                race_name = sorted_group.iloc[0]['場次']
                with st.expander(f"📍 {race_name} (賽事編號: {race_id})"):
                    filtered = sorted_group[(sorted_group['ev'] >= ev_threshold) & (sorted_group['獨贏賠率'] >= min_odds)]
                    if not filtered.empty:
                        st.markdown("### 🔥 AI 價值投注推薦")
                        formatted_display = filtered[['馬號', '馬名', '騎師', '練馬師', '獨贏賠率']].copy()
                        formatted_display['預測勝率'] = (filtered['pred_win_prob'] * 100).round(1).astype(str) + '%'
                        formatted_display['期望值(EV)'] = filtered['ev'].round(2)
                        st.dataframe(formatted_display, use_container_width=True)
                        
                        if len(sorted_group) >= 2:
                            top1, top2, top3 = sorted_group.iloc[0], sorted_group.iloc[1], sorted_group.iloc[2] if len(sorted_group) >= 3 else sorted_group.iloc[1]
                            suggestion_text = (
    f"💡 **組合建議**：\n"
    f"- **連贏(Q)**：{top1['馬號']} + {top2['馬號']} ({top1['馬名']} / {top2['馬名']})\n"
    f"- **位置Q(QP)**：{top1['馬號']} + {top2['馬號']} 或 {top1['馬號']} + {top3['馬號']}"
)
st.info(suggestion_text)
                    else:
                        st.warning("此場賽事暫無符合門檻的高 EV 馬匹。")
                    
                    st.markdown("---")
                    st.markdown("📊 **全場馬匹 AI 完整評分總表**")
                    full_view = sorted_group[['馬號', '馬名', '獨贏賠率']].copy()
                    full_view['預測勝率'] = (sorted_group['pred_win_prob'] * 100).round(1).astype(str) + '%'
                    full_view['EV'] = sorted_group['ev'].round(2)
                    st.dataframe(full_view, use_container_width=True)
        else:
            st.error("⚠️ 無法獲取該日期的賽事資料。")
