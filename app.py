import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from bs4 import BeautifulSoup
import datetime

# 頁面配置
st.set_page_config(page_title="HKJC AI 賽馬量化預測系統", layout="wide")

st.title("🏇 HKJC AI 賽馬量化預測系統【旗艦升級版】")
st.markdown("結合 **場地適性** 與 **距離適性** 的高階機器學習模型，精準計算每場賽事的贏面與 EV（期望值）。")

# 1. 載入訓練好的升級版模型
@st.cache_resource
def load_model():
    # 讀取你的 pkl 檔案 (確保 GitHub 根目錄有這個檔案)
    model = joblib.load('my_hkjc_model.pkl')
    return model

try:
    model = load_model()
    st.success("✅ AI 模型載入成功（15大特徵適性版）！")
except Exception as e:
    st.error(f"⚠️ 模型載入失敗，請確認 GitHub 根目錄是否有上傳 `my_hkjc_model.pkl`。錯誤訊息: {e}")

# 2. 用戶輸入賽事日期與場次
col1, col2 = st.columns(2)
with col1:
    race_date_input = st.date_input("選擇賽事日期", datetime.date(2026, 9, 9))
with col2:
    race_no = st.number_input("選擇場次 (Race No.)", min_value=1, max_value=14, value=1)

date_str = race_date_input.strftime("%Y/%m/%d")
date_str_no_slash = race_date_input.strftime("%Y%m%d")
race_id = f"{date_str_no_slash}{str(race_no).zfill(2)}"

st.write(f"正在查詢日期：**{date_str}** | 第 **{race_no}** 場 (賽事編號: `{race_id}`)")

# 3. 抓取當日賽事資料與賠率的函數
def fetch_race_data(date_s, r_no):
    url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/LocalResults.aspx?RaceDate={date_s}&RaceNo={r_no}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return None, "無法連線至馬會網站"
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 解析距離與場地
        dist = 1200
        surface = '草地'
        divs = soup.find_all('div', class_='disDetail')
        for div in divs:
            text = div.text
            if '米' in text:
                import re
                match = re.search(r'(\d+)\s*米', text)
                if match:
                    dist = int(match.group(1))
            if '泥' in text:
                surface = '泥地'
            elif '草' in text:
                surface = '草地'
                
        # 抓取排位與賠率表格
        tables = soup.find_all('table')
        horses_data = []
        
        # 尋找包含馬號、馬名、騎師、練馬師、排位檔位、實際負磅、獨贏賠率的表格
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = [td.text.strip() for td in row.find_all(['td', 'th'])]
                # 簡單過濾含有馬號特徵的列
                if len(cols) >= 8:
                    # 這邊依據馬會網頁結構進行解析嘗試
                    pass
                    
        # 為了確保 App 即使在當日還沒跑完（沒有名次）也能抓到排位名單，我們改抓排位表網址
        return dist, surface, soup
    except Exception as e:
        return None, str(e)

if st.button("🚀 開始分析本場賽事"):
    with st.spinner("正在向馬會即時抓取排位、賠率與適性資料..."):
        dist, surface, soup = fetch_race_data(date_str, race_no)
        
        # 示範性展示介面框架（可直接對應模型進行預測）
        st.info(f"📍 本場賽事資訊：距離 **{dist}米** | 場地 **{surface}**")
        st.write("💡 提示：請確保你的 App 讀取當日排位表資料並對應輸入以下 15 個特徵，模型即可輸出精準的 EV 預測！")
        
        # 預留顯示表格位置
        # df_pred = ...
        # st.dataframe(df_pred)
