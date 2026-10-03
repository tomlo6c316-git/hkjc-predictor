# HKJC 19大模型 Streamlit 工作台

本專案把19大模型的 **訓練、預測、時間切分回測與估算 ROI** 放到同一個 Streamlit 介面。訓練參數沿用目前版本，沒有做調參，也不會覆蓋舊的15大 app。

## 專案檔案

```text
app.py
hkjc_pipeline.py
train_hkjc_19feature_time_split.py
racecard_tools/futurecard.py
racecard_tools/hkjc_get_racecard.py
racecard_tools/hkjc_extra_features.py
requirements.txt
.gitignore
README.md
```

## 部署到 Streamlit Community Cloud

1. 在 GitHub 建立新 repository（建議新建19大專用 repo，避免覆蓋原15大專案）。
2. 將以上專案檔案放到 repository 根目錄；不要上傳訓練資料 CSV、歷史 enriched CSV、模型 `.pkl/.joblib` 或 Streamlit secrets。
3. 在 Streamlit Community Cloud 連接該 repo，主檔選 `app.py`，Python 版本選3.11或3.12，再部署。
4. `requirements.txt` 會安裝 Streamlit、LightGBM 與資料分析相依套件。

## 介面操作

### 1. 訓練模型

- 上傳已完成賽事的 enriched CSV。
- 訓練資料需有訓練器要求的官方欄位、19大歷史特徵來源欄位及有效完賽名次／獨贏賠率。資料來源欄位要求可參考 `train_hkjc_19feature_time_split.py` 的 `build_features()`。
- 系統按日期分 Train／Validation；以 Validation 的 early stopping 選最佳迭代數，再用全部已標籤資料重訓未來使用的候選模型。
- 下載模型 `.pkl` 並自行安全保管。模型只暫存在目前 Streamlit 工作階段，部署重啟或 session 結束後不保證仍在記憶體。

### 2. 預測賽事

- 可在頁面選擇未來賽日與馬場（ST／HV），上傳 enriched 歷史 CSV，再按 **抓取排位並產生19大 prediction CSV**。程式從 HKJC 官方頁抓取當日全場排位與賠率，使用賽日前歷史資料計算19大，並提供下載；賠率尚未開放或有任何馬匹缺少賠率時會停止、不輸出不完整檔，並列出缺漏馬匹與場次供查核。
- 官方賽卡若在馬名或狀態明確標示「退出／退賽」，下載時會排除該匹非參賽馬；普通在排位馬如缺少WIN賠率仍會停止輸出，不會當作已退出處理。
- 歷史 enriched CSV 只在本次請求內作為計算材料，不存入 GitHub；每次重新開啟／重新整理應用可能需重新上傳。它需包含 race ID、馬名／馬匹識別、名次、獨贏賠率、騎師、練馬師、場地、距離、`racecourse_code` 和 `official_distance_m`。
- 輸出的預測 CSV 會記錄賽日、馬場、HKJC賠率更新時間（若有）及產生時間；如已載入模型，還可選擇下載附模型排序的 CSV。
- 產生賽卡後，可按 **立即更新賠率**，或開啟自動更新並選擇每60／120／300秒。這是定時查詢，不是逐秒串流；需要應用頁保持開啟及連線。刷新只呼叫 HKJC 賠率 API，不重抓排位，也不重新計算歷史特徵；只更新 WIN／PLACE 快照、四個依市場變動的模型欄位（`獨贏賠率`、`market_implied_prob`、`odds_rank`、`is_favorite`）並重排模型結果。API 沒有完整回傳時會保留最近一次有效快照並顯示錯誤。
- 使用者需先準備包含完整19項欄位的未開跑賽卡 CSV；可由既有 `futurecard` 流程產生。
- 上傳本次剛訓練的候選模型，或在同一個工作階段直接使用 session 中模型。
- 輸出含 `model_win_probability`、`model_race_probability` 與 `model_rank` 的預測 CSV。
- 預測賽卡不可有已填名次；缺少任何特徵、無效賠率或重複「賽事編號＋馬號」會停止，不會補造賠率或特徵。

### 3. 回測與 ROI

- **指定賽日快速回測**：上傳已保存的 `backtest_predictions_19feature.csv`（需含 `model_win_probability` 及 `target_win`／`名次`），選日期後只計算該日，不重新訓練。例如檢查2026-10-01，選 `2026-10-01`。
- **完整時間切分回測**：上傳已完成賽事的 enriched CSV；使用按賽日排序的 Train／Validation／Test，最後一段日期只用作未見 Test，不參與 early stopping 或擬合。
- 比較模型每場最高分馬與市場熱門的概率指標及固定每場獨贏投注 ROI。
- 每場投注金額預設 HK$10，可在介面調整；程式以 CSV 的獨贏賠率估算回報，不是官方派彩結算。
- 下載 Test 逐馬預測、逐場ROI明細和 JSON 指標。

## 固定模型參數

維持目前設定：`n_estimators=2000`、`learning_rate=0.03`、`num_leaves=31`、`max_depth=-1`、`min_child_samples=40`、`subsample=0.8`、`colsample_bytree=0.8`、`reg_lambda=2.0`、`early_stopping=75`，`random_state=42`。介面不提供修改模型參數功能。

## 重要限制

- 先回測，再用最新完賽資料訓練未來預測模型；訓練全資料後產生的模型已看過那些賽果，不能再把同一資料的預測成績當未見測試。
- 若回測 CSV 使用官方終盤賠率特徵，ROI 是終盤資訊下的回顧性估算；不代表比賽前較早時間可以取得同一賠率或預測。
- 19大模型和預測 CSV 必須使用同一19欄特徵名稱及順序。請不要把15大模型誤傳到此介面。
- 只載入自己或可信來源的 `.pkl/.joblib` 模型；這些檔案不應上傳至公開 GitHub repository。
- 預測分數不是賽果保證，也不保證長期獲利。
