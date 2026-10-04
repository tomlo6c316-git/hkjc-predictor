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
tests/test_snapshot_backtest.py
tests/test_odds_refresh.py
tests/test_place_backtest.py
```

## 部署到 Streamlit Community Cloud

1. 在 GitHub 建立新 repository（建議新建19大專用 repo，避免覆蓋原15大專案）。
2. 將以上專案檔案放到 repository 根目錄；不要上傳訓練資料 CSV、歷史 enriched CSV、模型 `.pkl/.joblib` 或 Streamlit secrets。
3. 在 Streamlit Community Cloud 連接該 repo，主檔選 `app.py`，Python 版本選3.11或3.12，再部署。
4. `requirements.txt` 會安裝 Streamlit、LightGBM 與資料分析相依套件。

部署或修改後，可在專案目錄執行 `python -m unittest discover -s tests -v`，測試賽前快照回測的平頭馬、退出馬、取消賽事和缺漏賽果處理。

## 介面操作

### 1. 訓練模型

- 上傳已完成賽事的 enriched CSV。
- 訓練資料需有訓練器要求的官方欄位、19大歷史特徵來源欄位及有效完賽名次／獨贏賠率。資料來源欄位要求可參考 `train_hkjc_19feature_time_split.py` 的 `build_features()`。
- 系統按日期分 Train／Validation；以 Validation 的 early stopping 選最佳迭代數，再用全部已標籤資料重訓未來使用的候選模型。
- 下載模型 `.pkl` 並自行安全保管。模型只暫存在目前 Streamlit 工作階段，部署重啟或 session 結束後不保證仍在記憶體。

### 2. 預測賽事

- 可在頁面選擇未來賽日與馬場（ST／HV），上傳 enriched 歷史 CSV，再按 **抓取排位並產生19大 prediction CSV**。程式從 HKJC 官方頁抓取當日全場排位與WIN／PLACE賠率，使用賽日前歷史資料計算19大，並提供下載；WIN賠率尚未開放或有任何馬匹缺少有效WIN賠率時會停止、不輸出不完整檔。
- 官方賽卡若在馬名或狀態明確標示「退出／退賽」，下載時會排除該匹非參賽馬；普通在排位馬如缺少WIN賠率仍會停止輸出，不會當作已退出處理。
- 歷史 enriched CSV 只在本次請求內作為計算材料，不存入 GitHub；每次重新開啟／重新整理應用可能需重新上傳。它需包含 race ID、馬名／馬匹識別、名次、獨贏賠率、騎師、練馬師、場地、距離、`racecourse_code` 和 `official_distance_m`。
- 輸出的預測 CSV 會記錄賽日、馬場、HKJC賠率更新時間（若有）及產生時間；如已載入模型，還可選擇下載附模型排序的 CSV。
- 產生賽卡後，可按 **立即更新賠率**，或開啟自動更新並選擇每10／30／60／120／300秒。這是定時查詢，不是逐秒串流；需要應用頁保持開啟及連線。刷新只呼叫 HKJC 賠率 API，不重抓排位，也不重新計算歷史特徵；更新WIN／PLACE報價及市場特徵後重排模型結果。10秒間隔會較頻密地查詢HKJC；若遇到API限流或暫時錯誤，會保留最近一次有效快照並顯示錯誤。
- 在要記錄的賽前時點按 **保存本次賽前預測＋賠率快照**，再按 **下載所有已保存的賽前快照 CSV**。快照含唯一ID、香港時間、預測機率／排名、當時WIN賠率、19大特徵、模型名稱及可用時的模型SHA256。賠率會變動；每個決策時點都可另外保存一份。
- 快照只暫存在目前 Streamlit session 的記憶體；工作階段結束／服務重啟可能清除，**一定要下載快照 CSV 並自行保管**。請在預定作出投注決策的時間保存，不要在比賽後補存，也不要讓後續更新覆蓋已下載檔案。
- 使用者需先準備包含完整19項欄位的未開跑賽卡 CSV；可由既有 `futurecard` 流程產生。
- 上傳本次剛訓練的候選模型，或在同一個工作階段直接使用 session 中模型。
- 輸出含 `model_win_probability`、`model_race_probability` 與 `model_rank` 的預測 CSV。
- 如走「上傳已準備好的19大賽卡 CSV」流程，完成一次預測後可在該區按 **立即更新賠率**，或開啟10／30／60／120／300秒自動更新；系統會更新WIN／PLACE報價和模型排序。兩條抓卡路徑均有自己的刷新控制。
- 預測賽卡不可有已填名次；缺少任何特徵、無效賠率或重複「賽事編號＋馬號」會停止，不會補造賠率或特徵。

### 3. 回測與 ROI

- **賽前快照＋官方賽果回測**：上傳之前下載的快照 CSV，以及賽後官方賽果 CSV（需含 `賽事編號`、`馬號`、`名次`）。如快照檔含多個快照ID，先選一個要評估的時點。系統以「賽事編號＋馬號」匹配；任何快照馬匹在賽果檔找不到都會停止回測。官方結果應包含退出馬及其退出狀態；退出馬不當作落敗投注。
- 回測同時比較 **WIN每場一注首選**，以及 **模型／市場各自排序前三匹，每匹各買一注PLACE**。PLACE名次按HKJC本地賽規則：4–6匹出賽只計第一、二名；7匹或以上計第一至第三名；不足4匹不計。退出的已選馬匹視為退款／作廢，不以第四名自動替補。PLACE注命中率以選中注數計，也顯示「每場至少中一匹」場命中率。
- PLACE ROI使用賽前快照中保存的 `位置賠率`，每匹選馬的投注額獨立設定；前三匹最多三注。缺少任何有效位置賠率時，仍顯示名次命中統計，但不計部分ROI、不填造賠率。位置賠率只作估算，非HKJC官方派彩；平頭派彩、實際投注成交價可能不同。可下載WIN逐場明細、PLACE逐注明細及已配對逐馬資料。
- **指定賽日快速回測**：上傳已保存的 `backtest_predictions_19feature.csv`（需含 `model_win_probability`、`target_win`／`名次`；要計PLACE ROI須另有 `位置賠率`），選日期後只計算該日，不重新訓練。例如檢查2026-10-01，選 `2026-10-01`。
- **完整時間切分回測**：上傳已完成賽事的 enriched CSV；使用按賽日排序的 Train／Validation／Test，最後一段日期只用作未見 Test，不參與 early stopping 或擬合。此歷史檔通常沒有賽前位置賠率，因此可算PLACE名次命中，未有有效 `位置賠率` 時不會輸出PLACE ROI。
- 比較模型每場最高分馬與市場熱門的概率指標及固定每場獨贏投注 ROI。
- WIN每場投注及PLACE每匹選馬投注金額各預設 HK$10，可在介面分開調整；每場三匹PLACE全部下注時本金最多為三倍每匹投注額。賠率估算回報並非官方派彩結算。
- 下載 Test 逐馬預測、逐場ROI明細和 JSON 指標。

PLACE位置數規則參考：[HKJC Pari-Mutuel Local Pools 官方說明](https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/local-pools/)；本地賽4–6匹Declared Starters為頭二名，正常本地場7匹或以上按頭三名。實際派彩以HKJC正式規則及賽果為準。

## 固定模型參數

維持目前設定：`n_estimators=2000`、`learning_rate=0.03`、`num_leaves=31`、`max_depth=-1`、`min_child_samples=40`、`subsample=0.8`、`colsample_bytree=0.8`、`reg_lambda=2.0`、`early_stopping=75`，`random_state=42`。介面不提供修改模型參數功能。

## 重要限制

- 先回測，再用最新完賽資料訓練未來預測模型；訓練全資料後產生的模型已看過那些賽果，不能再把同一資料的預測成績當未見測試。
- 若回測 CSV 使用官方終盤賠率特徵，ROI 是終盤資訊下的回顧性估算；不代表比賽前較早時間可以取得同一賠率或預測。
- 19大模型和預測 CSV 必須使用同一19欄特徵名稱及順序。請不要把15大模型誤傳到此介面。
- 只載入自己或可信來源的 `.pkl/.joblib` 模型；這些檔案不應上傳至公開 GitHub repository。
- 預測分數不是賽果保證，也不保證長期獲利。
