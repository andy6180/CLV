# CLV 顧客價值分析系統

靜宜大學資料科學暨大數據分析與應用學系　CLV 顧客終身價值專題  
指導老師：袁淵明老師｜成員：范言提、蘇廷尉、陳均睿、李俊鋧

上傳交易資料後，系統會：
1. 檢查資料品質，彙總成訂單層級
2. 以 Frequency、Monetary 進行 K-Means 分群（忠誠客戶／VIP 客戶／一般客戶），呈現肘部法則與 PCA
3. 以 BG/NBD × Gamma-Gamma 預測未來 3／6／9／12 個月的 CLV

## 執行
```
pip install -r requirements.txt
streamlit run app.py
```

## 資料格式
必要欄位：顧客 ID、訂單 ID、交易日期、交易金額（或數量 × 單價）。選配：訂單狀態。  
Olist 原始三張表可先用 `python prepare_olist.py 資料夾路徑` 整理成單一交易檔。
