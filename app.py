# -*- coding: utf-8 -*-
"""
app.py — CLV 顧客價值分析系統（Streamlit 前端）

  ① 資料上傳：欄位設定、資料品質檢查、彙總成訂單層級
  ② 分析結果：只呈現五個結果
     1. 肘部法則　2. PCA 特徵對比　3. 客戶價值人數與百分比
     4. 未來 CLV 預測結果（BG/NBD × Gamma-Gamma）　5. 顧客未來 CLV 分客群

執行：streamlit run app.py
"""

import pandas as pd
import streamlit as st

import analysis as A

st.set_page_config(page_title="CLV 顧客價值分析系統", page_icon="📊", layout="wide")

NO_COLUMN = "（不使用）"

# ============================================================
# 退貨抵銷
# ============================================================

def full_cancellations(orders):
    """
    找出被整張退貨的原訂單：退貨單（金額 < 0）若與同一位顧客、在退貨之前（含同時）的
    某張訂單金額完全相反，視為那張訂單被整張取消，原訂單也要排除，否則營收會被高估。
    部分退貨無法對應到單一訂單，仍只排除退貨單本身。
    回傳要排除的原訂單 order_id 集合。
    """
    pos = orders[orders["amount"] > 0].sort_values("order_date")
    neg = orders[orders["amount"] < 0].sort_values("order_date")
    pool = {}
    for r in pos.itertuples(index=False):
        pool.setdefault((r.customer_id, round(r.amount, 2)), []).append((r.order_date, r.order_id))
    drop = set()
    for r in neg.itertuples(index=False):
        cands = pool.get((r.customer_id, round(-r.amount, 2)), [])
        for i in range(len(cands) - 1, -1, -1):
            if cands[i][0] <= r.order_date:
                drop.add(cands.pop(i)[1])
                break
    return drop


# ============================================================
# Session State
# ============================================================

for key, default in [("data_ready", False), ("orders", None), ("result", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def reset_results():
    st.session_state.result = None


# ============================================================
# 標題與導覽
# ============================================================

st.title("CLV 顧客價值分析系統")
st.caption("上傳交易資料後，系統會用 RFM + K-Means 分群，再用 BG/NBD 與 Gamma-Gamma 預測未來 CLV。")

pages = ["資料上傳", "分析結果"]
selected_page = st.radio("功能選單", pages, horizontal=True, label_visibility="collapsed")

if not st.session_state.data_ready and selected_page != "資料上傳":
    st.warning("請先完成資料上傳與資料檢查，才能查看分析結果。")
    st.stop()


def find_column(columns, keywords):
    """依關鍵字的優先順序尋找欄位（先比對完整名稱，再比對部分名稱）。"""
    lowered = {str(c).lower(): c for c in columns}
    for kw in keywords:
        if kw.lower() in lowered:
            return lowered[kw.lower()]
    for kw in keywords:
        for low, original in lowered.items():
            if kw.lower() in low:
                return original
    return None


def download_button(df, label, filename):
    st.download_button(
        label, df.to_csv(index=False).encode("utf-8-sig"), file_name=filename, mime="text/csv"
    )


# ============================================================
# ① 資料上傳
# ============================================================

if selected_page == "資料上傳":
    st.header("資料上傳")
    st.write("請上傳交易資料（CSV 或 XLSX），每一列為一筆交易或一筆付款紀錄。")

    uploaded_file = st.file_uploader("上傳檔案", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("必要欄位：顧客 ID、訂單 ID、交易日期、交易金額（或數量 + 單價）。選配欄位：訂單狀態。\n\n"
                "可分析的條件：有效訂單 ≥ 100 筆、顧客 ≥ 20 人、資料期間 ≥ 180 天、回購顧客 ≥ 20 人。")
        st.stop()

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:  # noqa: BLE001
        st.error(f"檔案讀取失敗：{e}")
        st.stop()

    c1, c2 = st.columns(2)
    c1.metric("資料筆數", f"{len(df):,}")
    c2.metric("欄位數", f"{len(df.columns):,}")
    st.dataframe(df.head(10).astype(str), width="stretch")

    st.divider()
    st.subheader("資料欄位設定")
    st.write("系統會自動推薦欄位，判斷錯誤時可以手動修改。")

    columns = list(df.columns)
    guesses = {
        "customer": find_column(columns, ["customer_unique_id", "customer_id", "member_id",
                                          "client_id", "customer", "member", "顧客", "會員", "客戶"]),
        "order": find_column(columns, ["order_id", "transaction_id", "invoice", "order",
                                       "transaction", "訂單", "交易編號"]),
        "date": find_column(columns, ["order_purchase_timestamp", "order_date", "purchase_date",
                                      "transaction_date", "invoicedate", "date", "日期"]),
        "amount": find_column(columns, ["payment_value", "amount", "sales", "revenue",
                                        "total", "金額", "消費", "營收"]),
        "quantity": find_column(columns, ["quantity", "qty", "數量"]),
        "price": find_column(columns, ["unitprice", "unit_price", "price", "單價", "價格"]),
        "status": find_column(columns, ["order_status", "status", "狀態"]),
    }

    def idx(col, options):
        return options.index(col) if col in options else 0

    c1, c2 = st.columns(2)
    customer_col = c1.selectbox("顧客 ID", columns, index=idx(guesses["customer"], columns))
    order_col = c1.selectbox("訂單 ID", columns, index=idx(guesses["order"], columns))
    date_col = c2.selectbox("交易日期", columns, index=idx(guesses["date"], columns))

    # 交易金額：直接使用金額欄位，或由「數量 × 單價」計算（商品明細格式的資料）
    AMOUNT_MODES = ["使用金額欄位", "數量 × 單價"]
    use_qty_price = (guesses["amount"] is None
                     and guesses["quantity"] is not None and guesses["price"] is not None)
    amount_mode = c2.radio("交易金額的來源", AMOUNT_MODES, index=1 if use_qty_price else 0,
                           horizontal=True,
                           help="資料只有數量和單價時選「數量 × 單價」，系統會逐列相乘後再依訂單加總")
    if amount_mode == AMOUNT_MODES[0]:
        amount_col = c2.selectbox("交易金額", columns,
                                  index=idx(guesses["amount"] or guesses["price"], columns))
        amount_cols = [amount_col]
    else:
        qc, pc = c2.columns(2)
        qty_col = qc.selectbox("數量", columns, index=idx(guesses["quantity"], columns))
        price_col = pc.selectbox("單價", columns, index=idx(guesses["price"], columns))
        amount_cols = [qty_col, price_col]

    status_options = [NO_COLUMN] + columns
    status_col = st.selectbox("訂單狀態（選配）", status_options,
                              index=idx(guesses["status"], status_options))
    keep_status = None
    if status_col != NO_COLUMN:
        values = sorted(df[status_col].dropna().astype(str).unique().tolist())
        default = [v for v in values if v.lower() in ("delivered", "completed", "已完成")] or values
        keep_status = st.multiselect("保留哪些狀態的訂單", values, default=default)

    required = [customer_col, order_col, date_col] + amount_cols
    if len(set(required)) != len(required):
        st.error("每個欄位設定都必須使用不同的欄位。")
        st.stop()

    if st.button("開始檢查資料", type="primary", width="stretch"):
        raw = df[[customer_col, order_col, date_col]].copy()
        raw.columns = ["customer_id", "order_id", "order_date"]
        raw["order_date"] = pd.to_datetime(raw["order_date"], errors="coerce")
        if len(amount_cols) == 1:
            raw["amount"] = pd.to_numeric(df[amount_cols[0]], errors="coerce")
        else:
            raw["amount"] = (pd.to_numeric(df[amount_cols[0]], errors="coerce")
                             * pd.to_numeric(df[amount_cols[1]], errors="coerce"))

        # ---------- 第 1 層：欄位完整性 ----------
        # 每一項記錄（層級, 檢查項目, 數量, 單位）。第 1、2 層以「列」計，第 3 層彙總後以「張訂單」計。
        # 第 1 層的四項可能重疊（同一列缺兩個欄位），所以另外記錄實際排除的列數。
        checks = [
            ("1 欄位完整性", "顧客 ID 缺失", int(raw["customer_id"].isna().sum()), "列"),
            ("1 欄位完整性", "訂單 ID 缺失", int(raw["order_id"].isna().sum()), "列"),
            ("1 欄位完整性", "日期無法辨識", int(raw["order_date"].isna().sum()), "列"),
            ("1 欄位完整性", "金額無法辨識", int(raw["amount"].isna().sum()), "列"),
        ]
        valid = raw.dropna().copy()
        checks.append(("1 欄位完整性", "→ 實際排除（扣除重疊）", len(raw) - len(valid), "列"))

        # ---------- 第 2 層：訂單狀態 ----------
        if keep_status is not None:
            status = df.loc[valid.index, status_col].astype(str)
            checks.append(("2 訂單狀態", "排除的訂單狀態", int((~status.isin(keep_status)).sum()), "列"))
            valid = valid[status.isin(keep_status)]

        # ---------- 第 3 層：彙總成訂單層級（同一張訂單多筆付款要加總） ----------
        multi_customer = valid.groupby("order_id")["customer_id"].nunique()
        checks.append(("3 訂單彙總", "訂單對應多位顧客", int((multi_customer > 1).sum()), "張訂單"))
        valid = valid[valid["order_id"].isin(multi_customer[multi_customer == 1].index)]

        orders = (
            valid.groupby("order_id", as_index=False)
            .agg(customer_id=("customer_id", "first"),
                 order_date=("order_date", "min"),
                 amount=("amount", "sum"))
        )
        checks.append(("3 訂單彙總", "訂單金額 ≤ 0（含退貨單）", int((orders["amount"] <= 0).sum()), "張訂單"))
        cancelled = full_cancellations(orders)
        checks.append(("3 訂單彙總", "被整張退貨抵銷的原訂單", len(cancelled), "張訂單"))
        orders = orders[(orders["amount"] > 0) & ~orders["order_id"].isin(cancelled)].reset_index(drop=True)

        # ---------- 第 4 層：CLV 適用性 ----------
        n_orders = len(orders)
        n_customers = orders["customer_id"].nunique()
        days = (orders["order_date"].max() - orders["order_date"].min()).days if n_orders else 0

        st.divider()
        st.subheader("資料品質檢查")
        check_df = pd.DataFrame(checks, columns=["層級", "檢查項目", "數量", "單位"])
        st.dataframe(check_df.style.format({"數量": "{:,}"}), width="stretch", hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("有效訂單數", f"{n_orders:,}")
        c2.metric("有效顧客數", f"{n_customers:,}")
        c3.metric("資料期間", f"{days:,} 天")
        if n_orders:
            st.caption(f"{orders['order_date'].min():%Y-%m-%d} ～ {orders['order_date'].max():%Y-%m-%d}"
                       f"｜原始 {len(df):,} 列 → 有效訂單 {n_orders:,} 張 → 顧客 {n_customers:,} 人")

        if n_orders and n_customers / n_orders > 0.995:
            st.warning("幾乎每張訂單都是不同顧客。可能是「顧客 ID」選錯了"
                       "（例如選成訂單編號），也可能這份資料本來就幾乎沒有人回購。")

        # 回購顧客：在不同天買過 2 次以上（Gamma-Gamma 只能用這些人估計客單價）
        buy_days = (orders.assign(day=orders["order_date"].dt.normalize())
                    .drop_duplicates(["customer_id", "day"])["customer_id"].value_counts())
        n_repeat = int((buy_days >= 2).sum())
        c4, c5 = st.columns(2)
        c4.metric("回購顧客數", f"{n_repeat:,}", help="在不同天購買 2 次以上的顧客")
        c5.metric("回購率", f"{n_repeat / n_customers:.1%}" if n_customers else "—")

        problems = []
        if n_orders < 100:
            problems.append(f"有效訂單 {n_orders:,} 筆，至少需要 100 筆")
        if n_customers < 20:
            problems.append(f"有效顧客 {n_customers:,} 人，至少需要 20 人")
        if days < 180:
            problems.append(f"資料期間 {days:,} 天，至少需要 180 天（要預測未來 3～12 個月）")
        if n_repeat < 20:
            problems.append(f"回購顧客 {n_repeat:,} 人，至少需要 20 人（Gamma-Gamma 需要回購顧客才能估計客單價）")

        if problems:
            st.warning("資料尚未符合 CLV 分析條件：\n\n" + "\n".join(f"- {p}" for p in problems))
        else:
            st.session_state.orders = orders
            reset_results()
            st.session_state.data_ready = True
            st.success("✓ 資料檢查完成，分析功能已解鎖。請點上方「分析結果」。")




# ============================================================
# ② 分析結果
# ============================================================

elif selected_page == "分析結果":
    orders = st.session_state.orders

    if st.session_state.result is None:
        with st.spinner("分析中（資料量大時約需 10～30 秒）…"):
            try:
                rfm = A.calculate_rfm(orders)
                seg, X = A.segment(rfm)
                el = A.elbow(X)
                Z, loadings = A.pca_2d(A.rfm_scaled(rfm))
                clv, info = A.predict_clv(orders, seg)
            except Exception as e:  # noqa: BLE001
                st.error(f"模型無法完成計算：{e}。請確認資料是否有足夠的回購顧客。")
                st.stop()
            st.session_state.result = dict(seg=seg, el=el, Z=Z, loadings=loadings, clv=clv, info=info,
                                           snap=A.snapshot_date(orders))
    r = st.session_state.result
    seg, clv = r["seg"], r["clv"]
    st.caption(f"分析基準日：{r['snap']:%Y-%m-%d}（最後一筆訂單的隔天）｜顧客 {len(seg):,} 人")

    # ---------- 1. 肘部法則 ----------
    st.header("ELBOW METHOD（肘部法則）")
    st.write("Elbow Method（肘部法則）是一種用來找出 K-Means 最佳分群數的方法。透過比較不同群數下的"
             "群內誤差（Inertia），觀察誤差下降的變化情形，並找出下降幅度開始變小的轉折點（肘部），"
             "作為最適合的分群數量。")
    c1, c2 = st.columns([2, 3])
    inert = r["el"].set_index("k")["inertia"]
    if all(k in inert.index for k in (2, 3, 4)) and inert[2] - inert[3] > 2 * (inert[3] - inert[4]):
        c1.write("分群使用 Frequency 與 Monetary 兩個指標。從圖中可以看到，群數由 1 增加到 3 時，"
                 "群內誤差明顯下降；超過 3 群之後，下降幅度明顯變小，因此選擇 3 群。")
    else:
        c1.write("分群使用 Frequency 與 Monetary 兩個指標。這份資料在 3 群附近的轉折不明顯，"
                 "系統仍固定分成 3 群（忠誠、VIP、一般），讓每位顧客只落在一群。")
    c1.write("Recency 沒有放進分群：否則近期購買又消費很高的顧客，會無法判斷該算近期客戶還是 VIP。"
             "顧客會不會流失，改由有使用 Recency 的 BG/NBD 模型預測。")
    c2.pyplot(A.fig_elbow(r["el"]))

    # ---------- 2. PCA ----------
    st.divider()
    st.header("PCA（主成分分析）特徵對比")
    c1, c2 = st.columns([3, 2])
    c1.pyplot(A.fig_pca(r["Z"], seg, r["loadings"]))
    c2.dataframe(A.RULES, hide_index=True, width="stretch")
    c2.write("圖中每個點代表一位顧客，不同顏色代表不同客群。透過 PCA 將 RFM 特徵降維後，"
             "可以觀察各客群在二維空間中的分布情形。結果顯示，大部分顧客集中於左側區域，"
             "表示其消費行為較為相似；而少數顧客分布於右側與上方區域，顯示其消費特徵與一般顧客存在較大差異。")

    # ---------- 3. 人數與百分比 ----------
    st.divider()
    st.header("客戶價值人數與百分比")
    ct = A.count_table(seg)
    st.dataframe(ct.style.format({"人數": "{:,}", "百分比 %": "{:.2f}%"}),
                 hide_index=True, width="stretch")
    p = ct.set_index("客群")["百分比 %"]
    low, high = p["一般客戶"], p["VIP 客戶"] + p["忠誠客戶"]
    if low > high:
        st.write(f"K-Means 結果顯示，一般客戶占總客群約 **{low:.2f}%**，"
                 f"顯示大多數顧客消費次數與金額都不高；VIP 客戶與忠誠客戶合計僅占 **{high:.2f}%**。")
    else:
        st.write(f"K-Means 結果顯示，VIP 客戶與忠誠客戶合計占總客群約 **{high:.2f}%**，"
                 f"一般客戶占 **{low:.2f}%**。")

    # ---------- 4. 未來 CLV ----------
    st.divider()
    st.header("未來 CLV 預測結果")
    st.caption("總 CLV = BG/NBD 預期購買次數 × Gamma-Gamma 預期客單價（單位：原資料的幣別）")
    num_cols = ["人數"] + list(A.HORIZONS)
    st.dataframe(clv.style.format({c: "{:,.0f}" for c in num_cols})
                 .apply(lambda row: ["font-weight: bold"] * len(row) if row["客群"] == "全部顧客"
                        else [""] * len(row), axis=1),
                 hide_index=True, width="stretch")
    t = clv.set_index("客群")
    pl, pv = t.loc["忠誠客戶"], t.loc["VIP 客戶"]
    loyal_pp = pl["12 個月"] / pl["人數"] if pl["人數"] else 0
    vip_pp = pv["12 個月"] / pv["人數"] if pv["人數"] else 0
    sentence = (f"忠誠客戶人數占約 {pl['人數'] / t.loc['全部顧客', '人數']:.1%}，"
                f"12 個月總 CLV 有 {pl['12 個月']:,.0f}；每人約 {loyal_pp:,.2f}，")
    if vip_pp > 0 and loyal_pp >= vip_pp:
        sentence += f"是 VIP（每人約 {vip_pp:,.2f}）的 {loyal_pp / vip_pp:.1f} 倍。"
    else:
        sentence += f"VIP 每人約 {vip_pp:,.2f}。"
    st.write(sentence)
    download_button(clv, "下載 CLV 表（CSV）", "future_clv.csv")

    # ---------- 5. 分客群 ----------
    st.divider()
    st.header("顧客未來 CLV 分客群")
    st.caption("各客群「人數占比」與「未來 12 個月總 CLV 占比」比較")
    sh = A.share_table(clv)
    c1, c2 = st.columns([3, 2])
    c1.pyplot(A.fig_share(sh))
    c2.markdown("\n".join(f"- {A.share_comment(row)}" for _, row in sh.iterrows()))
