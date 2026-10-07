# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:07:58 2026

@author: timfa
"""
# -*- coding: utf-8 -*-
hello
import streamlit as st
import pandas as pd

from rfm_analysis import (
    calculate_rfm,
    run_kmeans,
    assign_cluster_names
)

from clv_model import (
    prepare_clv_data,
    fit_clv_models,
    predict_future_clv,
    summarize_clv
)


# ============================================================
# 網頁設定
# ============================================================

st.set_page_config(
    page_title="CLV 顧客價值分析系統",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# Session State
# ============================================================

if "data_ready" not in st.session_state:
    st.session_state.data_ready = False

if "valid_df" not in st.session_state:
    st.session_state.valid_df = None

if "rfm" not in st.session_state:
    st.session_state.rfm = None

if "clustered" not in st.session_state:
    st.session_state.clustered = None

if "profile" not in st.session_state:
    st.session_state.profile = None

if "explained" not in st.session_state:
    st.session_state.explained = None

if "clv_result" not in st.session_state:
    st.session_state.clv_result = None

if "clv_summary" not in st.session_state:
    st.session_state.clv_summary = None


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #666666;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .locked {
        color: #999999;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 標題
# ============================================================

st.markdown(
    '<div class="main-title">CLV 顧客價值分析系統</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    '上傳交易資料後，系統將依序進行顧客分群、CLV 預測與商家決策分析。'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 上方導覽列
# ============================================================

if st.session_state.data_ready:

    menu_options = [
        "資料上傳",
        "顧客分析",
        "CLV 預測",
        "商家決策"
    ]

else:

    menu_options = [
        "資料上傳",
        "顧客分析 🔒",
        "CLV 預測 🔒",
        "商家決策 🔒"
    ]


selected_page = st.radio(
    "功能選單",
    menu_options,
    horizontal=True,
    label_visibility="collapsed"
)


# ============================================================
# 防止未上傳資料直接進入其他頁面
# ============================================================

if not st.session_state.data_ready:

    if selected_page != "資料上傳":

        st.warning(
            "請先完成資料上傳與資料檢查，"
            "才能使用顧客分析、CLV 預測與商家決策功能。"
        )

        st.stop()


# ============================================================
# 工具：自動尋找欄位
# ============================================================

def find_column(columns, keywords):

    for col in columns:

        col_lower = str(col).lower()

        for keyword in keywords:

            if keyword.lower() in col_lower:

                return col

    return None


# ============================================================
# ============================================================
# ① 資料上傳
# ============================================================
# ============================================================

if selected_page == "資料上傳":

    st.header("資料上傳")

    st.write(
        "請上傳商家的交易資料 CSV 或 XLSX。"
        "完整 CLV 分析需要顧客 ID、訂單 ID、交易日期與交易金額。"
    )

    uploaded_file = st.file_uploader(
        "上傳 CSV 檔案",
        type=["csv", "xlsx"]
    )


    # --------------------------------------------------------
    # 尚未上傳
    # --------------------------------------------------------

    if uploaded_file is None:

        st.info(
            "請先上傳 CSV 或 XLSX，完成資料檢查後即可開始分析。"
        )

        st.subheader("最低需要的資料")

        st.code(
            "顧客 ID\n"
            "訂單 ID\n"
            "交易日期\n"
            "交易金額"
        )

        st.stop()


    # --------------------------------------------------------
    # 讀取 CSV
    # --------------------------------------------------------

    try:

        if uploaded_file.name.lower().endswith(".csv"):
        
            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.lower().endswith(".xlsx"):

            df = pd.read_excel(uploaded_file)

        else:

            st.error("不支援的檔案格式。")

            st.stop()

    except Exception as e:

        st.error(
            f"檔案讀取失敗：{e}"
        )

        st.stop()


    # --------------------------------------------------------
    # 資料基本資訊
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "資料筆數",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "欄位數",
            f"{len(df.columns):,}"
        )

    with col3:

        # 先用目前選定的顧客欄位計算真正的顧客數
        # 尚未完成欄位設定前，顯示資料中的最大唯一值僅作為預覽
        st.metric(
            "顧客數（預覽）",
            f"{df.nunique().max():,}"
        )


    st.subheader("資料預覽")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    st.divider()


    # --------------------------------------------------------
    # 欄位設定
    # --------------------------------------------------------

    st.subheader("資料欄位設定")

    st.write(
        "系統會自動推薦欄位，如果判斷錯誤，可以手動修改。"
    )

    columns = list(df.columns)


    auto_customer = find_column(
        columns,
        [
            "customer_id",
            "customer",
            "member_id",
            "member",
            "client_id",
            "顧客",
            "會員",
            "客戶"
        ]
    )


    auto_order = find_column(
        columns,
        [
            "order_id",
            "order",
            "transaction_id",
            "transaction",
            "訂單",
            "交易編號"
        ]
    )


    auto_date = find_column(
        columns,
        [
            "order_date",
            "purchase_date",
            "transaction_date",
            "date",
            "datetime",
            "日期",
            "購買日期",
            "交易日期"
        ]
    )


    auto_amount = find_column(
        columns,
        [
            "amount",
            "sales",
            "revenue",
            "price",
            "payment_value",
            "total",
            "金額",
            "消費",
            "營收",
            "價格"
        ]
    )


    # --------------------------------------------------------
    # 找不到時先使用第一欄
    # --------------------------------------------------------

    if auto_customer is None:
        auto_customer = columns[0]

    if auto_order is None:
        auto_order = columns[0]

    if auto_date is None:
        auto_date = columns[0]

    if auto_amount is None:
        auto_amount = columns[0]


    col1, col2 = st.columns(2)


    with col1:

        customer_col = st.selectbox(
            "顧客 ID",
            columns,
            index=columns.index(auto_customer)
        )

        order_col = st.selectbox(
            "訂單 ID",
            columns,
            index=columns.index(auto_order)
        )


    with col2:

        date_col = st.selectbox(
            "交易日期",
            columns,
            index=columns.index(auto_date)
        )

        amount_col = st.selectbox(
            "交易金額",
            columns,
            index=columns.index(auto_amount)
        )


    # --------------------------------------------------------
    # 欄位重複檢查
    # --------------------------------------------------------

    selected_columns = [
        customer_col,
        order_col,
        date_col,
        amount_col
    ]


    if len(selected_columns) != len(set(selected_columns)):

        st.error(
            "欄位設定錯誤：顧客 ID、訂單 ID、"
            "交易日期與交易金額不能使用同一個欄位。"
        )

        st.stop()


    # --------------------------------------------------------
    # 開始資料檢查
    # --------------------------------------------------------

    if st.button(
        "開始檢查資料",
        type="primary",
        use_container_width=True
    ):

        analysis_df = df[
            [
                customer_col,
                order_col,
                date_col,
                amount_col
            ]
        ].copy()


        analysis_df.columns = [
            "customer_id",
            "order_id",
            "order_date",
            "amount"
        ]


        # ----------------------------------------------------
        # 格式轉換
        # ----------------------------------------------------

        analysis_df["order_date"] = pd.to_datetime(
            analysis_df["order_date"],
            errors="coerce"
        )


        analysis_df["amount"] = pd.to_numeric(
            analysis_df["amount"],
            errors="coerce"
        )


        # ----------------------------------------------------
        # 資料品質檢查
        # ----------------------------------------------------

        missing_customer = (
            analysis_df["customer_id"].isna().sum()
        )

        missing_order = (
            analysis_df["order_id"].isna().sum()
        )

        invalid_date = (
            analysis_df["order_date"].isna().sum()
        )

        invalid_amount = (
            analysis_df["amount"].isna().sum()
        )

        negative_amount = (
            analysis_df["amount"] < 0
        ).sum()


        # ----------------------------------------------------
        # 有效資料
        # ----------------------------------------------------
        # 完整 RFM + K-Means + BG/NBD + Gamma-Gamma CLV
        # 需要四個欄位都有效。
        # 不自行補造 Order ID，避免把商品明細誤當成不同訂單。

        valid_df = analysis_df.dropna(
            subset=[
                "customer_id",
                "order_id",
                "order_date",
                "amount"
            ]
        ).copy()

        # 只保留正的交易金額
        valid_df = valid_df[
            valid_df["amount"] > 0
        ].copy()

        # 只使用完整有效交易資料進行後續分析
        valid_transaction_count = len(valid_df)

        valid_customer_count = (
            valid_df["customer_id"].nunique()
        )

        valid_order_count = (
            valid_df["order_id"].nunique()
        )


        # ----------------------------------------------------
        # 日期範圍
        # ----------------------------------------------------

        if len(valid_df) > 0:

            min_date = valid_df["order_date"].min()

            max_date = valid_df["order_date"].max()

            data_days = (
                max_date - min_date
            ).days

        else:

            min_date = None
            max_date = None
            data_days = 0


        # ----------------------------------------------------
        # 顯示檢查結果
        # ----------------------------------------------------

        st.divider()

        st.subheader("資料品質檢查")


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            if missing_customer == 0:

                st.success("✓ 顧客 ID")

            else:

                st.error(
                    f"✗ 顧客 ID\n"
                    f"缺失 {missing_customer:,} 筆"
                )


        with c2:

            if missing_order == 0:

                st.success("✓ 訂單 ID")

            else:

                st.error(
                    f"✗ 訂單 ID\n"
                    f"缺失 {missing_order:,} 筆"
                )


        with c3:

            if invalid_date == 0:

                st.success("✓ 交易日期")

            else:

                st.error(
                    f"✗ 交易日期\n"
                    f"無效 {invalid_date:,} 筆"
                )


        with c4:

            if invalid_amount == 0:

                st.success("✓ 交易金額")

            else:

                st.error(
                    f"✗ 交易金額\n"
                    f"無效 {invalid_amount:,} 筆"
                )


        # ----------------------------------------------------
        # 統計
        # ----------------------------------------------------

        st.subheader("有效資料統計")


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "有效交易筆數",
                f"{len(valid_df):,}"
            )


        with c2:

            st.metric(
                "有效顧客數",
                f"{valid_customer_count:,}"
            )


        with c3:

            st.metric(
                "有效訂單數",
                f"{valid_order_count:,}"
            )

        valid_rate = (
            len(valid_df) / len(analysis_df) * 100
            if len(analysis_df) > 0 else 0
        )

        st.caption(
            f"完整有效交易資料：{len(valid_df):,} / {len(analysis_df):,} 筆 "
            f"（{valid_rate:.1f}%）"
        )

        st.info(
            "後續 RFM、K-Means 與 CLV 模型只會使用上述完整有效交易資料。"
        )


        # ----------------------------------------------------
        # 日期
        # ----------------------------------------------------

        if min_date is not None:

            st.subheader("交易資料期間")

            st.write(
                f"最早交易日期："
                f"**{min_date.strftime('%Y-%m-%d')}**"
            )

            st.write(
                f"最新交易日期："
                f"**{max_date.strftime('%Y-%m-%d')}**"
            )

            st.write(
                f"資料期間：約 **{data_days} 天**"
            )


        # ----------------------------------------------------
        # CLV 適用性
        # ----------------------------------------------------

        st.divider()

        st.subheader("CLV 分析適用性")


        # ----------------------------------------------------
        # CLV 是否可以建立
        #
        # 後續模型只使用「有效交易資料」。
        # 因此不再要求原始資料的缺失筆數必須為 0，
        # 而是判斷清理後是否有足夠的有效交易、有效顧客
        # 以及至少 365 天的有效交易期間。
        # ----------------------------------------------------
        basic_ok = (
            valid_transaction_count >= 100
            and valid_customer_count >= 20
            and data_days >= 365
        )


        if basic_ok:

            st.success(
                "✓ 資料符合目前 CLV 分析的基本條件。"
            )

            st.info(
                "資料檢查完成，現在可以開始使用其他分析功能。"
            )


            # ------------------------------------------------
            # 儲存資料
            # ------------------------------------------------

            st.session_state.valid_df = valid_df

            # 新資料上傳後，清除上一份資料的分析結果
            st.session_state.rfm = None
            st.session_state.clustered = None
            st.session_state.profile = None
            st.session_state.explained = None
            st.session_state.clv_result = None
            st.session_state.clv_summary = None

            st.session_state.data_ready = True


            st.success(
                "✓ 分析功能已解鎖！"
            )

            st.rerun()


        else:

            st.warning(
                "目前資料尚未完全符合完整 CLV 分析條件。"
            )

            st.info(
                "系統不會自行補造訂單 ID 或交易日期，"
                "避免改變原始交易資料的真實購買行為。"
            )


            if valid_transaction_count < 100:

                st.write(
                    f"• 有效交易筆數目前為 {valid_transaction_count:,} 筆，至少需要 100 筆"
                )


            if valid_customer_count < 20:

                st.write(
                    f"• 有效顧客數目前為 {valid_customer_count:,} 人，至少需要 20 人"
                )


            if data_days < 365:

                st.write(
                    "• 完整 CLV 分析需要有效交易資料至少涵蓋 365 天"
                )


            if negative_amount > 0:

                st.write(
                    f"• 發現 {negative_amount:,} 筆負數金額"
                )


# ============================================================
# ============================================================
# ② 顧客分析
# ============================================================
# ============================================================

elif selected_page == "顧客分析":

    st.header("顧客分析")

    valid_df = st.session_state.valid_df


    if valid_df is None:

        st.error(
            "尚未取得有效資料。"
        )

        st.stop()


    # --------------------------------------------------------
    # RFM
    # --------------------------------------------------------

    st.subheader("RFM 顧客分析")

    st.write(
        "RFM 分別觀察顧客最近一次購買時間、"
        "購買頻率與消費金額。"
    )


    if st.session_state.rfm is None:

        with st.spinner("正在計算 RFM..."):

            rfm = calculate_rfm(
                valid_df
            )

            st.session_state.rfm = rfm

    else:

        rfm = st.session_state.rfm


    st.success("RFM 計算完成！")


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "平均 Recency",
            f"{rfm['Recency'].mean():.1f} 天"
        )


    with c2:

        st.metric(
            "平均 Frequency",
            f"{rfm['Frequency'].mean():.1f} 次"
        )


    with c3:

        st.metric(
            "平均 Monetary",
            f"{rfm['Monetary'].mean():,.2f}"
        )


    st.dataframe(
        rfm,
        use_container_width=True
    )


    # --------------------------------------------------------
    # K-Means
    # --------------------------------------------------------

    st.divider()

    st.subheader("K-Means 顧客分群")


    cluster_count = st.slider(
        "選擇分群數 K",
        min_value=2,
        max_value=6,
        value=4
    )


    if st.button(
        "重新執行顧客分群",
        type="primary"
    ):

        with st.spinner(
            "正在進行 K-Means 顧客分群..."
        ):

            clustered, kmeans, scaler, explained = run_kmeans(
                rfm,
                n_clusters=cluster_count
            )


            clustered, profile = assign_cluster_names(
                clustered
            )


            st.session_state.clustered = clustered

            st.session_state.profile = profile

            st.session_state.explained = explained


    if st.session_state.clustered is None:

        st.info(
            "請選擇 K 值後，按「重新執行顧客分群」。"
        )

        st.stop()


    clustered = st.session_state.clustered

    profile = st.session_state.profile

    explained = st.session_state.explained


    st.success(
        f"K-Means 分群完成，目前 K = {cluster_count}"
    )


    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    st.subheader("PCA 顧客分群視覺化")


    st.write(
        f"PC1 解釋變異：{explained[0]:.2f}%"
    )

    st.write(
        f"PC2 解釋變異：{explained[1]:.2f}%"
    )


    st.scatter_chart(
        clustered,
        x="PC1",
        y="PC2",
        color="Segment"
    )


    # --------------------------------------------------------
    # 顧客群統計
    # --------------------------------------------------------

    st.subheader("顧客分群結果")


    display_profile = profile[
        [
            "Segment",
            "Customer_Count",
            "Avg_Recency",
            "Avg_Frequency",
            "Avg_Monetary",
            "Avg_AOV",
            "Avg_Gap"
        ]
    ].copy()


    display_profile.columns = [
        "顧客群",
        "顧客數",
        "平均 Recency",
        "平均 Frequency",
        "平均 Monetary",
        "平均 AOV",
        "平均購買間隔"
    ]


    st.dataframe(
        display_profile,
        use_container_width=True
    )


    # --------------------------------------------------------
    # 顧客數
    # --------------------------------------------------------

    st.subheader("各顧客群人數")


    segment_count = (
        clustered["Segment"]
        .value_counts()
        .rename_axis("顧客群")
        .reset_index(name="顧客數")
    )


    st.bar_chart(
        segment_count,
        x="顧客群",
        y="顧客數"
    )


    # --------------------------------------------------------
    # 顧客明細
    # --------------------------------------------------------

    st.subheader("顧客分群明細")


    st.dataframe(
        clustered[
            [
                "customer_id",
                "Recency",
                "Frequency",
                "Monetary",
                "AOV",
                "Avg_Gap",
                "Cluster",
                "Segment"
            ]
        ],
        use_container_width=True
    )


# ============================================================
# ============================================================
# ③ CLV 預測
# ============================================================
# ============================================================

elif selected_page == "CLV 預測":

    st.header("CLV 顧客終身價值預測")


    valid_df = st.session_state.valid_df

    clustered = st.session_state.clustered


    if valid_df is None:

        st.error(
            "尚未取得交易資料。"
        )

        st.stop()


    if clustered is None:

        st.warning(
            "請先到「顧客分析」完成 K-Means 顧客分群。"
        )

        st.stop()


    # --------------------------------------------------------
    # 執行 CLV
    # --------------------------------------------------------

    if st.session_state.clv_result is None:

        progress_text = st.empty()

        progress_bar = st.progress(0)


        try:

            progress_text.info(
                "正在準備 CLV 資料..."
            )

            progress_bar.progress(20)


            clv_data = prepare_clv_data(
                valid_df
            )


            # ------------------------------------------------
            # 加入顧客群
            # ------------------------------------------------

            clv_data = clv_data.merge(
                clustered[
                    [
                        "customer_id",
                        "Segment"
                    ]
                ],
                on="customer_id",
                how="left"
            )


            progress_text.info(
                "正在建立 BG/NBD + Gamma-Gamma 模型..."
            )

            progress_bar.progress(50)


            bgf, ggf, clv_data = fit_clv_models(
                clv_data
            )


            progress_text.info(
                "正在預測未來顧客價值..."
            )

            progress_bar.progress(80)


            clv_result = predict_future_clv(
                bgf,
                ggf,
                clv_data
            )


            clv_summary = summarize_clv(
                clv_result
            )


            progress_bar.progress(100)


            progress_text.success(
                "CLV 分析完成！"
            )


            st.session_state.clv_result = clv_result

            st.session_state.clv_summary = clv_summary


        except Exception as e:

            progress_bar.progress(100)

            progress_text.error(
                "CLV 模型建立失敗"
            )

            st.error(
                f"錯誤原因：{e}"
            )

            st.stop()


    else:

        clv_result = st.session_state.clv_result

        clv_summary = st.session_state.clv_summary


    # --------------------------------------------------------
    # 顧客 Future CLV
    # --------------------------------------------------------

    st.subheader("顧客 Future CLV")


    display_clv = clv_result[
        [
            "customer_id",
            "Segment",
            "Frequency",
            "AOV",
            "Expected_AOV",
            "Expected_Purchases_3M",
            "Future_CLV_3M",
            "Future_CLV_6M",
            "Future_CLV_9M",
            "Future_CLV_12M"
        ]
    ].copy()


    display_clv.columns = [
        "顧客 ID",
        "顧客群",
        "重複購買次數",
        "歷史平均客單價",
        "預期平均客單價",
        "預期購買次數(3M)",
        "Future CLV(3M)",
        "Future CLV(6M)",
        "Future CLV(9M)",
        "Future CLV(12M)"
    ]


    st.dataframe(
        display_clv,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CLV 群組統計
    # --------------------------------------------------------

    st.divider()

    st.subheader("各顧客群 Future CLV")


    display_summary = clv_summary[
        [
            "Segment",
            "Customer_Count",
            "Avg_CLV_3M",
            "Avg_CLV_6M",
            "Avg_CLV_9M",
            "Avg_CLV_12M",
            "Total_CLV_12M"
        ]
    ].copy()


    display_summary.columns = [
        "顧客群",
        "顧客數",
        "平均 CLV(3M)",
        "平均 CLV(6M)",
        "平均 CLV(9M)",
        "平均 CLV(12M)",
        "總 CLV(12M)"
    ]


    st.dataframe(
        display_summary,
        use_container_width=True
    )


    # --------------------------------------------------------
    # 12M CLV
    # --------------------------------------------------------

    st.subheader("各顧客群 12 個月預期 CLV")


    st.bar_chart(
        display_summary,
        x="顧客群",
        y="平均 CLV(12M)"
    )


    # --------------------------------------------------------
    # TOP 10
    # --------------------------------------------------------

    st.subheader("高價值顧客 TOP 10")


    top_customers = (
        clv_result
        .sort_values(
            "Future_CLV_12M",
            ascending=False
        )
        .head(10)
    )


    st.dataframe(
        top_customers[
            [
                "customer_id",
                "Segment",
                "Future_CLV_3M",
                "Future_CLV_6M",
                "Future_CLV_9M",
                "Future_CLV_12M"
            ]
        ],
        use_container_width=True
    )


    st.info(
        "Future CLV 為模型預測值，代表預期未來顧客價值，"
        "並不代表實際一定會產生相同營收。"
    )


# ============================================================
# ============================================================
# ④ 商家決策
# ============================================================
# ============================================================

elif selected_page == "商家決策":

    st.header("商家決策")


    clustered = st.session_state.clustered

    clv_summary = st.session_state.clv_summary


    if clustered is None:

        st.warning(
            "請先完成顧客分析。"
        )

        st.stop()


    if clv_summary is None:

        st.warning(
            "請先完成 CLV 預測。"
        )

        st.stop()


    # --------------------------------------------------------
    # 整體顧客狀況
    # --------------------------------------------------------

    st.subheader("目前顧客狀況")


    total_customers = len(clustered)


    segment_count = (
        clustered["Segment"]
        .value_counts()
    )


    largest_segment = (
        segment_count.idxmax()
    )


    largest_count = (
        segment_count.max()
    )


    largest_percent = (
        largest_count
        / total_customers
        * 100
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "顧客總數",
            f"{total_customers:,}"
        )


    with c2:

        st.metric(
            "主要顧客群",
            largest_segment
        )


    with c3:

        st.metric(
            "主要顧客群比例",
            f"{largest_percent:.1f}%"
        )


    # --------------------------------------------------------
    # 各群策略
    # --------------------------------------------------------

    st.divider()

    st.subheader("顧客群經營策略")


    segments = clustered[
        "Segment"
    ].dropna().unique()


    for segment in segments:

        segment_data = clustered[
            clustered["Segment"] == segment
        ]


        customer_num = len(
            segment_data
        )


        percentage = (
            customer_num
            / total_customers
            * 100
        )


        st.markdown(
            f"### {segment}"
        )


        st.write(
            f"目前共有 **{customer_num:,} 位顧客**，"
            f"約占全部顧客 **{percentage:.1f}%**。"
        )


        # ----------------------------------------------------
        # VIP
        # ----------------------------------------------------

        if segment == "VIP Customers":

            st.write(
                "**經營方向：維持高價值顧客。**"
            )

            st.write(
                "• 提供 VIP 會員福利"
            )

            st.write(
                "• 個人化商品推薦"
            )

            st.write(
                "• 優先提供新品或優惠資訊"
            )


        # ----------------------------------------------------
        # Loyal
        # ----------------------------------------------------

        elif segment == "Loyal Customers":

            st.write(
                "**經營方向：提高忠誠度與消費頻率。**"
            )

            st.write(
                "• 推出會員回饋"
            )

            st.write(
                "• 設計累積消費獎勵"
            )

            st.write(
                "• 推薦相關商品，提高客單價"
            )


        # ----------------------------------------------------
        # Recent
        # ----------------------------------------------------

        elif segment == "Recent Customers":

            st.write(
                "**經營方向：促進第二次購買。**"
            )

            st.write(
                "• 發送首次購買後優惠"
            )

            st.write(
                "• 推薦相關商品"
            )

            st.write(
                "• 建立會員關係"
            )


        # ----------------------------------------------------
        # Lost
        # ----------------------------------------------------

        elif segment == "Lost Customers":

            st.write(
                "**經營方向：喚回沉睡顧客。**"
            )

            st.write(
                "• 發送回購優惠"
            )

            st.write(
                "• 提供限時折扣"
            )

            st.write(
                "• 針對過去購買商品進行再次推薦"
            )


        # ----------------------------------------------------
        # 其他群組
        # ----------------------------------------------------

        else:

            st.write(
                "**經營方向：根據顧客行為進行差異化行銷。**"
            )

            st.write(
                "• 觀察購買頻率"
            )

            st.write(
                "• 觀察消費金額"
            )

            st.write(
                "• 針對不同顧客需求提供行銷活動"
            )


    # --------------------------------------------------------
    # CLV 最高顧客群
    # --------------------------------------------------------

    st.divider()

    st.subheader("CLV 重點發現")


    highest_clv_row = clv_summary.loc[
        clv_summary["Avg_CLV_12M"].idxmax()
    ]


    highest_clv_segment = (
        highest_clv_row["Segment"]
    )


    highest_clv_value = (
        highest_clv_row["Avg_CLV_12M"]
    )


    st.success(
        f"目前平均 12 個月 CLV 最高的顧客群為 "
        f"**{highest_clv_segment}**，"
        f"平均預期 CLV 約為 "
        f"**{highest_clv_value:,.2f}**。"
    )


    # --------------------------------------------------------
    # 模型說明
    # --------------------------------------------------------

    st.divider()

    st.subheader("分析結果怎麼看？")


    st.write(
        "**RFM：**"
        "觀察顧客最近多久購買、購買幾次，以及消費多少。"
    )


    st.write(
        "**K-Means：**"
        "將消費行為相似的顧客分成不同群組。"
    )


    st.write(
        "**BG/NBD：**"
        "預測顧客未來可能購買幾次。"
    )


    st.write(
        "**Gamma-Gamma：**"
        "預測顧客未來可能產生的平均客單價。"
    )


    st.write(
        "**Future CLV：**"
        "以預期購買次數 × 預期平均客單價，"
        "估計未來顧客價值。"
    )


    st.info(
        "以上結果是根據上傳的交易資料建立模型後產生，"
        "不同商家的資料會得到不同的顧客分群與 CLV 結果。"
    )
