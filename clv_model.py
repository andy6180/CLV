import pandas as pd
import numpy as np

from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter


# ============================================================
# 1. 建立 Observation / Future 資料
# ============================================================

def prepare_clv_data(df):

    data = df.copy()

    # --------------------------------------------------------
    # 基本格式
    # --------------------------------------------------------

    data["order_date"] = pd.to_datetime(
        data["order_date"],
        errors="coerce"
    )

    data["amount"] = pd.to_numeric(
        data["amount"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "customer_id",
            "order_id",
            "order_date",
            "amount"
        ]
    )

    # 只保留正數交易
    data = data[
        data["amount"] > 0
    ].copy()

    # 每個顧客、每個訂單只算一次
    data = data.drop_duplicates(
        subset=[
            "customer_id",
            "order_id"
        ]
    )

    # --------------------------------------------------------
    # 資料最後一天
    # --------------------------------------------------------

    data_start = data["order_date"].min()
    data_end = data["order_date"].max()

    # 最後 12 個月作為 Future
    cutoff_date = (
        data_end
        - pd.Timedelta(days=365)
    )

    # --------------------------------------------------------
    # Observation / Future
    # --------------------------------------------------------

    observation_df = data[
        data["order_date"] < cutoff_date
    ].copy()

    future_df = data[
        data["order_date"] >= cutoff_date
    ].copy()

    # 如果資料不足一年
    if len(observation_df) == 0:

        raise ValueError(
            "交易資料不足一年，"
            "無法建立 12 個月 Future CLV 驗證期間。"
        )

    # --------------------------------------------------------
    # Observation RFM
    # --------------------------------------------------------

    snapshot_date = cutoff_date

    customer_rfm = (
        observation_df
        .groupby("customer_id")
        .agg(

            Recency=(
                "order_date",
                lambda x:
                (
                    snapshot_date - x.max()
                ).days
            ),

            Frequency=(
                "order_id",
                "nunique"
            ),

            Monetary=(
                "amount",
                "sum"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # AOV
    # --------------------------------------------------------

    customer_rfm["AOV"] = (
        customer_rfm["Monetary"]
        /
        customer_rfm["Frequency"]
    )

    # --------------------------------------------------------
    # First Purchase
    # --------------------------------------------------------

    first_purchase = (
        observation_df
        .groupby("customer_id")[
            "order_date"
        ]
        .min()
        .reset_index(
            name="First_Purchase"
        )
    )

    # --------------------------------------------------------
    # BG/NBD 的 T
    # --------------------------------------------------------

    bg_data = customer_rfm[
        [
            "customer_id",
            "Frequency",
            "Recency"
        ]
    ].copy()

    bg_data = bg_data.merge(
        first_purchase,
        on="customer_id",
        how="left"
    )

    bg_data["T"] = (
        snapshot_date
        -
        bg_data["First_Purchase"]
    ).dt.days

    # --------------------------------------------------------
    # 清理 BG/NBD 資料
    # --------------------------------------------------------

    bg_data = bg_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    bg_data = bg_data.dropna(
        subset=[
            "Frequency",
            "Recency",
            "T"
        ]
    )

    bg_data = bg_data[
        (bg_data["Frequency"] >= 0)
        &
        (bg_data["Recency"] >= 0)
        &
        (bg_data["T"] > 0)
        &
        (bg_data["Recency"] <= bg_data["T"])
    ].copy()

    # --------------------------------------------------------
    # 把 T 放回 customer_rfm
    # --------------------------------------------------------

    customer_rfm = customer_rfm.merge(
        bg_data[
            [
                "customer_id",
                "T"
            ]
        ],
        on="customer_id",
        how="inner"
    )

    # --------------------------------------------------------
    # Future 實際資料
    # --------------------------------------------------------

    future_revenue = (
        future_df
        .groupby("customer_id")
        .agg(
            Actual_Future_Purchases=(
                "order_id",
                "nunique"
            ),

            Actual_Future_Revenue=(
                "amount",
                "sum"
            )
        )
        .reset_index()
    )

    customer_rfm = customer_rfm.merge(
        future_revenue,
        on="customer_id",
        how="left"
    )

    customer_rfm[
        "Actual_Future_Purchases"
    ] = (
        customer_rfm[
            "Actual_Future_Purchases"
        ]
        .fillna(0)
    )

    customer_rfm[
        "Actual_Future_Revenue"
    ] = (
        customer_rfm[
            "Actual_Future_Revenue"
        ]
        .fillna(0)
    )

    return customer_rfm


# ============================================================
# 2. BG/NBD + Gamma-Gamma
# ============================================================

def fit_clv_models(customer_data):

    model_data = customer_data.copy()

    # --------------------------------------------------------
    # BG/NBD
    # --------------------------------------------------------

    bg_data = model_data[
        [
            "customer_id",
            "Frequency",
            "Recency",
            "T"
        ]
    ].copy()

    bg_data = bg_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    bg_data = bg_data.dropna(
        subset=[
            "Frequency",
            "Recency",
            "T"
        ]
    )

    bg_data = bg_data[
        (bg_data["Frequency"] >= 0)
        &
        (bg_data["Recency"] >= 0)
        &
        (bg_data["T"] > 0)
        &
        (bg_data["Recency"] <= bg_data["T"])
    ].copy()

    if len(bg_data) < 10:

        raise RuntimeError(
            "符合 BG/NBD 條件的顧客數量不足。"
        )

    # --------------------------------------------------------
    # 使用原本成功版本的參數
    # --------------------------------------------------------

    penalizer_values = [
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0
    ]

    bgf = None
    last_error = None

    for penalty in penalizer_values:

        try:

            print(
                f"嘗試 BG/NBD penalizer = {penalty}"
            )

            model = BetaGeoFitter(
                penalizer_coef=penalty
            )

            model.fit(
                frequency=bg_data["Frequency"],
                recency=bg_data["Recency"],
                T=bg_data["T"]
            )

            bgf = model

            print(
                f"BG/NBD 模型成功收斂！"
                f" penalizer = {penalty}"
            )

            break

        except Exception as e:

            last_error = e

            print(
                f"BG/NBD penalizer = "
                f"{penalty} 無法收斂"
            )

    if bgf is None:

        raise RuntimeError(
            "BG/NBD 模型仍無法收斂。"
            f"最後錯誤：{last_error}"
        )

    # --------------------------------------------------------
    # 預測 3 / 6 / 9 / 12 個月
    # --------------------------------------------------------

    result = model_data.copy()

    for months, days in [
        (3, 90),
        (6, 180),
        (9, 270),
        (12, 365)
    ]:

        col = (
            f"Expected_Purchases_{months}M"
        )

        result[col] = (
            bgf
            .conditional_expected_number_of_purchases_up_to_time(
                days,

                result["Frequency"],

                result["Recency"],

                result["T"]
            )
        )

    # --------------------------------------------------------
    # 清理預測結果
    # --------------------------------------------------------

    purchase_columns = [
        "Expected_Purchases_3M",
        "Expected_Purchases_6M",
        "Expected_Purchases_9M",
        "Expected_Purchases_12M"
    ]

    for col in purchase_columns:

        result[col] = (
            result[col]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
            .clip(lower=0)
        )

    # ========================================================
    # Gamma-Gamma
    # ========================================================

    gg_data = result[
        result["Frequency"] > 1
    ].copy()

    gg_data["frequency_gg"] = (
        gg_data["Frequency"] - 1
    )

    gg_data["monetary_value_gg"] = (
        gg_data["AOV"]
    )

    gg_data = gg_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    gg_data = gg_data.dropna(
        subset=[
            "frequency_gg",
            "monetary_value_gg"
        ]
    )

    gg_data = gg_data[
        (gg_data["frequency_gg"] > 0)
        &
        (gg_data["monetary_value_gg"] > 0)
    ].copy()

    ggf = None
    gamma_success = False

    # --------------------------------------------------------
    # Gamma-Gamma 0.1
    # --------------------------------------------------------

    try:

        ggf = GammaGammaFitter(
            penalizer_coef=0.1
        )

        ggf.fit(
            frequency=(
                gg_data["frequency_gg"]
            ),

            monetary_value=(
                gg_data["monetary_value_gg"]
            )
        )

        gamma_success = True

        print(
            "Gamma-Gamma 模型成功收斂！"
        )

    except Exception:

        print(
            "Gamma-Gamma 使用 0.1 未收斂，"
            "改用 0.5"
        )

        # ----------------------------------------------------
        # Gamma-Gamma 0.5
        # ----------------------------------------------------

        try:

            ggf = GammaGammaFitter(
                penalizer_coef=0.5
            )

            ggf.fit(
                frequency=(
                    gg_data["frequency_gg"]
                ),

                monetary_value=(
                    gg_data["monetary_value_gg"]
                )
            )

            gamma_success = True

            print(
                "Gamma-Gamma 模型成功收斂！"
            )

        except Exception:

            print(
                "Gamma-Gamma 無法收斂，"
                "改用中位數 AOV。"
            )

    # --------------------------------------------------------
    # Expected AOV
    # --------------------------------------------------------

    result["Expected_AOV"] = np.nan

    if gamma_success:

        repeat_mask = (
            result["Frequency"] > 1
        )

        result.loc[
            repeat_mask,
            "Expected_AOV"
        ] = (
            ggf
            .conditional_expected_average_profit(

                result.loc[
                    repeat_mask,
                    "Frequency"
                ] - 1,

                result.loc[
                    repeat_mask,
                    "AOV"
                ]
            )
        )

    # --------------------------------------------------------
    # Fallback AOV
    # --------------------------------------------------------

    fallback_aov = (
        gg_data[
            "monetary_value_gg"
        ].median()
    )

    # 如果沒有有效 Gamma-Gamma 資料
    if pd.isna(fallback_aov):

        fallback_aov = (
            result["AOV"].median()
        )

    result["Expected_AOV"] = (
        result["Expected_AOV"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(fallback_aov)
    )

    # --------------------------------------------------------
    # Future CLV
    # --------------------------------------------------------

    result["Future_CLV_3M"] = (
        result["Expected_Purchases_3M"]
        *
        result["Expected_AOV"]
    )

    result["Future_CLV_6M"] = (
        result["Expected_Purchases_6M"]
        *
        result["Expected_AOV"]
    )

    result["Future_CLV_9M"] = (
        result["Expected_Purchases_9M"]
        *
        result["Expected_AOV"]
    )

    result["Future_CLV_12M"] = (
        result["Expected_Purchases_12M"]
        *
        result["Expected_AOV"]
    )

    # --------------------------------------------------------
    # 清理 CLV
    # --------------------------------------------------------

    clv_columns = [
        "Future_CLV_3M",
        "Future_CLV_6M",
        "Future_CLV_9M",
        "Future_CLV_12M"
    ]

    for col in clv_columns:

        result[col] = (
            result[col]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
            .clip(lower=0)
        )

    return bgf, ggf, result


# ============================================================
# 3. CLV 群組統計
# ============================================================

def summarize_clv(result):

    summary = (
        result
        .groupby("Segment")
        .agg(

            Customer_Count=(
                "customer_id",
                "count"
            ),

            Avg_CLV_3M=(
                "Future_CLV_3M",
                "mean"
            ),

            Avg_CLV_6M=(
                "Future_CLV_6M",
                "mean"
            ),

            Avg_CLV_9M=(
                "Future_CLV_9M",
                "mean"
            ),

            Avg_CLV_12M=(
                "Future_CLV_12M",
                "mean"
            ),

            Total_CLV_12M=(
                "Future_CLV_12M",
                "sum"
            ),

            Actual_Future_Revenue=(
                "Actual_Future_Revenue",
                "sum"
            )
        )
        .reset_index()
    )

    return summary

# ============================================================
# 4. Future CLV 預測
# ============================================================

def predict_future_clv(bgf, ggf, customer_data):

    result = customer_data.copy()

    # ========================================================
    # BG/NBD：預測未來購買次數
    # ========================================================

    for months, days in [
        (3, 90),
        (6, 180),
        (9, 270),
        (12, 365)
    ]:

        col = f"Expected_Purchases_{months}M"

        result[col] = (
            bgf
            .conditional_expected_number_of_purchases_up_to_time(
                days,
                result["Frequency"],
                result["Recency"],
                result["T"]
            )
        )

    # 清理預測結果
    purchase_columns = [
        "Expected_Purchases_3M",
        "Expected_Purchases_6M",
        "Expected_Purchases_9M",
        "Expected_Purchases_12M"
    ]

    for col in purchase_columns:

        result[col] = (
            result[col]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
            .clip(lower=0)
        )

    # ========================================================
    # Gamma-Gamma：預測平均客單價
    # ========================================================

    result["Expected_AOV"] = np.nan

    if ggf is not None:

        repeat_mask = (
            result["Frequency"] > 1
        )

        result.loc[
            repeat_mask,
            "Expected_AOV"
        ] = (
            ggf
            .conditional_expected_average_profit(
                result.loc[
                    repeat_mask,
                    "Frequency"
                ] - 1,

                result.loc[
                    repeat_mask,
                    "AOV"
                ]
            )
        )

    # ========================================================
    # AOV fallback
    # ========================================================

    fallback_aov = (
        result["AOV"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .median()
    )

    result["Expected_AOV"] = (
        result["Expected_AOV"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(fallback_aov)
    )

    # ========================================================
    # Future CLV
    # ========================================================

    result["Future_CLV_3M"] = (
        result["Expected_Purchases_3M"]
        *
        result["Expected_AOV"]
    )

    result["Future_CLV_6M"] = (
        result["Expected_Purchases_6M"]
        *
        result["Expected_AOV"]
    )

    result["Future_CLV_9M"] = (
        result["Expected_Purchases_9M"]
        *
        result["Expected_AOV"]
    )

    result["Future_CLV_12M"] = (
        result["Expected_Purchases_12M"]
        *
        result["Expected_AOV"]
    )

    return result