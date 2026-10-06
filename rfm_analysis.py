# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 23:20:08 2026

@author: timfa
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def calculate_rfm(df):

    data = df.copy()

    # 確保日期格式
    data["order_date"] = pd.to_datetime(
        data["order_date"],
        errors="coerce"
    )

    # 確保金額為數字
    data["amount"] = pd.to_numeric(
        data["amount"],
        errors="coerce"
    )

    # 移除無效資料
    data = data.dropna(
        subset=[
            "customer_id",
            "order_id",
            "order_date",
            "amount"
        ]
    )

    # --------------------------------------------------
    # 設定分析基準日
    # --------------------------------------------------

    snapshot_date = (
        data["order_date"].max()
        + pd.Timedelta(days=1)
    )

    # --------------------------------------------------
    # RFM
    # --------------------------------------------------

    rfm = data.groupby(
        "customer_id"
    ).agg(

        Recency=(
            "order_date",
            lambda x: (
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
    ).reset_index()

    # --------------------------------------------------
    # AOV
    # --------------------------------------------------

    rfm["AOV"] = (
        rfm["Monetary"]
        / rfm["Frequency"]
    )

    # --------------------------------------------------
    # 平均購買間隔
    # --------------------------------------------------

    customer_gap = (
        data
        .sort_values(
            ["customer_id", "order_date"]
        )
        .groupby("customer_id")[
            "order_date"
        ]
        .apply(
            lambda x:
            x.drop_duplicates()
            .diff()
            .dt.days
            .mean()
        )
        .reset_index(
            name="Avg_Gap"
        )
    )

    rfm = rfm.merge(
        customer_gap,
        on="customer_id",
        how="left"
    )

    # 沒有重複購買的顧客
    # 用 0 表示沒有可計算的購買間隔
    rfm["Avg_Gap"] = rfm["Avg_Gap"].fillna(0)

    return rfm


def run_kmeans(rfm, n_clusters=4):

    data = rfm.copy()

    features = [
        "Recency",
        "Frequency",
        "Monetary",
        "AOV",
        "Avg_Gap"
    ]

    # --------------------------------------------------
    # 避免極端值影響
    # --------------------------------------------------

    model_data = data[features].replace(
        [np.inf, -np.inf],
        np.nan
    )

    model_data = model_data.fillna(0)

    # --------------------------------------------------
    # 標準化
    # --------------------------------------------------

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        model_data
    )

    # --------------------------------------------------
    # K-Means
    # --------------------------------------------------

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    data["Cluster"] = kmeans.fit_predict(
        scaled
    )

    # --------------------------------------------------
    # PCA
    # --------------------------------------------------

    pca = PCA(
        n_components=2
    )

    pca_result = pca.fit_transform(
        scaled
    )

    data["PC1"] = pca_result[:, 0]
    data["PC2"] = pca_result[:, 1]

    explained = (
        pca.explained_variance_ratio_
        * 100
    )

    return data, kmeans, scaler, explained


def assign_cluster_names(rfm):

    result = rfm.copy()

    profile = (
        result
        .groupby("Cluster")
        .agg(
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Avg_AOV=("AOV", "mean"),
            Avg_Gap=("Avg_Gap", "mean"),
            Customer_Count=("customer_id", "count")
        )
        .reset_index()
    )

    # --------------------------------------------------
    # 動態命名
    # --------------------------------------------------

    clusters = list(
        profile["Cluster"]
    )

    names = {}

    # 最近購買
    recent_cluster = (
        profile
        .sort_values("Avg_Recency")
        .iloc[0]["Cluster"]
    )

    names[recent_cluster] = "Recent Customers"

    remaining = [
        x for x in clusters
        if x != recent_cluster
    ]

    # 最高消費
    vip_cluster = (
        profile[
            profile["Cluster"].isin(
                remaining
            )
        ]
        .sort_values(
            "Avg_Monetary",
            ascending=False
        )
        .iloc[0]["Cluster"]
    )

    names[vip_cluster] = "VIP Customers"

    remaining = [
        x for x in remaining
        if x != vip_cluster
    ]

    # 剩餘群組中，Recency 最大者
    if remaining:

        lost_cluster = (
            profile[
                profile["Cluster"].isin(
                    remaining
                )
            ]
            .sort_values(
                "Avg_Recency",
                ascending=False
            )
            .iloc[0]["Cluster"]
        )

        names[lost_cluster] = "Lost Customers"

        remaining = [
            x for x in remaining
            if x != lost_cluster
        ]

    # 最後一群
    for cluster in remaining:
        names[cluster] = "Loyal Customers"

    result["Segment"] = (
        result["Cluster"]
        .map(names)
    )

    profile["Segment"] = (
        profile["Cluster"]
        .map(names)
    )

    return result, profile