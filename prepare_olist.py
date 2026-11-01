# -*- coding: utf-8 -*-
"""
prepare_olist.py — 把 Olist 原始三張表整理成系統可上傳的交易檔

用法（在 Anaconda Prompt）：
    python prepare_olist.py 資料夾路徑
資料夾內需有：
    olist_orders_dataset.csv
    olist_order_payments_dataset.csv
    olist_customers_dataset.csv
輸出：olist_transactions.csv（每張訂單一列）

正確順序：
1. 先把 payments 依 order_id 加總（一張訂單可能有多筆付款，例如信用卡 + 優惠券）
2. 只保留 delivered 訂單
3. 合併 customers，取得 customer_unique_id
   （customer_id 是每張訂單一個，用它的話每位顧客都會變成只買一次）
4. 再合併付款金額
"""

import sys
from pathlib import Path

import pandas as pd


def main(folder):
    folder = Path(folder)
    orders = pd.read_csv(folder / "olist_orders_dataset.csv")
    payments = pd.read_csv(folder / "olist_order_payments_dataset.csv")
    customers = pd.read_csv(folder / "olist_customers_dataset.csv")

    print(f"原始訂單數：{len(orders):,}")

    payment_by_order = payments.groupby("order_id", as_index=False)["payment_value"].sum()

    delivered = orders[orders["order_status"] == "delivered"]
    print(f"delivered 訂單數：{len(delivered):,}")

    merged = (
        delivered[["order_id", "customer_id", "order_purchase_timestamp", "order_status"]]
        .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left")
        .merge(payment_by_order, on="order_id", how="inner")
    )
    merged = merged[merged["payment_value"] > 0]
    print(f"有付款且金額 > 0 的訂單數：{len(merged):,}")
    print(f"不重複顧客數（customer_unique_id）：{merged['customer_unique_id'].nunique():,}")

    out = merged[["customer_unique_id", "order_id", "order_purchase_timestamp",
                  "payment_value", "order_status"]]
    out.to_csv(folder / "olist_transactions.csv", index=False, encoding="utf-8-sig")
    print(f"已輸出：{folder / 'olist_transactions.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
