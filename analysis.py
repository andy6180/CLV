# -*- coding: utf-8 -*-
"""
analysis.py — 網站的五個分析結果

  1. 肘部法則（Elbow Method）
  2. PCA 特徵對比
  3. 客戶價值人數與百分比
  4. 未來 CLV 預測結果（BG/NBD × Gamma-Gamma，3／6／9／12 個月總額）
  5. 顧客未來 CLV 分客群（人數占比 vs 12 個月總 CLV 占比）

輸入是 app.py 整理好的訂單表（每列一張訂單）：customer_id, order_id, order_date, amount
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Windows 用微軟正黑體，Mac 用蘋方，Linux 用 Noto；找不到的字型 matplotlib 會自動跳過
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Heiti TC",
                                   "Noto Sans CJK TC", "Noto Sans CJK JP", "Arial Unicode MS",
                                   "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

SEGMENTS = ["忠誠客戶", "VIP 客戶", "一般客戶"]                   # 表格順序
CLV_ORDER = SEGMENTS
COLORS = {"一般客戶": "gray", "忠誠客戶": "blue", "VIP 客戶": "red"}
RULES = pd.DataFrame({"客群": ["忠誠客戶", "VIP 客戶", "一般客戶"],
                      "判定依據": ["Frequency 最高", "Monetary 最高", "Frequency、Monetary 皆低"]})
HORIZONS = {"3 個月": 90, "6 個月": 180, "9 個月": 270, "12 個月": 360}
BEST_K = 3


def snapshot_date(orders):
    """分析基準日 = 最後一筆訂單的隔天（Olist 為 2018-08-30，與上學期相同）。"""
    return orders["order_date"].max().normalize() + pd.Timedelta(days=1)


# ============================================================
# RFM 與 K-Means 分群
# ============================================================

def calculate_rfm(orders):
    snap = snapshot_date(orders)
    g = orders.groupby("customer_id")
    rfm = pd.DataFrame({
        "R": (snap - g["order_date"].max()).dt.days,
        "F": g["order_id"].nunique(),
        "M": g["amount"].sum(),
    })
    return rfm


def segment(rfm):
    """只用 F、M 標準化 → K-Means(3)，再依平均值命名三群。

    R（多久沒買）不放進分群：最近買很多的人一律依消費金額歸類，不會出現
    「算近期還是 VIP」的重疊。顧客會不會流失，交給有用到 R 的 BG/NBD 預測。

    命名有先後順序，每位顧客只會落在一群：
      1. 購買次數（F）平均最高的群 → 忠誠客戶
      2. 其餘兩群中，消費金額（M）平均較高 → VIP 客戶
      3. 剩下的 → 一般客戶
    回傳的 X 是 F、M 標準化後的矩陣（肘部法則用）。
    """
    X = StandardScaler().fit_transform(rfm[["F", "M"]].values)
    labels = KMeans(BEST_K, random_state=42, n_init=10).fit_predict(X)
    prof = rfm.assign(k=labels).groupby("k")[["F", "M"]].mean()
    loyal = prof["F"].idxmax()
    rest = prof.drop(loyal)
    vip = rest["M"].idxmax()
    general = rest.drop(vip).index[0]
    names = {loyal: "忠誠客戶", vip: "VIP 客戶", general: "一般客戶"}
    out = rfm.copy()
    out["客群"] = pd.Series(labels, index=rfm.index).map(names)
    return out, X


def rfm_scaled(rfm):
    """R、F、M 標準化（PCA 圖用，呈現三個指標的整體分布）。"""
    return StandardScaler().fit_transform(rfm[["R", "F", "M"]].values)


def elbow(X, k_max=10):
    ks = list(range(1, min(k_max, len(np.unique(X, axis=0))) + 1))
    inertia = [KMeans(k, random_state=42, n_init=10).fit(X).inertia_ for k in ks]
    return pd.DataFrame({"k": ks, "inertia": inertia})


def pca_2d(X):
    """PCA 降到 2 維。主成分方向固定成「負荷量總和為正」，每次畫出來方向一致。"""
    pca = PCA(2).fit(X)
    comps = pca.components_.copy()
    for i in range(2):
        if comps[i].sum() < 0:
            comps[i] *= -1
    Z = (X - pca.mean_) @ comps.T
    loadings = pd.DataFrame(comps.T, index=["R", "F", "M"], columns=["PC1", "PC2"])
    return Z, loadings


def axis_label(loadings, pc):
    """依負荷量自動寫軸名稱，例如「PC1（主要反映 F、M）」。"""
    col = loadings[pc].abs()
    main = [f for f in ["R", "F", "M"] if col[f] >= 0.4]
    return f"{pc}（主要反映 {'、'.join(main)}）"


def count_table(seg):
    n = seg["客群"].value_counts().reindex(SEGMENTS).fillna(0).astype(int)
    return pd.DataFrame({"客群": SEGMENTS, "人數": n.values,
                         "百分比 %": (n.values / n.sum() * 100)})


# ============================================================
# BG/NBD × Gamma-Gamma
# ============================================================

def _fit(fitter_cls, *args):
    """先用 penalizer 0（與簡報相同）；無法收斂時才加極小的 penalizer。"""
    for pen in (0.0, 0.001, 0.01):
        try:
            m = fitter_cls(penalizer_coef=pen)
            m.fit(*args)
            if np.all(np.isfinite(m.params_.values)):
                return m, pen
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"{fitter_cls.__name__} 無法收斂")


def predict_clv(orders, seg):
    snap = snapshot_date(orders)
    s = summary_data_from_transaction_data(orders, "customer_id", "order_date", "amount",
                                           freq="D", observation_period_end=snap)
    bgf, pen_bg = _fit(BetaGeoFitter, s["frequency"], s["recency"], s["T"])
    rep = s[(s["frequency"] > 0) & (s["monetary_value"] > 0)]
    ggf, pen_gg = _fit(GammaGammaFitter, rep["frequency"], rep["monetary_value"])
    aov = ggf.conditional_expected_average_profit(s["frequency"], s["monetary_value"])
    for name, days in HORIZONS.items():
        s[name] = bgf.conditional_expected_number_of_purchases_up_to_time(
            days, s["frequency"], s["recency"], s["T"]) * aov
    s = s.join(seg["客群"])

    tot = s.groupby("客群")[list(HORIZONS)].sum().reindex(CLV_ORDER).fillna(0)
    n = s.groupby("客群").size().reindex(CLV_ORDER).fillna(0).astype(int)
    tot.loc["全部顧客"] = s[list(HORIZONS)].sum()
    n.loc["全部顧客"] = len(s)
    table = tot.round(0)
    table.insert(0, "人數", n)
    info = {"bg_params": bgf.params_.to_dict(), "gg_params": ggf.params_.to_dict(),
            "penalizer": (pen_bg, pen_gg), "n_repeat": len(rep)}
    return table.reset_index(names="客群"), info


def share_table(clv_table):
    t = clv_table.set_index("客群")
    seg = t.loc[CLV_ORDER]
    return pd.DataFrame({
        "客群": CLV_ORDER,
        "人數占比": (seg["人數"] / t.loc["全部顧客", "人數"] * 100).values,
        "CLV 占比": (seg["12 個月"] / t.loc["全部顧客", "12 個月"] * 100).values,
    })


# ============================================================
# 圖
# ============================================================

def fig_elbow(el):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    y = el["inertia"] / 1e4
    ax.plot(el["k"], y, marker="o", color="#1f77b4")
    if BEST_K in el["k"].values:
        yk = y[el["k"] == BEST_K].iloc[0]
        ax.plot(BEST_K, yk, "o", ms=11, color="red", zorder=5)
        ax.annotate(f"Best K = {BEST_K}", (BEST_K, yk), xytext=(BEST_K + 1, yk + (y.max() - y.min()) * 0.1),
                    color="red", fontsize=11, arrowprops=dict(arrowstyle="->", color="red"))
    ax.set_title("Elbow Method", fontsize=14)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia（×10⁴）")
    ax.set_xticks(el["k"])
    ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def fig_pca(Z, seg, loadings):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    labels = seg["客群"].values
    for name in ["一般客戶", "忠誠客戶", "VIP 客戶"]:
        m = labels == name
        ax.scatter(Z[m, 0], Z[m, 1], s=5, c=COLORS[name], alpha=.5, linewidths=0, label=name)
    ax.set_title("PCA of RFM Features", fontsize=14)
    ax.set_xlabel(axis_label(loadings, "PC1"))
    ax.set_ylabel(axis_label(loadings, "PC2"))
    ax.legend(title="客群", markerscale=3, loc="upper right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def fig_share(sh):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(sh))
    w = .38
    b1 = ax.bar(x - w / 2, sh["人數占比"], w, color="#B8C4CC", label="人數占比")
    b2 = ax.bar(x + w / 2, sh["CLV 占比"], w, color="#13A3B5", label="12 個月總 CLV 占比")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}%",
                    ha="center", fontsize=11, color="#2C3E50")
    ax.set_xticks(x)
    ax.set_xticklabels(sh["客群"], fontsize=12)
    ax.set_ylabel("占比（%）")
    ax.set_ylim(0, max(sh["人數占比"].max(), sh["CLV 占比"].max()) * 1.15 + 1)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def share_comment(row):
    """依「CLV 占比 ÷ 人數占比」寫一句說明。"""
    p, c = row["人數占比"], row["CLV 占比"]
    if p == 0:
        return f"**{row['客群']}**：沒有顧客。"
    r = c / p
    if r >= 1.5:
        tail = f"貢獻約是人數比例的 {r:.1f} 倍。"
    elif r >= 0.8:
        tail = "貢獻和人數比例差不多。"
    else:
        tail = "貢獻低於人數比例。"
    return f"**{row['客群']}**：人數 {p:.1f}%，12 個月總 CLV 占 {c:.1f}%，{tail}"
