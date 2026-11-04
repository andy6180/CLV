# -*- coding: utf-8 -*-
"""
analysis.py — 網站的分析結果

  1. 肘部法則（Elbow Method）
  2. 分群結果（F、M 散佈圖）
  3. PCA 特徵對比（三群分開）
  4. 客戶價值人數與百分比
  5. 未來 CLV 預測結果（BG/NBD × Gamma-Gamma，3／6／9／12 個月總額）
  6. 顧客未來 CLV 分客群（人數占比 vs 12 個月總 CLV 占比）

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
COLORS = {"一般客戶": "#9aa3ad", "忠誠客戶": "#2f5fd0", "VIP 客戶": "#d6453d"}
RULES = pd.DataFrame({"客群": ["忠誠客戶", "VIP 客戶", "一般客戶"],
                      "判定依據": ["購買 2 次以上（有回購）",
                               "只買 1 次，消費金額在 K-Means 分界以上",
                               "只買 1 次，消費金額在 K-Means 分界以下"]})
REPEAT_WARN = 0.20   # 回購顧客比例超過此值時，提醒 VIP 定義可能不適用
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
    """只用 F、M 標準化 → K-Means(3)，依平均值命名三群，再套用回購規則。

    R（多久沒買）不放進分群：最近買很多的人一律依消費金額歸類，不會出現
    「算近期還是 VIP」的重疊。顧客會不會流失，交給有用到 R 的 BG/NBD 預測。

    命名有先後順序，已命名的群不再參與下一步，三群各得到不同名稱：
      1. 購買次數（F）平均最高的群 → 忠誠客戶
      2. 其餘兩群中，消費金額（M）平均較高 → VIP 客戶
      3. 剩下的 → 一般客戶
    回購規則（讓每個客群都能用一句話定義）：
      - 購買 2 次以上一律為忠誠客戶；
      - 只買 1 次者只會是 VIP 或一般客戶：K-Means 若把只買 1 次的人分到忠誠群，
        改依距離 VIP、一般兩個群中心何者較近決定。
    因此 VIP 與一般客戶的分界，就是 K-Means 在只買 1 次顧客中找出的金額分界。
    回傳的 X 是 F、M 標準化後的矩陣（肘部法則用）。
    """
    X = StandardScaler().fit_transform(rfm[["F", "M"]].values)
    km = KMeans(BEST_K, random_state=42, n_init=10).fit(X)
    labels, km_centers = km.labels_, km.cluster_centers_
    prof = rfm.assign(k=labels).groupby("k")[["F", "M"]].mean()
    loyal = prof["F"].idxmax()
    rest = prof.drop(loyal)
    vip = rest["M"].idxmax()
    general = rest.drop(vip).index[0]
    names = {loyal: "忠誠客戶", vip: "VIP 客戶", general: "一般客戶"}
    out = rfm.copy()
    out["客群"] = pd.Series(labels, index=rfm.index).map(names)
    # 記錄套用規則前各群的平均 M（說明 VIP 金額分界的由來）
    out.attrs["kmeans_mean_M"] = out.groupby("客群")["M"].mean().to_dict()
    # 回購規則
    once = (out["F"] == 1).values
    stray = once & (out["客群"] == "忠誠客戶").values
    if stray.any():
        cv, cg = km_centers[vip], km_centers[general]
        near_vip = ((X[stray] - cv) ** 2).sum(1) < ((X[stray] - cg) ** 2).sum(1)
        out.loc[stray, "客群"] = np.where(near_vip, "VIP 客戶", "一般客戶")
    out.loc[~once, "客群"] = "忠誠客戶"
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


def vip_threshold(seg):
    """一般客戶與 VIP 客戶之間的金額分界（兩群金額不重疊時才回傳）。"""
    g = seg.loc[seg["客群"] == "一般客戶", "M"]
    v = seg.loc[seg["客群"] == "VIP 客戶", "M"]
    if len(g) and len(v) and g.max() < v.min():
        return (g.max() + v.min()) / 2
    return None


def _per_person(clv_table, name):
    t = clv_table.set_index("客群")
    n = t.loc[name, "人數"]
    return t.loc[name, "12 個月"] / n if n else 0.0


def _threshold_origin(seg, thr):
    """說明 VIP 金額分界：K-Means 一般客戶群與 VIP 群中心（套用回購規則前）的中點。"""
    m = seg.attrs.get("kmeans_mean_M", {})
    g, v = m.get("一般客戶"), m.get("VIP 客戶")
    if not (thr and g is not None and v is not None):
        return "K-Means 依資料找出"
    return (f"K-Means 一般客戶群中心（平均 {g:,.2f}）與 VIP 群中心（平均 {v:,.2f}，套用回購規則前）的中點："
            f"({g:,.2f} + {v:,.2f}) ÷ 2 ≈ {(g + v) / 2:,.0f}；只買 1 次的顧客，總消費離哪個中心近就歸哪一群")


def vip_definition(seg):
    """VIP 客戶定義表（數字依資料計算）與 VIP 中只買 1 次的比例。"""
    v = seg[seg["客群"] == "VIP 客戶"]
    n = len(v)
    thr = vip_threshold(seg)
    once = int((v["F"] == 1).sum())
    multi = v[v["F"] >= 2]
    rng = f"{v['M'].min():,.2f}～{v['M'].max():,.2f}" if n else "—"
    buy = f"全部只買 1 次（{once:,} 人）" if once == n else f"{once:,} 人只買 1 次"
    if len(multi):
        times = (f"{int(multi['F'].min())} 次" if multi["F"].nunique() == 1
                 else f"{int(multi['F'].min())}～{int(multi['F'].max())} 次")
        buy += f"；{len(multi):,} 人買 {times}（金額皆 ≥ {multi['M'].min():,.0f}）"
    avg_all = seg["M"].mean()
    rows = [
        ("定義", "只買過 1 次、但累積消費金額高的顧客（不是買很多次的顧客）"),
        ("判定方式", "只買 1 次的顧客中，消費金額在 K-Means 分界以上者；買 2 次以上一律為忠誠客戶"),
        ("金額門檻", (f"總消費約 {thr:,.0f} 以上（{rng}），由 K-Means 依資料找出" if thr
                      else f"實際範圍 {rng}，由 K-Means 依資料找出")),
        ("門檻由來", _threshold_origin(seg, thr)),
        ("購買次數", buy),
        ("人數", f"{n:,} 人（{n / len(seg):.2%}）"),
        ("平均消費", (f"{v['M'].mean():,.2f}，為全體平均 {avg_all:,.2f} 的 {v['M'].mean() / avg_all:.1f} 倍"
                      if n else "—")),
    ]
    return pd.DataFrame(rows, columns=["項目", "內容"]), (once / n if n else 0.0)


def daily_compare(seg, clv_table):
    """本分群與日常生活認知的比較表（數字依資料計算）。"""
    loyal = seg[seg["客群"] == "忠誠客戶"]
    vip = seg[seg["客群"] == "VIP 客戶"]
    pct = seg["客群"].value_counts(normalize=True)
    lp, vp = _per_person(clv_table, "忠誠客戶"), _per_person(clv_table, "VIP 客戶")
    thr = vip_threshold(seg)

    vip_txt = "只買過 1 次、消費金額高"
    vip_txt += f"；12 個月每人 CLV {vp:,.2f}"
    vip_txt += f"，{'低' if vp < lp else '高'}於忠誠客戶 {lp:,.2f}"
    if not len(loyal):
        loyal_txt = "—"
    else:
        loyal_txt = (f"買 2 次以上即屬之，平均 {loyal['F'].mean():.2f} 次；"
                     f"平均消費 {loyal['M'].mean():,.2f}")
    if len(loyal) and len(vip) and loyal["M"].mean() < vip["M"].mean():
        loyal_txt += "，低於 VIP"
    gen_txt = f"占 {pct.get('一般客戶', 0):.2%}"
    if pct.idxmax() == "一般客戶":
        gen_txt += "，人數最多"
    rows = [
        ("VIP 客戶", "常來又花很多，最有價值", vip_txt),
        ("忠誠客戶", "長期、經常回購", loyal_txt),
        ("一般客戶", "人數不多的普通顧客", gen_txt),
        ("分群門檻", "會員等級、事先訂定",
         f"K-Means 依資料找出（約 {thr:,.0f}），換資料就會改變" if thr else "K-Means 依資料找出，換資料就會改變"),
        ("最近是否購買", "通常會考慮", "分群不使用 Recency"),
    ]
    return pd.DataFrame(rows, columns=["項目", "日常生活認知", "本研究分群"])


def fig_clusters(seg):
    """分群結果：畫在分群實際使用的 F、M 上（橫軸購買次數、縱軸金額取對數），三群可以完全分開。"""
    rng = np.random.default_rng(0)
    cap = 6
    f = np.minimum(seg["F"].values, cap).astype(float)
    x = f + rng.uniform(-0.32, 0.32, len(f))
    y = seg["M"].values
    labels = seg["客群"].values
    n = seg["客群"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150)
    for name in ["一般客戶", "VIP 客戶", "忠誠客戶"]:
        m = labels == name
        if not m.any():
            continue
        small = name == "一般客戶"
        ax.scatter(x[m], y[m], s=4 if small else 6, c=COLORS[name], alpha=.35 if small else .6,
                   linewidths=0, rasterized=True, label=f"{name}（{n.get(name, 0):,} 人）")
    thr = vip_threshold(seg)
    if thr:
        ax.plot([0.62, 1.38], [thr, thr], color="#1f2d3a", lw=1.2, ls="--")
        ax.annotate(f"分界約 {thr:,.0f}", (1.38, thr), xytext=(1.55, thr * 1.9), fontsize=10, color="#1f2d3a",
                    arrowprops=dict(arrowstyle="-", color="#1f2d3a", lw=.8))
    ax.set_yscale("log")
    ax.set_xticks(range(1, cap + 1))
    ax.set_xticklabels([str(i) for i in range(1, cap)] + [f"{cap} 次以上"])
    ax.set_xlim(0.5, cap + 0.5)
    ax.set_xlabel("購買次數 F（次；點左右錯開避免重疊）")
    ax.set_ylabel("總消費金額 M（對數刻度）")
    ax.set_title("K-Means 分群結果（分群使用的 F、M）", fontsize=13)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    lg = ax.legend(loc="upper right", markerscale=4, frameon=False, fontsize=10)
    for h in lg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout()
    return fig


def fig_pca(Z, seg, loadings):
    """PCA 分布，三個客群各畫一張（淺灰為全部顧客）。"""
    labels = seg["客群"].values
    n = seg["客群"].value_counts()
    lo = np.percentile(Z, 0.5, axis=0)
    hi = np.percentile(Z, 99.5, axis=0)
    pad = (hi - lo) * 0.05
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), dpi=150, sharex=True, sharey=True)
    for ax, name in zip(axs, ["忠誠客戶", "VIP 客戶", "一般客戶"]):
        ax.scatter(Z[:, 0], Z[:, 1], s=3, c="#d9dde1", linewidths=0, rasterized=True)
        m = labels == name
        ax.scatter(Z[m, 0], Z[m, 1], s=4, c=COLORS[name] if name != "一般客戶" else "#5d6b78",
                   alpha=.6, linewidths=0, rasterized=True)
        ax.set_title(f"{name}（{n.get(name, 0):,} 人）", fontsize=12, color="#1f2d3a")
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.grid(alpha=.25)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axs[0].set_ylabel(axis_label(loadings, "PC2"))
    axs[1].set_xlabel(axis_label(loadings, "PC1"))
    fig.suptitle("PCA 分布：三個客群分開呈現（淺灰為全部顧客）", fontsize=13)
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
