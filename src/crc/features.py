"""Customer features as of a cutoff date, using only lines before the cutoff, and labels from the window after it."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["recency_days", "frequency", "tenure_days", "monetary_total", "avg_order_value", "orders_last_90d",
            "revenue_last_90d", "revenue_last_365d", "mean_gap_days", "std_gap_days", "days_since_vs_mean_gap", "distinct_products",
            "avg_lines_per_order", "return_rate", "is_uk"]


def orders(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per invoice (an order), dated to the day."""
    o = lines.groupby(["CustomerID", "Invoice"]).agg(date=("InvoiceDate", "min"), revenue=("revenue", "sum"),
                                                    n_lines=("StockCode", "size"), country=("Country", "first"))
    o = o.reset_index()
    o["date"] = o["date"].dt.normalize()
    return o


def build(lines: pd.DataFrame, cancels: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int,
          p_alive: pd.Series | None = None) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    hist = lines[lines["InvoiceDate"] < cutoff]
    fut = lines[(lines["InvoiceDate"] >= cutoff) & (lines["InvoiceDate"] < cutoff + pd.Timedelta(days=horizon_days))]
    o = orders(hist)
    g = o.groupby("CustomerID")
    f = pd.DataFrame({
        "first": g["date"].min(), "last": g["date"].max(), "frequency": g["date"].nunique(),
        "monetary_total": g["revenue"].sum(), "avg_order_value": g["revenue"].mean(),
        "avg_lines_per_order": g["n_lines"].mean(), "is_uk": (g["country"].first() == "United Kingdom").astype(int)})
    f["recency_days"] = (cutoff - f["last"]).dt.days
    f["tenure_days"] = (cutoff - f["first"]).dt.days
    recent = o[o["date"] >= cutoff - pd.Timedelta(days=90)].groupby("CustomerID")
    f["orders_last_90d"] = recent["date"].nunique().reindex(f.index).fillna(0)
    f["revenue_last_90d"] = recent["revenue"].sum().reindex(f.index).fillna(0)
    f["revenue_last_365d"] = o[o["date"] >= cutoff - pd.Timedelta(days=365)].groupby("CustomerID")["revenue"].sum().reindex(f.index).fillna(0)
    days = o.drop_duplicates(["CustomerID", "date"]).sort_values("date")
    gaps = days.groupby("CustomerID")["date"].diff().dt.days
    gg = gaps.groupby(days["CustomerID"])
    f["mean_gap_days"] = gg.mean().reindex(f.index)
    f["std_gap_days"] = gg.std().reindex(f.index)
    f["days_since_vs_mean_gap"] = f["recency_days"] / f["mean_gap_days"]
    f["distinct_products"] = hist.groupby("CustomerID")["StockCode"].nunique().reindex(f.index)
    c = cancels[cancels["InvoiceDate"] < cutoff].groupby("CustomerID")["Invoice"].nunique()
    f["return_rate"] = (c.reindex(f.index).fillna(0) / f["frequency"]).clip(0, 5)
    f["p_alive"] = np.nan if p_alive is None else p_alive.reindex(f.index)
    fo = orders(fut).groupby("CustomerID")
    f["future_orders"] = fo["date"].nunique().reindex(f.index).fillna(0)
    f["future_revenue"] = fo["revenue"].sum().reindex(f.index).fillna(0)
    f["churned"] = (f["future_orders"] == 0).astype(int)
    f["cutoff"] = cutoff
    return f.drop(columns=["first", "last"]).reset_index()
