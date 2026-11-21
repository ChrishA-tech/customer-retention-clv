"""Load and clean Online Retail II (shared with retail-demand-copilot)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MIRROR = "https://raw.githubusercontent.com/allanvc/onlineretail2/master/data/onlineretail2.rda"
NON_PRODUCT = {"POST", "DOT", "M", "C2", "BANK CHARGES", "AMAZONFEE", "ADJUST", "ADJUST2", "TEST001", "TEST002",
               "PADS", "D", "CRUK", "S", "B", "m", "gift_0001_10", "gift_0001_20", "gift_0001_30", "gift_0001_40",
               "gift_0001_50"}


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".rda":
        import pyreadr
        df = next(iter(pyreadr.read_r(str(path)).values()))
    elif path.suffix in {".xlsx", ".xls"}:
        df = pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
        df = df.rename(columns={"Customer ID": "CustomerID"})
    else:
        df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop cancellations, returns, zero prices, non-product codes and exact duplicates. Returns a log of removals."""
    log = {"raw_rows": len(df)}
    steps = [
        ("cancelled_invoices", df["Invoice"].str.upper().str.startswith("C")),
        ("non_positive_quantity", df["Quantity"] <= 0),
        ("non_positive_price", df["Price"] <= 0),
        ("non_product_codes", df["StockCode"].isin(NON_PRODUCT) | df["StockCode"].str.upper().str.startswith("TEST")),
    ]
    keep = pd.Series(True, index=df.index)
    for name, mask in steps:
        log[name] = int((mask & keep).sum())
        keep &= ~mask
    out = df[keep]
    dup = out.duplicated()
    log["exact_duplicates"] = int(dup.sum())
    out = out[~dup].copy()
    log["clean_rows"] = len(out)
    log["removed_pct"] = round(100 * (1 - len(out) / log["raw_rows"]), 2)
    return out, log


def customer_lines(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean purchase lines with a customer ID, plus the cancellation lines (used for a returns-rate feature)."""
    raw = raw[raw["CustomerID"].notna()].copy()
    raw["CustomerID"] = raw["CustomerID"].astype(int)
    cancels = raw[raw["Invoice"].str.upper().str.startswith("C") & ~raw["StockCode"].isin(NON_PRODUCT)]
    clean_df, log = clean(raw)
    clean_df["revenue"] = clean_df["Quantity"] * clean_df["Price"]
    return clean_df, cancels, log
