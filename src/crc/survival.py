"""Time to next order (days), right-censored at the end of the observation window: Kaplan-Meier and Cox PH."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter


def durations(orders_df: pd.DataFrame, feats: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    nxt = orders_df[orders_df["date"] >= cutoff].groupby("CustomerID")["date"].min()
    d = feats.set_index("CustomerID")
    days = (nxt.reindex(d.index) - cutoff).dt.days
    d["event"] = (days.notna() & (days < horizon_days)).astype(int)
    d["duration"] = days.where(d["event"] == 1, horizon_days).clip(lower=0.5)
    return d.reset_index()


def km_by_group(d: pd.DataFrame, group: pd.Series) -> dict:
    out = {}
    for g in sorted(group.unique()):
        m = group == g
        km = KaplanMeierFitter().fit(d.loc[m, "duration"], d.loc[m, "event"])
        med = km.median_survival_time_
        out[str(g)] = {"customers": int(m.sum()), "median_days_to_next_order": None if np.isinf(med) else float(med),
                       "share_ordering_within_90d": round(float(1 - km.survival_function_at_times(90).iloc[0]), 3)}
    return out


def cox(d: pd.DataFrame) -> dict:
    X = pd.DataFrame({"log_recency": np.log1p(d["recency_days"]), "log_frequency": np.log1p(d["frequency"]),
                      "log_revenue_365d": np.log1p(d["revenue_last_365d"]), "return_rate": d["return_rate"],
                      "is_uk": d["is_uk"], "duration": d["duration"], "event": d["event"]})
    m = CoxPHFitter(penalizer=0.01).fit(X, "duration", "event")
    hr = m.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(4)
    return {"concordance": round(float(m.concordance_index_), 4),
            "hazard_ratios": {k: {"hr": float(v["exp(coef)"]), "ci95": [float(v["exp(coef) lower 95%"]), float(v["exp(coef) upper 95%"])],
                                  "p": float(v["p"])} for k, v in hr.iterrows()}}
