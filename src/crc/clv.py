"""Probabilistic customer lifetime value: BG/NBD for repeat purchases, Gamma-Gamma for order value
(Fader, Hardie & Lee, 2005), fitted with the lifetimes package on a calibration window and checked on a holdout."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data


def rfm(orders_df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Daily transactions before cutoff -> frequency (repeat days), recency, T (days), monetary (mean repeat order value)."""
    tx = orders_df[orders_df["date"] < cutoff].groupby(["CustomerID", "date"], as_index=False)["revenue"].sum()
    return summary_data_from_transaction_data(tx, "CustomerID", "date", monetary_value_col="revenue",
                                              observation_period_end=cutoff - pd.Timedelta(days=1), freq="D")


def fit(s: pd.DataFrame, penalizers=(0.001, 0.003, 0.01, 0.03)):
    """Fit BG/NBD with the smallest penalizer that converges (lifetimes' optimiser can fail on short histories)."""
    from lifetimes.utils import ConvergenceError
    for penalizer in penalizers:
        try:
            bgf = BetaGeoFitter(penalizer_coef=penalizer).fit(s["frequency"], s["recency"], s["T"])
            break
        except ConvergenceError:
            continue
    else:
        raise RuntimeError("BG/NBD did not converge")
    bgf.penalizer_used = penalizer
    rep = s[(s["frequency"] > 0) & (s["monetary_value"] > 0)]
    ggf = GammaGammaFitter(penalizer_coef=penalizer).fit(rep["frequency"], rep["monetary_value"])
    return bgf, ggf


def predict(bgf, ggf, s: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    out = pd.DataFrame(index=s.index)
    out["pred_orders"] = bgf.conditional_expected_number_of_purchases_up_to_time(horizon_days, s["frequency"], s["recency"], s["T"])
    out["p_alive"] = bgf.conditional_probability_alive(s["frequency"], s["recency"], s["T"])
    rep = s[(s["frequency"] > 0) & (s["monetary_value"] > 0)]
    pop = float(rep["monetary_value"].median())
    if ggf.params_["q"] > 1:
        aov = ggf.conditional_expected_average_profit(s["frequency"], s["monetary_value"])
    else:
        # q <= 1 means the Gamma-Gamma prior has no finite mean, so its estimates are unusable; shrink each
        # customer's observed mean toward the population median instead, weighted by repeat orders
        w = s["frequency"] / (s["frequency"] + 2)
        aov = w * s["monetary_value"].where(s["monetary_value"] > 0, pop) + (1 - w) * pop
    # one-time buyers have no repeat order value: use the population median of repeat orders
    out["pred_aov"] = np.where(s["frequency"] > 0, aov, pop)
    out.attrs["gamma_gamma_q"] = float(ggf.params_["q"])
    out["pred_revenue"] = out["pred_orders"] * out["pred_aov"]
    return out
