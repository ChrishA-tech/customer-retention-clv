"""Who to contact with a retention offer, and how big the test must be to measure whether it works.

Uplift from an offer is unknown here (the data has no randomised campaign), so this module (1) ranks customers by
revenue at risk and checks on the holdout how much lost revenue each ranking would have reached, and (2) sizes a
randomised test that would measure the real uplift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def capture(df: pd.DataFrame, score_col: str, k: int) -> dict:
    """Share of the revenue lost to churners (their last-12-month spend) reached by contacting the top-k."""
    lost = df["monetary_12m"] * df["churned"]
    top = df.nlargest(k, score_col)
    return {"churners_reached": int(top["churned"].sum()), "lost_revenue_reached_gbp": round(float(lost.loc[top.index].sum()), 0),
            "share_of_lost_revenue": round(float(lost.loc[top.index].sum() / lost.sum()), 4)}


def random_capture(df: pd.DataFrame, k: int, n: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    lost = (df["monetary_12m"] * df["churned"]).to_numpy()
    shares = [lost[rng.choice(len(df), k, replace=False)].sum() / lost.sum() for _ in range(n)]
    return {"share_of_lost_revenue": round(float(np.mean(shares)), 4)}


def test_size(base_retention: float, lift_pp: float, alpha: float = 0.05, power: float = 0.8) -> int:
    es = proportion_effectsize(min(base_retention + lift_pp, 0.999), base_retention)
    return int(np.ceil(NormalIndPower().solve_power(effect_size=abs(es), alpha=alpha, power=power, ratio=1.0)))


def mde(base_retention: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    es = NormalIndPower().solve_power(nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0)
    # invert Cohen's h numerically
    for lift in np.arange(0.001, 0.6, 0.001):
        if abs(proportion_effectsize(min(base_retention + lift, 0.999), base_retention)) >= es:
            return round(float(lift), 3)
    return float("nan")
