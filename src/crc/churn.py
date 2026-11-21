"""Churn = no order in the next 188 days. Trained on an earlier cutoff, tested on a later one (out-of-time)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0)


def _X(df: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    X = df[FEATURES].copy()
    if log:
        for c in ["recency_days", "frequency", "tenure_days", "monetary_total", "avg_order_value", "revenue_last_90d",
                  "revenue_last_365d",
                  "mean_gap_days", "std_gap_days", "distinct_products", "avg_lines_per_order", "days_since_vs_mean_gap"]:
            X[c] = np.log1p(X[c].clip(lower=0))
        X = X.fillna(X.median())
    return X


def ece(y, p, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def fit_models(train: pd.DataFrame, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cal = rng.random(len(train)) < 0.2  # held out from training, used only for isotonic calibration
    tr, ca = train[~cal], train[cal]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(_X(tr, True), tr["churned"])
    gbm = lgb.LGBMClassifier(**LGB).fit(_X(tr), tr["churned"])
    iso = IsotonicRegression(out_of_bounds="clip").fit(gbm.predict_proba(_X(ca))[:, 1], ca["churned"])
    val_auc = {"logistic": roc_auc_score(ca["churned"], lr.predict_proba(_X(ca, True))[:, 1]),
               "lightgbm": roc_auc_score(ca["churned"], gbm.predict_proba(_X(ca))[:, 1])}
    # prefer the simpler, interpretable model unless the boosted model is clearly better on validation
    champion = "lightgbm" if val_auc["lightgbm"] - val_auc["logistic"] > 0.005 else "logistic"
    return {"logistic": lr, "lightgbm": gbm, "isotonic": iso, "n_train": int((~cal).sum()), "n_cal": int(cal.sum()),
            "validation_auc": {k: round(v, 4) for k, v in val_auc.items()}, "champion": champion}


def champion_proba(models: dict, df: pd.DataFrame) -> np.ndarray:
    if models["champion"] == "logistic":
        return models["logistic"].predict_proba(_X(df, True))[:, 1]
    return models["isotonic"].predict(models["lightgbm"].predict_proba(_X(df))[:, 1])


def score(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame({"CustomerID": df["CustomerID"].to_numpy(), "churned": df["churned"].to_numpy()})
    s["bgnbd_1_minus_p_alive"] = 1 - df["p_alive"].to_numpy()
    s["logistic"] = models["logistic"].predict_proba(_X(df, True))[:, 1]
    raw = models["lightgbm"].predict_proba(_X(df))[:, 1]
    s["lightgbm_raw"] = raw
    s["lightgbm_calibrated"] = models["isotonic"].predict(raw)
    return s


def metrics(s: pd.DataFrame, cols: list[str]) -> dict:
    y = s["churned"].to_numpy()
    return {c: {"roc_auc": round(roc_auc_score(y, s[c]), 4), "pr_auc": round(average_precision_score(y, s[c]), 4),
                "brier": round(brier_score_loss(y, s[c].clip(0, 1)), 4), "ece": round(ece(y, s[c].clip(0, 1).to_numpy()), 4)}
            for c in cols}


def auc_diff_ci(s: pd.DataFrame, a: str, b: str, n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    y, pa, pb = s["churned"].to_numpy(), s[a].to_numpy(), s[b].to_numpy()
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"auc_diff": round(roc_auc_score(y, pa) - roc_auc_score(y, pb), 4), "ci95": [round(lo, 4), round(hi, 4)]}
