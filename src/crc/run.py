"""End-to-end: CLV holdout test, out-of-time churn test, survival, targeting, test sizing, serving artifact."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import churn, clv, data, features, survival, targeting

warnings.filterwarnings("ignore")
H = 188  # days: 2011-06-05 -> 2011-12-10 (end of data)
TRAIN_CUT, TEST_CUT, END = pd.Timestamp("2010-11-29"), pd.Timestamp("2011-06-05"), pd.Timestamp("2011-12-10")


def clv_holdout(o: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    s = clv.rfm(o, TEST_CUT)
    bgf, ggf = clv.fit(s)
    p = clv.predict(bgf, ggf, s, H)
    act = o[o["date"] >= TEST_CUT].groupby("CustomerID").agg(orders=("date", "nunique"), revenue=("revenue", "sum"))
    act = act.reindex(s.index).fillna(0)
    naive_orders = s["frequency"] / s["T"].clip(lower=1) * H
    hist_rev = o[(o["date"] < TEST_CUT) & (o["date"] >= TEST_CUT - pd.Timedelta(days=H))].groupby("CustomerID")["revenue"].sum().reindex(s.index).fillna(0)
    top = p["pred_revenue"].rank(ascending=False, method="first") <= len(p) * 0.1
    top_naive = hist_rev.rank(ascending=False, method="first") <= len(p) * 0.1
    res = {
        "customers": len(s), "params": {k: round(float(v), 4) for k, v in bgf.params_.items()}, "penalizer": bgf.penalizer_used,
        "gamma_gamma_q": round(p.attrs["gamma_gamma_q"], 3),
        "order_value_method": "Gamma-Gamma" if p.attrs["gamma_gamma_q"] > 1 else "shrunk observed mean (Gamma-Gamma q <= 1)",
        "total_repeat_orders": {"predicted": round(float(p["pred_orders"].sum()), 0), "actual": int(act["orders"].sum())},
        "mae_orders": {"bgnbd": round(float((p["pred_orders"] - act["orders"]).abs().mean()), 3),
                       "naive_rate": round(float((naive_orders - act["orders"]).abs().mean()), 3)},
        "spearman_revenue": {"bgnbd_x_order_value": round(float(spearmanr(p["pred_revenue"], act["revenue"]).correlation), 3),
                             "last_188d_revenue": round(float(spearmanr(hist_rev, act["revenue"]).correlation), 3)},
        "top10pct_revenue_capture": {"bgnbd_x_order_value": round(float(act["revenue"][top].sum() / act["revenue"].sum()), 4),
                                     "last_188d_revenue": round(float(act["revenue"][top_naive].sum() / act["revenue"].sum()), 4)},
        "one_time_buyers": {"customers": int((s["frequency"] == 0).sum()),
                            "p_alive_assigned": round(float(p.loc[s["frequency"] == 0, "p_alive"].mean()), 3)},
    }
    return res, p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/onlineretail2.rda")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args(argv)
    out, art = Path(a.out), Path("artifacts"); out.mkdir(exist_ok=True); art.mkdir(exist_ok=True)
    raw = data.load_raw(a.data)
    lines, canc, log = data.customer_lines(raw)
    o = features.orders(lines)

    clv_res, clv_pred = clv_holdout(o)

    tr = features.build(lines, canc, TRAIN_CUT, H)  # BG/NBD p_alive is only a test-time baseline, not a feature
    te = features.build(lines, canc, TEST_CUT, H, clv_pred["p_alive"])
    m = churn.fit_models(tr)
    sc = churn.score(m, te)
    cols = ["bgnbd_1_minus_p_alive", "logistic", "lightgbm_raw", "lightgbm_calibrated"]
    one_time = te["frequency"] == 1
    churn_res = {"train_cutoff": str(TRAIN_CUT.date()), "test_cutoff": str(TEST_CUT.date()), "horizon_days": H,
                 "train_customers": len(tr), "test_customers": len(te),
                 "churn_rate": {"train": round(float(tr["churned"].mean()), 4), "test": round(float(te["churned"].mean()), 4)},
                 "one_time_buyer_churn_rate_test": round(float(te.loc[one_time, "churned"].mean()), 4),
                 "repeat_buyer_churn_rate_test": round(float(te.loc[~one_time, "churned"].mean()), 4),
                 "validation_auc": m["validation_auc"], "champion_chosen_on_validation": m["champion"],
                 "test": churn.metrics(sc, cols),
                 "lightgbm_minus_logistic_auc": churn.auc_diff_ci(sc, "lightgbm_raw", "logistic")}

    d = survival.durations(o, te, TEST_CUT, H)
    band = pd.cut(d["frequency"], [0, 1, 3, 10, 10_000], labels=["1 order", "2-3", "4-10", "11+"])
    surv_res = {"kaplan_meier_by_orders": survival.km_by_group(d, band.astype(str)), "cox": survival.cox(d)}

    te = te.assign(p_churn=churn.champion_proba(m, te))
    te["value_at_risk"] = te["p_churn"] * te["revenue_last_365d"]
    te["monetary_12m"] = te["revenue_last_365d"]
    K = 500
    tgt = {"k": K, "random": targeting.random_capture(te, K),
           "highest_recency": targeting.capture(te, "recency_days", K),
           "highest_p_churn": targeting.capture(te, "p_churn", K),
           "value_at_risk": targeting.capture(te, "value_at_risk", K)}
    top = te.nlargest(K, "value_at_risk")
    base_ret = 1 - float(top["churned"].mean())
    tgt["test_design"] = {"targeted_base_retention": round(base_ret, 4), "customers_per_arm_if_split": K // 2,
                          "min_detectable_lift_pp_at_250_per_arm": round(100 * targeting.mde(base_ret, K // 2), 1),
                          "customers_per_arm_for_5pp": targeting.test_size(base_ret, 0.05),
                          "customers_per_arm_for_10pp": targeting.test_size(base_ret, 0.10),
                          "alpha": 0.05, "power": 0.8}

    # serving: features at the end of the data, model refit on the test cutoff (labels now known)
    m_final = churn.fit_models(te.drop(columns=["p_churn", "value_at_risk", "monetary_12m"]))
    s_all = clv.rfm(o, END); bg_all, gg_all = clv.fit(s_all); p_all = clv.predict(bg_all, gg_all, s_all, H)
    now = features.build(lines, canc, END, 1, p_all["p_alive"])
    now["p_churn"] = churn.champion_proba(m_final, now)
    now = now.merge(p_all[["pred_orders", "pred_revenue"]], left_on="CustomerID", right_index=True, how="left")
    now["value_at_risk"] = now["p_churn"] * now["revenue_last_365d"]
    v_hi = now["revenue_last_365d"].quantile(0.8)
    now["segment"] = np.select([(now["revenue_last_365d"] >= v_hi) & (now["p_churn"] >= 0.5),
                                (now["revenue_last_365d"] >= v_hi) & (now["p_churn"] < 0.5),
                                (now["revenue_last_365d"] < v_hi) & (now["p_churn"] >= 0.5)],
                               ["high_value_at_risk", "high_value_stable", "low_value_at_risk"], "low_value_stable")
    seg = now.groupby("segment").agg(customers=("CustomerID", "size"), avg_p_churn=("p_churn", "mean"),
                                     revenue_last_365d_gbp=("revenue_last_365d", "sum"), avg_orders=("frequency", "mean"),
                                     predicted_revenue_next_188d_gbp=("pred_revenue", "sum"), avg_recency_days=("recency_days", "mean")).round(3)
    seg["share_of_revenue"] = (seg["revenue_last_365d_gbp"] / seg["revenue_last_365d_gbp"].sum()).round(4)
    custs = {int(r.CustomerID): {"segment": r.segment, "p_churn": round(float(r.p_churn), 3),
                                 "revenue_last_365d_gbp": round(float(r.revenue_last_365d), 2), "orders": int(r.frequency),
                                 "recency_days": int(r.recency_days),
                                 "predicted_revenue_next_188d_gbp": round(float(r.pred_revenue), 2) if pd.notna(r.pred_revenue) else None,
                                 "priority_rank": int(i + 1)}
             for i, r in enumerate(now.sort_values("value_at_risk", ascending=False).itertuples())}
    serving = {"as_of": str(END.date()), "champion": m_final["champion"], "segments": seg.reset_index().to_dict("records"),
               "customers": custs, "test_design": tgt["test_design"]}
    (art / "serving.json").write_text(json.dumps(serving))
    res = {"data": {"source": "UCI Online Retail II (Chen, 2012), CC BY 4.0", **log, "customers": int(lines["CustomerID"].nunique()),
                    "orders": len(o), "first_day": str(o["date"].min().date()), "last_day": str(o["date"].max().date())},
           "clv_holdout": clv_res, "churn": churn_res, "survival": surv_res, "targeting": tgt,
           "segments_as_of_end": seg.reset_index().to_dict("records")}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    sc.to_csv(out / "churn_scores_test.csv", index=False)
    print(json.dumps({k: res[k] for k in ["clv_holdout", "churn", "targeting"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
