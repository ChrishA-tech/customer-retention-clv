import numpy as np
import pandas as pd

from crc import agent, features, targeting


def _lines():
    rows = []
    for cid, dates in {1: ["2010-01-05", "2010-03-01", "2010-06-01"], 2: ["2010-02-01"], 3: ["2010-05-20", "2010-07-15"]}.items():
        for i, d in enumerate(dates):
            rows.append({"CustomerID": cid, "Invoice": f"{cid}{i}", "InvoiceDate": pd.Timestamp(d), "StockCode": "X",
                         "Quantity": 2, "Price": 5.0, "revenue": 10.0, "Country": "United Kingdom"})
    return pd.DataFrame(rows)


def test_features_use_only_history_and_label_uses_future():
    f = features.build(_lines(), _lines().iloc[0:0], pd.Timestamp("2010-06-15"), 60).set_index("CustomerID")
    assert f.loc[1, "frequency"] == 3 and f.loc[1, "recency_days"] == 14
    assert f.loc[3, "frequency"] == 1 and f.loc[3, "churned"] == 0  # orders on 2010-07-15, inside the window
    assert f.loc[2, "churned"] == 1


def test_test_size_and_mde_are_consistent():
    n = targeting.test_size(0.65, 0.10)
    assert 300 < n < 400
    assert abs(targeting.mde(0.65, n) - 0.10) < 0.01


def test_capture_counts_only_churners_revenue():
    df = pd.DataFrame({"monetary_12m": [100, 50, 10], "churned": [1, 0, 1], "s": [3, 2, 1]})
    c = targeting.capture(df, "s", 2)
    assert c["lost_revenue_reached_gbp"] == 100 and c["share_of_lost_revenue"] == round(100 / 110, 4)


def test_verifier():
    assert agent.unsupported("Churn probability is 0.57.", [{"p": 0.572}], "q") == []
    assert agent.unsupported("About 1,200 customers.", [{"n": 1162}], "q") == [1200.0]
