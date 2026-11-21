import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("reports")


def main():
    r = json.loads((R / "results.json").read_text())
    t = r["targeting"]
    labels = ["Random", "Longest since\nlast order", "Highest churn\nprobability", "Churn probability\n× 12-month spend"]
    vals = [t["random"]["share_of_lost_revenue"], t["highest_recency"]["share_of_lost_revenue"],
            t["highest_p_churn"]["share_of_lost_revenue"], t["value_at_risk"]["share_of_lost_revenue"]]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(labels, [100 * v for v in vals], color=["0.75", "0.75", "0.75", "k"])
    for i, v in enumerate(vals):
        ax.text(i, 100 * v + 1, f"{100 * v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("% of revenue lost to churn reached")
    ax.set_title(f"Contacting {t['k']} customers: who to pick (held-out period)")
    ax.set_ylim(0, 55); ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(R / "01_targeting.png", dpi=150); plt.close(fig)

    km = r["survival"]["kaplan_meier_by_orders"]
    order = ["1 order", "2-3", "4-10", "11+"]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(order, [100 * km[k]["share_ordering_within_90d"] for k in order], color="0.3")
    for i, k in enumerate(order):
        ax.text(i, 100 * km[k]["share_ordering_within_90d"] + 1.5, f'{100 * km[k]["share_ordering_within_90d"]:.0f}%', ha="center", fontsize=9)
    ax.set_xlabel("Orders before the cutoff"); ax.set_ylabel("% ordering again within 90 days")
    ax.set_title("Kaplan-Meier: repeat buyers come back far sooner"); ax.set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(R / "02_reorder_km.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
