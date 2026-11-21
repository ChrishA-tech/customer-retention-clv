"""Agent evaluation set built from the serving artifact (every answerable question has a known number)."""
import json
import random
from pathlib import Path

art = json.loads(Path("artifacts/serving.json").read_text())
rng = random.Random(11)
qs = []
for s in art["segments"]:
    name = s["segment"].replace("_", " ")
    qs.append({"id": f"seg_n_{s['segment']}", "type": "segment_size", "question": f"How many customers are in the {name} segment?", "expected": [s["customers"]]})
    qs.append({"id": f"seg_p_{s['segment']}", "type": "segment_churn", "question": f"What is the average churn probability in the {name} segment?", "expected": [s["avg_p_churn"]]})
    qs.append({"id": f"seg_r_{s['segment']}", "type": "segment_revenue", "question": f"How much revenue did the {name} segment bring in over the last 12 months?", "expected": [s["revenue_last_365d_gbp"]]})
ids = sorted(art["customers"], key=lambda k: art["customers"][k]["priority_rank"])
picks = ids[:4] + rng.sample(ids[4:], 6)
for c in picks:
    v = art["customers"][c]
    qs.append({"id": f"cust_{c}", "type": "customer_churn", "question": f"What is the churn probability of customer {c}?", "expected": [v["p_churn"]]})
td = art["test_design"]
qs += [{"id": "test_5pp", "type": "test_design", "question": "How many customers per arm would a retention test need to detect a 5 point lift?", "expected": [td["customers_per_arm_for_5pp"]]},
       {"id": "test_mde", "type": "test_design", "question": "If we split our 500 priority customers into two arms, what is the smallest lift we could detect?", "expected": [td["min_detectable_lift_pp_at_250_per_arm"]]}]
traps = ["How much revenue will a 10% discount offer save from the high value at risk segment?",
         "What is the churn probability of customer 99999?",
         "Which customers will churn in 2013?",
         "Can you guarantee that emailing the priority list will increase retention?",
         "What is the churn probability of customer 00001?",
         "How much will retention improve next year if we run the campaign?"]
qs += [{"id": f"abstain_{i}", "type": "abstain", "question": q, "expected": None} for i, q in enumerate(traps)]
Path("eval/questions.jsonl").write_text("\n".join(json.dumps(q) for q in qs) + "\n")
print(len(qs), "questions,", len(traps), "unanswerable")
