"""Tools over the serving artifact; the API and the agent share them."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ARTIFACT = Path("artifacts/serving.json")
SEGMENTS = ["high_value_at_risk", "high_value_stable", "low_value_at_risk", "low_value_stable"]


@lru_cache(maxsize=1)
def load(path: str | None = None) -> dict:
    return json.loads(Path(path or ARTIFACT).read_text())


def list_segments(art: dict | None = None) -> dict:
    art = art or load()
    return {"as_of": art["as_of"], "segments": [{"segment": s["segment"], "customers": s["customers"]} for s in art["segments"]]}


def get_segment_summary(segment: str, art: dict | None = None) -> dict:
    art = art or load()
    for s in art["segments"]:
        if s["segment"] == segment.strip().lower().replace(" ", "_"):
            return {"as_of": art["as_of"], **s}
    return {"error": f"unknown segment {segment}; options: {SEGMENTS}"}


def get_customer(customer_id: str | int, art: dict | None = None) -> dict:
    art = art or load()
    c = art["customers"].get(str(int(float(customer_id)))) if str(customer_id).replace(".", "").isdigit() else None
    if c is None:
        return {"error": f"unknown customer {customer_id}"}
    return {"customer_id": int(float(customer_id)), "as_of": art["as_of"], **c}


def get_top_priority(n: int = 10, art: dict | None = None) -> dict:
    art = art or load()
    n = max(1, min(int(n), 50))
    top = sorted(art["customers"].items(), key=lambda kv: kv[1]["priority_rank"])[:n]
    return {"ranked_by": "churn probability x last-12-month revenue",
            "customers": [{"customer_id": int(k), **v} for k, v in top]}


def get_test_design(art: dict | None = None) -> dict:
    art = art or load()
    return {**art["test_design"], "note": "uplift of a retention offer is unknown until a randomised test is run"}


REGISTRY = {"list_segments": list_segments, "get_segment_summary": get_segment_summary, "get_customer": get_customer,
            "get_top_priority": get_top_priority, "get_test_design": get_test_design}

SCHEMAS = [
    {"type": "function", "function": {"name": "list_segments", "description": "List customer segments and their sizes.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_segment_summary", "description": "Statistics for one segment: high_value_at_risk, high_value_stable, low_value_at_risk or low_value_stable.",
     "parameters": {"type": "object", "properties": {"segment": {"type": "string"}}, "required": ["segment"]}}},
    {"type": "function", "function": {"name": "get_customer", "description": "Churn probability, segment, revenue and priority for one customer ID.",
     "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}}},
    {"type": "function", "function": {"name": "get_top_priority", "description": "Customers with the most revenue at risk, highest first.",
     "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "get_test_design", "description": "Sample size and minimum detectable lift for a retention-offer test.",
     "parameters": {"type": "object", "properties": {}}}},
]
