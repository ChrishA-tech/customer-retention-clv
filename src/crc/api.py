"""FastAPI service. Run: uvicorn crc.api:app --port 8080"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import tools

app = FastAPI(title="Customer retention & CLV", version="1.0.0")


def _ok(r: dict) -> dict:
    if "error" in r:
        raise HTTPException(404, r["error"])
    return r


@app.get(os.environ.get("AIP_HEALTH_ROUTE", "/health"))
def health() -> dict:
    a = tools.load()
    return {"status": "ok", "customers": len(a["customers"]), "as_of": a["as_of"], "model": a["champion"]}


@app.get("/segments")
def segments() -> dict:
    return tools.list_segments()


@app.get("/segments/{segment}")
def segment(segment: str) -> dict:
    return _ok(tools.get_segment_summary(segment))


@app.get("/customers/{customer_id}")
def customer(customer_id: str) -> dict:
    return _ok(tools.get_customer(customer_id))


@app.get("/priority")
def priority(n: int = 10) -> dict:
    return tools.get_top_priority(n)


@app.get("/test-design")
def test_design() -> dict:
    return tools.get_test_design()


class Batch(BaseModel):
    instances: list[dict]


@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/predict"))
def predict(b: Batch) -> dict:
    return {"predictions": [tools.get_customer(i.get("customer_id", "")) for i in b.instances]}
