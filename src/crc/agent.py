"""Tool-calling planning agent with a number verifier.

Backends:
  ollama   - any OpenAI-compatible chat endpoint (default http://localhost:11434/v1, e.g. Ollama with qwen2.5:7b)
  scripted - a deterministic keyword router used only to test the plumbing in CI. It is NOT an LLM and its
             scores must never be reported as model results.

Guardrail: every number in the final answer must match a number returned by a tool (or written in the question).
If not, the agent gets one retry with the unsupported numbers listed, then abstains.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import httpx

from . import tools

SYSTEM = ("You are a retention analyst assistant for an online gift retailer. Answer ONLY with numbers returned by the "
          "tools. Always call a tool before giving any number. If the tools cannot answer the question (for example an "
          "unknown customer, a future year, or how much an offer WILL save, which is unknown until a test is run), reply exactly: "
          "'I can't answer that from the available data.' Keep answers to 1-3 sentences.")
ABSTAIN = "I can't answer that from the available data."
NUM = re.compile(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> list[float]:
    out = []
    for m in NUM.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _flatten(obj) -> list[float]:
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _flatten(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _flatten(v)]
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return [float(obj)]
    if isinstance(obj, str):
        return numbers(obj)
    return []


def unsupported(answer: str, evidence: list, question: str) -> list[float]:
    allowed = [x for e in evidence for x in _flatten(e)] + numbers(question)
    allowed += [a * 100 for a in allowed if abs(a) <= 1]  # 0.875 may be written as 87.5%
    bad = []
    for n in numbers(answer):
        if not any(abs(n - a) <= max(0.051, 0.01 * abs(a)) for a in allowed):
            bad.append(n)
    return bad


@dataclass
class Result:
    question: str
    answer: str
    tool_calls: list = field(default_factory=list)
    verified: bool = True
    retried: bool = False
    abstained: bool = False
    latency_s: float = 0.0


def _call_tool(name: str, args: dict) -> dict:
    fn = tools.REGISTRY.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": str(e)}


class OllamaAgent:
    def __init__(self, model: str = "qwen2.5:7b", base_url: str | None = None, max_steps: int = 5, timeout: float = 120):
        self.model, self.max_steps = model, max_steps
        self.url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")).rstrip("/") + "/chat/completions"
        self.key = os.environ.get("OPENAI_API_KEY", "ollama")
        self.http = httpx.Client(timeout=timeout)

    def _chat(self, messages):
        r = self.http.post(self.url, headers={"Authorization": f"Bearer {self.key}"},
                           json={"model": self.model, "messages": messages, "tools": tools.SCHEMAS, "temperature": 0})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]

    def _loop(self, messages, res: Result, evidence: list) -> str:
        for _ in range(self.max_steps):
            msg = self._chat(messages)
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content") or ""
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls:
                name = c["function"]["name"]
                raw = c["function"].get("arguments") or "{}"
                args = json.loads(raw) if isinstance(raw, str) else raw
                out = _call_tool(name, args)
                res.tool_calls.append({"name": name, "args": args})
                evidence.append(out)
                messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": json.dumps(out)})
        return ABSTAIN

    def ask(self, question: str) -> Result:
        t = time.time()
        res, evidence = Result(question, ""), []
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        ans = self._loop(messages, res, evidence)
        bad = unsupported(ans, evidence, question)
        if bad:
            res.retried = True
            messages.append({"role": "assistant", "content": ans})
            messages.append({"role": "user", "content": f"These numbers are not in any tool result: {bad}. "
                             f"Answer again using only tool results, or say: {ABSTAIN}"})
            ans = self._loop(messages, res, evidence)
            if unsupported(ans, evidence, question):
                res.verified, ans = False, ABSTAIN
        res.answer, res.abstained = ans, ABSTAIN.lower()[:20] in ans.lower()
        res.latency_s = round(time.time() - t, 2)
        return res


class ScriptedAgent:
    """Keyword router for CI only (not an LLM)."""
    CID = re.compile(r"\b(\d{5})\b")

    def ask(self, question: str) -> Result:
        t, q = time.time(), question.lower()
        res = Result(question, "")
        m = self.CID.search(question)
        seg = next((s for s in tools.SEGMENTS if s.replace("_", " ") in q or s in q), None)
        if any(w in q for w in ["will save", "2013", "next year", "guarantee", "will it increase"]):
            res.answer, res.abstained = ABSTAIN, True
        elif m:
            out = tools.get_customer(m.group(1)); res.tool_calls.append({"name": "get_customer", "args": {"customer_id": m.group(1)}})
            if "error" in out:
                res.answer, res.abstained = ABSTAIN, True
            else:
                res.answer = f"Customer {out['customer_id']}: churn probability {out['p_churn']}, segment {out['segment']}, priority {out['priority_rank']}."
        elif seg:
            out = tools.get_segment_summary(seg); res.tool_calls.append({"name": "get_segment_summary", "args": {"segment": seg}})
            res.answer = (f"{seg}: {out['customers']} customers, average churn probability {out['avg_p_churn']}, "
                          f"{out['share_of_revenue']} of last-12-month revenue (£{out['revenue_last_365d_gbp']}).")
        elif "test" in q or "sample" in q or "detect" in q:
            out = tools.get_test_design(); res.tool_calls.append({"name": "get_test_design", "args": {}})
            res.answer = (f"{out['customers_per_arm_for_5pp']} customers per arm to detect 5pp; with {out['customers_per_arm_if_split']} "
                          f"per arm the minimum detectable lift is {out['min_detectable_lift_pp_at_250_per_arm']}pp.")
        else:
            res.answer, res.abstained = ABSTAIN, True
        res.latency_s = round(time.time() - t, 4)
        return res
