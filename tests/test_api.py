from fastapi.testclient import TestClient

from crc import tools
from crc.agent import ScriptedAgent
from crc.api import app

c = TestClient(app)
cid = next(iter(tools.load()["customers"]))


def test_health_and_segments():
    assert c.get("/health").json()["customers"] > 0
    assert len(c.get("/segments").json()["segments"]) == 4


def test_customer_and_404():
    r = c.get(f"/customers/{cid}").json(); assert 0 <= r["p_churn"] <= 1
    assert c.get("/customers/99999999").status_code == 404


def test_predict_contract():
    assert len(c.post("/predict", json={"instances": [{"customer_id": cid}]}).json()["predictions"]) == 1


def test_scripted_agent_refuses_uplift_claims():
    assert ScriptedAgent().ask("Which customers will churn in 2013?").abstained
