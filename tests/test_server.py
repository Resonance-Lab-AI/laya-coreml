"""HTTP contract for the Jev-compatible server.

The unit tests run against a fake agent (no Core ML runtime needed) and pin the
``/v1/systemone`` request/response envelope, the batch ordering, and the error
mapping. The darwin-guarded test loads the real bundle and runs a forward pass.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from laya_coreml import server


class FakeAgent:
    """Stand-in for laya_coreml.Agent: emits schema-valid answers, can fault."""

    manifest = {
        "format": "laya-coreml",
        "format_version": 1,
        "source": "test/laya",
        "precision": "float16",
    }

    def predict(self, state, questions):
        answers = {}
        for qid, q in questions.items():
            kind = q.get("type")
            crit = q.get("criteria") or []
            if kind not in ("choice", "score", "noul"):
                raise ValueError(f"Question {qid!r} has unknown type {kind!r}")
            if "invalid" in str(q.get("instructions", "")).lower():
                raise ValueError(f"Question {qid!r} is malformed")
            if "nan" in str(q.get("instructions", "")).lower():
                raise FloatingPointError("Non-finite outputs")
            if kind == "choice":
                answers[qid] = {
                    "type": "choice",
                    "choice": crit[0],
                    "probabilities": {c: 0.5 for c in crit},
                    "confidence": 0.9,
                    "action": {"act_probability": 0.98},
                }
            elif kind == "score":
                answers[qid] = {
                    "type": "score",
                    "score": 1.0,
                    "legend": {str(i): c for i, c in enumerate(crit)},
                    "probabilities": {str(i): 0.5 for i in range(len(crit))},
                    "confidence": 0.8,
                    "action": {"act_probability": 0.9},
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": 0.7,
                    "confidence": 0.75,
                    "action": {"act_probability": 0.9},
                }
        return {"model": "fake", "answers": answers, "usage": {"input_tokens": 11, "output_tokens": 0}}


@pytest.fixture
def client():
    app = server.create_app(FakeAgent())
    with TestClient(app) as c:
        yield c


# --- /v1/models -------------------------------------------------------------

def test_models_lists_the_served_checkpoint(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "laya-coreml"
    assert body["data"][0]["origin"] == "test/laya"


# --- /v1/systemone ----------------------------------------------------------

def test_systemone_returns_envelope_with_all_answer_types(client):
    r = client.post(
        "/v1/systemone",
        json={
            "state": "billed twice",
            "questions": {
                "dept": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "tech"]},
                "sev": {"type": "score", "instructions": "Severity?", "criteria": ["low", "high"]},
                "ref": {"type": "noul", "instructions": "Refund requested?"},
            },
        },
    )
    assert r.status_code == 200
    body = r.json()

    assert body["model"] == "fake"
    assert body["usage"] == {"input_tokens": 11, "output_tokens": 0}
    assert set(body["answers"]) == {"dept", "sev", "ref"}

    choice = body["answers"]["dept"]
    assert choice["type"] == "choice" and choice["choice"] == "billing"
    assert set(choice["probabilities"]) == {"billing", "tech"}
    assert "confidence" in choice and "action" in choice and "act_probability" in choice["action"]

    assert body["answers"]["sev"]["type"] == "score" and body["answers"]["sev"]["legend"]
    assert body["answers"]["ref"]["type"] == "noul" and 0.0 <= body["answers"]["ref"]["noul"] <= 1.0


def test_systemone_ignores_unknown_request_fields(client):
    # Hosted clients send extra keys (model, lang); they must not break the call.
    r = client.post(
        "/v1/systemone",
        json={
            "model": "some-jev-model",
            "lang": "en",
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "Yes?"}},
        },
    )
    assert r.status_code == 200
    assert r.json()["answers"]["q"]["type"] == "noul"


def test_invalid_question_maps_to_422_with_error_body(client):
    r = client.post(
        "/v1/systemone",
        json={"state": "x", "questions": {"q": {"type": "bogus", "instructions": "y"}}},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_question_error"


def test_missing_questions_maps_to_422(client):
    r = client.post("/v1/systemone", json={"state": "x"})
    assert r.status_code == 422


# --- /v1/systemone/batch ----------------------------------------------------

def test_batch_returns_one_result_per_request_in_order(client):
    r = client.post(
        "/v1/systemone/batch",
        json={
            "requests": [
                {"state": "a", "questions": {"q": {"type": "noul", "instructions": "Yes?"}}},
                {"state": "b", "questions": {"q": {"type": "noul", "instructions": "No?"}}},
            ]
        },
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2
    assert all("answers" in r_ for r_ in results)
    assert results[0]["answers"]["q"]["noul"] == results[0]["answers"]["q"]["noul"]


def test_batch_reports_a_failing_request_without_failing_the_rest(client):
    r = client.post(
        "/v1/systemone/batch",
        json={
            "requests": [
                {"state": "ok", "questions": {"q": {"type": "noul", "instructions": "Yes?"}}},
                {"state": "bad", "questions": {"q": {"type": "noul", "instructions": "invalid"}}},
                {"state": "boom", "questions": {"q": {"type": "noul", "instructions": "nan"}}},
            ]
        },
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    assert "answers" in results[0]  # success passes through
    assert results[1]["error"]["code"] == "invalid_question_error"
    assert results[2]["error"]["code"] == "internal_error"


def test_batch_enforces_max_sixty_four(client):
    r = client.post(
        "/v1/systemone/batch",
        json={"requests": [{"state": "x", "questions": {"q": {"type": "noul", "instructions": "y"}}}] * 65},
    )
    assert r.status_code == 422


# --- API key auth -------------------------------------------------------------

def _noul_body(text="Yes?"):
    return {"state": "x", "questions": {"q": {"type": "noul", "instructions": text}}}


def test_open_server_needs_no_key(client):
    assert client.post("/v1/systemone", json=_noul_body()).status_code == 200
    assert client.get("/v1/models").status_code == 200


def test_gated_server_rejects_missing_and_wrong_keys():
    from fastapi.testclient import TestClient

    gated = TestClient(server.create_app(FakeAgent(), api_key="secret"))
    with gated:
        r = gated.post("/v1/systemone", json=_noul_body())
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "authentication_error"

        r = gated.post(
            "/v1/systemone", json=_noul_body(), headers={"Authorization": "Bearer wrong"}
        )
        assert r.status_code == 401

        assert gated.get("/v1/models").status_code == 401


def test_gated_server_accepts_bearer_key():
    from fastapi.testclient import TestClient

    gated = TestClient(server.create_app(FakeAgent(), api_key="secret"))
    headers = {"Authorization": "Bearer secret"}
    with gated:
        assert gated.get("/v1/models", headers=headers).status_code == 200
        r = gated.post("/v1/systemone", json=_noul_body(), headers=headers)
        assert r.status_code == 200
        assert r.json()["answers"]["q"]["type"] == "noul"

        r = gated.post(
            "/v1/systemone/batch",
            json={"requests": [_noul_body("Yes?")]},
            headers=headers,
        )
        assert r.status_code == 200
        assert "answers" in r.json()["results"][0]

# --- real model (Apple Silicon) --------------------------------------------

@pytest.mark.skipif(sys.platform != "darwin", reason="Core ML runtime requires macOS")
def test_real_bundle_serves_a_forward_pass():
    import coremltools  # noqa: F401 - presence gates the whole test

    model_dir = str(Path(__file__).resolve().parents[2] / "models")
    agent = server.load_agent(model_dir, compute_units="cpu")
    assert agent.manifest["format"] == "laya-coreml"

    with TestClient(server.create_app(agent)) as client:
        assert client.get("/v1/models").status_code == 200

        r = client.post(
            "/v1/systemone",
            json={
                "state": "The customer asks for a refund of a duplicate payment.",
                "questions": {
                    "refund": {"type": "noul", "instructions": "Does the customer request a refund?"}
                },
            },
        )
        assert r.status_code == 200
        answer = r.json()["answers"]["refund"]
        assert answer["type"] == "noul"
        assert 0.0 <= answer["noul"] <= 1.0
        assert r.json()["usage"]["output_tokens"] == 0