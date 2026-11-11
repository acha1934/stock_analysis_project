"""Basic API endpoint tests using FastAPI's TestClient.

Run with:
    pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_get_tools():
    resp = client.get("/api/tools")
    assert resp.status_code == 200
    tools = resp.json()
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert "name" in tools[0]


def test_chat_empty_query():
    resp = client.post("/api/chat", json={"query": ""})
    assert resp.status_code == 400


def test_chat_valid_query():
    resp = client.post("/api/chat", json={"query": "What is the price of TCS?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "routing_mode" in data
    assert "latency_ms" in data
