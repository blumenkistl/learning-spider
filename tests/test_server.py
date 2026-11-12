import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from spider.life import Life
from spider.server import make_app


@pytest.fixture
async def client(tmp_path):
    life = Life("random", seed=1, pace=0.0, runs=tmp_path)
    for _ in range(3):
        life.tick()
    app = make_app(life)
    async with TestClient(TestServer(app)) as client:
        yield client


async def test_index_and_static(client):
    page = await client.get("/")
    assert page.status == 200
    text = await page.text()
    assert "<canvas" in text and "/static/app.js" in text
    script = await client.get("/static/app.js")
    assert script.status == 200
    three = await client.get("/static/vendor/three/three.module.min.js")
    assert three.status == 200


async def test_state_and_describe(client):
    state = await (await client.get("/api/state")).json()
    assert state["moments"] == 3
    assert len(state["pose"]["joints"]) == 16
    assert len(state["pose"]["legs"]) == 8 and len(state["pose"]["legs"][0]) == 6
    assert len(state["pose"]["planted"]) == 8
    assert state["policy"] == "random"
    described = await (await client.get("/api/describe")).json()
    assert described["inputs"] == len(described["input_names"]) == 152
    assert described["brain"] is None and described["control"]["policy"] == "random"


async def test_controls(client):
    ok = await client.post("/api/control", json={"op": "pause"})
    assert ok.status == 200 and (await ok.json())["ok"] is True
    state = await (await client.get("/api/state")).json()
    assert state["paused"] is True
    pace = await client.post("/api/control", json={"op": "pace", "value": 5})
    assert pace.status == 200
    bad = await client.post("/api/control", json={"op": "fly"})
    assert bad.status == 400
    worse = await client.post("/api/control", data="not json", headers={"Content-Type": "application/json"})
    assert worse.status == 400


async def test_websocket_frame(client):
    async with client.ws_connect("/ws") as ws:
        message = await ws.receive_str()
    frame = json.loads(message)
    assert len(frame["pose"]["joints"]) == 16
    assert frame["moments"] == 3


async def test_brainview_route(tmp_path):
    none = Life("random", seed=1, pace=0.0, runs=tmp_path)
    async with TestClient(TestServer(make_app(none))) as client:
        assert await (await client.get("/api/brainview")).json() is None
    life = Life("brain", seed=1, pace=0.0, runs=tmp_path, backend="cpu")
    life.tick()
    async with TestClient(TestServer(make_app(life))) as client:
        payload = await (await client.get("/api/brainview")).json()
        assert payload["format"] == "cadence.zebrafish-skeletons/v1"
        assert len(payload["cells"]) == payload["counts"]["neurons"] == len(payload["order"])
        state = await (await client.get("/api/state")).json()
        assert len(state["activity"]["all"]) == payload["counts"]["neurons"]
