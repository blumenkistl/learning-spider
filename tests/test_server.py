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
    assert state["displacement_m"] >= 0.0 and "distance_net" not in state
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


def test_flags_mirror_the_nursery_and_the_environment_sets_defaults(tmp_path, monkeypatch):
    from spider.server import arguments, life_from

    monkeypatch.delenv("HIGH_WATER", raising=False)
    args = arguments(["--high-water", "--stance-pay", "0.1", "--window", "20", "--port", "8090",
                      "--policy", "random", "--runs", str(tmp_path), "--autosave-minutes", "0",
                      "--gene", "modules=48,24", "--per", "4"])
    assert args.port == 8090 and args.high_water is True and args.stance_pay == 0.1 and args.window == 20
    life = life_from(args)
    assert life.high_water is True and life.stance_pay == 0.1 and life.window == 20
    assert life.runs == tmp_path and life.autosave_minutes == 0.0 and life.senses.inputs < 152
    assert life.policy.policy == "random"

    monkeypatch.setenv("HIGH_WATER", "1")
    monkeypatch.setenv("STANCE_PAY", "0.1")
    monkeypatch.setenv("PORT", "8091")
    args = arguments([])
    assert args.high_water is True and args.stance_pay == 0.1 and args.port == 8091
    monkeypatch.setenv("HIGH_WATER", "0")
    assert arguments([]).high_water is False and arguments(["--high-water"]).high_water is True
    monkeypatch.setenv("OLD_GROUND", "0.2")
    assert arguments([]).old_ground == 0.2 and arguments(["--old-ground", "0.5"]).old_ground == 0.5
    assert life_from(arguments(["--high-water", "--policy", "random", "--runs", str(tmp_path)])).old_ground == 0.2
    assert arguments([]).layout == "joint" and arguments(["--layout", "leg"]).layout == "leg"
    monkeypatch.setenv("LAYOUT", "leg")
    assert arguments([]).layout == "leg"
    assert life_from(arguments(["--high-water", "--policy", "random", "--runs", str(tmp_path)])).policy.layout == "leg"
    assert arguments([]).drift == 0.0 and arguments(["--drift", "0.002"]).drift == 0.002
    monkeypatch.setenv("DRIFT", "0.005")
    assert arguments([]).drift == 0.005
    assert life_from(arguments(["--high-water", "--policy", "random", "--runs", str(tmp_path)])).body.drift == 0.005


async def test_the_law_and_a_birth_are_controls(client):
    r = await client.post("/api/control", json={"op": "law", "value": {"high_water": True, "stance_pay": 0.1}})
    assert r.status == 200 and (await r.json())["changed"] is True
    described = await (await client.get("/api/describe")).json()
    assert described["reward"]["high_water"] is True and described["reward"]["stance_pay"] == 0.1
    assert len(described["law_changes"]) == 1
    r = await client.post("/api/control", json={"op": "law", "value": {"old_ground": 2}})
    assert r.status == 400
    r = await client.post("/api/control", json={"op": "law", "value": {"drift": 0.002}})
    assert r.status == 200 and (await r.json())["law"]["drift"] == 0.002
    described = await (await client.get("/api/describe")).json()
    assert described["reward"]["drift"] == 0.002 and len(described["law_changes"]) == 2
    r = await client.post("/api/control", json={"op": "law", "value": {"drift": -1}})
    assert r.status == 400
    r = await client.post("/api/control", json={"op": "law", "value": 3})
    assert r.status == 400
    r = await client.post("/api/control", json={"op": "reset", "value": {"seed": 7, "motors": "target", "genes": "gamma=0.99"}})
    assert r.status == 200 and (await r.json())["seed"] == 7
    described = await (await client.get("/api/describe")).json()
    assert described["seed"] == 7 and described["motors"] == "target" and described["genes"] == {"gamma": 0.99}
    assert described["reward"]["high_water"] is True                # the law is the life's, a birth keeps it
    r = await client.post("/api/control", json={"op": "reset", "value": {"setpoints": True, "motors": "rate"}})
    assert r.status == 400
