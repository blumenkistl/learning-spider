"""The web face of one spider's life: aiohttp serves the page, pushes the latest moment over a
WebSocket at the life's pace (at most 30 frames a second) and takes controls as JSON.

    python server.py                    # http://127.0.0.1:8080
    PORT=8090 POLICY=random python -m spider

Environment: HOST (127.0.0.1), PORT (8080), POLICY (brain), SEED (0), PACE (20 moments/s),
BACKEND (auto | cpu | torch | torch:<device>), RUNS (runs), AUTOSAVE_MINUTES (5),
STANCE_PAY (0), YAW_PAY (0) and PROGRESS_WINDOW (1), the reward's settings.
A saved life in RUNS is continued; shutdown saves it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from aiohttp import web

from .life import PROGRESS_WINDOW, STANCE_PAY, YAW_PAY, Life

STATIC = Path(__file__).resolve().parent.parent / "static"
FRAME_SECONDS = 1.0 / 30.0
log = logging.getLogger("spider")
LIFE = web.AppKey("life", Life)
SOCKETS = web.AppKey("sockets", set)


def make_app(life: Life) -> web.Application:
    app = web.Application()
    app[LIFE] = life
    app[SOCKETS] = set()

    async def index(request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC / "index.html")

    async def state(request: web.Request) -> web.Response:
        return web.json_response(life.latest)

    async def describe(request: web.Request) -> web.Response:
        return web.json_response(life.describe())

    async def brainview(request: web.Request) -> web.Response:
        # the whole connectome laid out for the page's brain view, built once per brain (~1.5 MB)
        loop = asyncio.get_running_loop()
        return web.json_response(await loop.run_in_executor(None, life.spider))

    async def control(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise web.HTTPBadRequest(text="a JSON body {op, value?} is required")
        op = body.get("op") if isinstance(body, dict) else None
        if not isinstance(op, str):
            raise web.HTTPBadRequest(text="op must be a string")
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, life.control, op, body.get("value"))
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error))
        return web.json_response(result)

    async def socket(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        app[SOCKETS].add(ws)
        sender = asyncio.create_task(_push(ws, life))
        try:
            async for _ in ws:            # the page sends nothing; this drains until close
                pass
        finally:
            sender.cancel()
            app[SOCKETS].discard(ws)
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/api/state", state)
    app.router.add_get("/api/describe", describe)
    app.router.add_get("/api/brainview", brainview)
    app.router.add_post("/api/control", control)
    app.router.add_get("/ws", socket)
    app.router.add_static("/static/", STATIC, show_index=False)
    return app


async def _push(ws: web.WebSocketResponse, life: Life) -> None:
    last = None
    while not ws.closed:
        latest = life.latest
        if latest is not last:                       # a new moment, or a control's snapshot
            last = latest
            await ws.send_str(json.dumps(latest))
        await asyncio.sleep(FRAME_SECONDS)


def life_from_env() -> Life:
    runs = Path(os.environ.get("RUNS", "runs"))
    policy = os.environ.get("POLICY", "brain")
    life = Life(
        policy,
        seed=int(os.environ.get("SEED", "0")),
        pace=float(os.environ.get("PACE", "20")),
        runs=runs,
        backend=os.environ.get("BACKEND") or None,
        stance_pay=float(os.environ.get("STANCE_PAY", STANCE_PAY)),
        yaw_pay=float(os.environ.get("YAW_PAY", YAW_PAY)),
        window=int(os.environ.get("PROGRESS_WINDOW", PROGRESS_WINDOW)),
    )
    if policy == "brain" and life.brain_file.exists():
        report = life.load()
        log.info("continued %s: moment %s, custody %s", life.brain_file, life.moments, report.get("custody"))
    return life


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    life = life_from_env()
    app = make_app(life)

    async def on_startup(app: web.Application) -> None:
        life.start()

    async def on_shutdown(app: web.Application) -> None:
        life.stop()
        for ws in set(app[SOCKETS]):
            await ws.close()
        if life.brain is not None and life.policy is life.brain:
            log.info("saved %s", life.save())

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    log.info("spider on http://%s:%d policy=%s", host, port, life.policy.policy)
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
