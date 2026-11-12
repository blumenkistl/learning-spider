"""The web face of one spider's life: aiohttp serves the page, pushes the latest moment over a
WebSocket at the life's pace (at most 30 frames a second) and takes controls as JSON.

    python server.py                    # http://127.0.0.1:8080
    python server.py --high-water --stance-pay 0.1
    PORT=8090 POLICY=random python -m spider

The flags are the nursery's (``python server.py --help``). Each has an environment variable
of the same meaning that sets its default: HOST (127.0.0.1), PORT (8080), POLICY (brain),
SEED (0), PACE (20 moments/s), BACKEND (auto | cpu | torch | torch:<device>), RUNS (runs),
AUTOSAVE_MINUTES (5), STANCE_PAY (0), YAW_PAY (0), PROGRESS_WINDOW (1), HIGH_WATER (0), PAY_PER_METRE (40)
and BACKWARD_PAY (0), the reward's settings, and MOTORS (rate) for a newborn's body. A saved life in RUNS is continued, with its own genes; shutdown
saves it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from aiohttp import web

from .body import MOTORS
from .life import BACKWARD_PAY, HIGH_WATER, PROGRESS_PAY, PROGRESS_WINDOW, STANCE_PAY, YAW_PAY, Life
from .nursery import genes_from, parse_gene
from .senses import Senses

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


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() not in ("", "0", "false", "no")


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """The server's flags, the nursery's names; the environment sets each one's default."""
    env = os.environ.get
    parser = argparse.ArgumentParser(description="the learning spider's web server")
    parser.add_argument("--host", default=env("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(env("PORT", "8080")))
    parser.add_argument("--policy", choices=("brain", "random", "frozen"), default=env("POLICY", "brain"))
    parser.add_argument("--seed", type=int, default=int(env("SEED", "0")))
    parser.add_argument("--pace", type=float, default=float(env("PACE", "20")),
                        help="moments a second (default 20; 0 = as fast as possible)")
    parser.add_argument("--backend", default=env("BACKEND") or None, help="auto (default), cpu, torch or torch:<device>")
    parser.add_argument("--runs", type=Path, default=Path(env("RUNS", "runs")), help="where the life is saved and continued from")
    parser.add_argument("--autosave-minutes", type=float, default=float(env("AUTOSAVE_MINUTES", "5")),
                        help="minutes between autosaves (default 5; 0 = off)")
    parser.add_argument("--gene", type=parse_gene, action="append", default=[],
                        help="name=value for a newborn brain, e.g. modules=48,24 (repeatable; a saved brain keeps its own)")
    parser.add_argument("--per", type=int, default=None, help="tuned inputs per reading (default 8)")
    parser.add_argument("--stance-pay", type=float, default=float(env("STANCE_PAY", STANCE_PAY)),
                        help=f"reward taken per foot short of half the feet planted (default {STANCE_PAY}; 0 = off)")
    parser.add_argument("--yaw-pay", type=float, default=float(env("YAW_PAY", YAW_PAY)),
                        help=f"reward taken per radian turned in a moment (default {YAW_PAY}; 0 = off)")
    parser.add_argument("--window", type=int, default=int(env("PROGRESS_WINDOW", PROGRESS_WINDOW)),
                        help=f"moments the progress term is averaged over (default {PROGRESS_WINDOW})")
    parser.add_argument("--high-water", action="store_true", default=_flag("HIGH_WATER", HIGH_WATER),
                        help="pay new ground only: progress beyond the farthest point reached (default off)")
    parser.add_argument("--pay-per-metre", type=float, default=float(env("PAY_PER_METRE", PROGRESS_PAY)),
                        help=f"reward per metre of progress, under the cap of 0.5 a moment (default {PROGRESS_PAY})")
    parser.add_argument("--backward-pay", type=float, default=float(env("BACKWARD_PAY", BACKWARD_PAY)),
                        help=f"share of the pay per metre charged for backward progress, 0 to 1 (default {BACKWARD_PAY}; 0 = free)")
    parser.add_argument("--motors", choices=MOTORS, default=env("MOTORS", "rate"),
                        help="the body's motor model for a newborn: rate (the founder) or target, position servos")
    return parser.parse_args(argv)


def life_from(args: argparse.Namespace) -> Life:
    """The life the flags describe; a saved life under ``--runs`` is continued."""
    life = Life(
        args.policy,
        seed=args.seed,
        pace=args.pace,
        runs=args.runs,
        genes=genes_from(args.gene) or None,
        backend=args.backend,
        senses=Senses(per=args.per) if args.per else Senses(),
        stance_pay=args.stance_pay,
        yaw_pay=args.yaw_pay,
        window=args.window,
        high_water=args.high_water,
        pay_per_metre=args.pay_per_metre,
        motors=args.motors,
        backward_pay=args.backward_pay,
    )
    life.autosave_minutes = float(args.autosave_minutes)
    if args.policy == "brain" and life.brain_file.exists():
        try:
            report = life.load()
        except ValueError as e:
            # a saved brain for other senses: better no server than a life that refuses every moment
            raise SystemExit(f"cannot continue the saved life: {e}") from e
        log.info("continued %s: moment %s, custody %s", life.brain_file, life.moments, report.get("custody"))
    return life


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = arguments(argv)
    life = life_from(args)
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
    log.info("spider on http://%s:%d policy=%s reward=%s", args.host, args.port, life.policy.policy,
             life.describe()["reward"])
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
