"""Headless measurement: run one policy for a number of moments and print a line per block.

    python -m spider.nursery --moments 3000 --policy brain --seed 0
    python -m spider.nursery --moments 3000 --policy random --seed 0
    python -m spider.nursery --moments 3000 --policy frozen --brain runs/spider.npz
    python -m spider.nursery --moments 600 --policy brain --backend torch
    python -m spider.nursery --moments 3000 --policy brain --gene modules=48,24 --gene arousal.need=0

The world's reward is the only teacher. Random is the baseline; frozen (a newborn, or a saved
brain that answers greedily and never learns) is the control for "kept learning". ``--brain``
with ``--policy brain`` continues a saved life, with its world and pending outcome when the
sidecar is beside it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .body import DEFAULT_MOTORS, MOTORS
from .brain import DEFAULT_LAYOUT, LAYOUTS
from .brain import genes_from, parse_gene  # noqa: F401  (the server imports them from here too)
from .life import (BACKWARD_PAY, BLOCK, COURSE, HIGH_WATER, OLD_GROUND, PROGRESS_PAY, PROGRESS_WINDOW, STANCE_PAY, YAW_PAY,
                   Life, checked_course)
from .senses import PER, Senses


HEAD = (f"{'moment':>7} {'forward_m':>9} {'r/moment':>8} {'aroused':>7} {'sweeps':>6} "
        f"{'learn':>6} {'refused':>7} {'ms':>6} {'stance':>6} {'turned':>7} {'course_m':>8}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    senses = Senses(per=args.per or PER, setpoints=args.setpoints, compass=args.compass, ground=args.ground)
    genes = genes_from(args.gene) or None
    runs = args.save.parent if args.save else Path("runs")
    name = args.save.stem if args.save else "spider"
    continuing = bool(args.brain and args.policy == "brain" and args.brain.with_suffix(".json").exists())
    life = Life(args.policy, seed=args.seed, pace=0.0, runs=args.brain.parent if continuing else runs,
                genes=genes, backend=args.backend, senses=senses,
                brain_path=None if continuing else args.brain,
                name=args.brain.stem if continuing else name, stance_pay=args.stance_pay,
                yaw_pay=args.yaw_pay, window=args.window, high_water=args.high_water, old_ground=args.old_ground,
                pay_per_metre=args.pay_per_metre, motors=args.motors, backward_pay=args.backward_pay,
                course=args.course, layout=args.layout)
    life.autosave_minutes = 0.0
    if continuing:
        life.load()
        life.runs, life.name = runs, name
    started = life.moments
    block = max(1, int(args.block))
    began = time.perf_counter()
    rows: list[dict[str, Any]] = []
    window: list[dict[str, Any]] = []
    mark = life.distance_forward
    course_mark = life.distance_course
    turned_mark = life.turned
    if not args.quiet:
        print(HEAD, flush=True)
    for _ in range(args.moments):
        window.append(life.tick())
        if len(window) == block:
            row = {
                "moment": life.moments,
                "forward_m": round(life.distance_forward - mark, 4),
                "reward_per_moment": round(sum(w["reward"] or 0.0 for w in window) / block, 4),
                "aroused_share": round(sum(1 for w in window if w["mode"] == "aroused") / block, 3),
                "sweeps_per_moment": round(sum(w["sweeps"] for w in window) / block, 2),
                "learning_sweeps": int(sum(w["learning_sweeps"] for w in window)),
                "refused": int(sum(1 for w in window if w["refused"])),
                "ms_per_moment": round(sum(w["ms"] for w in window) / block, 2),
                "mean_stance": round(sum(sum(w["pose"]["planted"]) for w in window) / block, 2),
                "turned_rad": round(life.turned - turned_mark, 3),
                "course_m": round(life.distance_course - course_mark, 4),
            }
            rows.append(row)
            mark = life.distance_forward
            course_mark = life.distance_course
            turned_mark = life.turned
            window = []
            if not args.quiet:
                print(f"{row['moment']:>7} {row['forward_m']:>9.3f} {row['reward_per_moment']:>8.4f} "
                      f"{row['aroused_share']:>7.2f} {row['sweeps_per_moment']:>6.1f} {row['learning_sweeps']:>6} "
                      f"{row['refused']:>7} {row['ms_per_moment']:>6.2f} {row['mean_stance']:>6.2f} "
                      f"{row['turned_rad']:>7.2f} {row['course_m']:>8.3f}", flush=True)
    elapsed = time.perf_counter() - began
    half = len(rows) // 2
    summary: dict[str, Any] = {
        "policy": args.policy,
        "seed": args.seed,
        "moments": args.moments,
        "from_moment": started,
        "forward_m": round(sum(r["forward_m"] for r in rows), 4),
        "course_m": round(sum(r["course_m"] for r in rows), 4),
        "first_half_m": round(sum(r["forward_m"] for r in rows[:half]), 4),
        "second_half_m": round(sum(r["forward_m"] for r in rows[half:]), 4),
        "refused": int(sum(r["refused"] for r in rows)),
        "turned_rad": round(sum(r["turned_rad"] for r in rows), 3),
        "ms_per_moment": round(1000.0 * elapsed / max(1, args.moments), 3),
        "seconds": round(elapsed, 1),
        "placement": life.brain.describe()["placement"] if life.brain is not None else None,
        "genes": genes or {},
        **life.law(),                       # a continued life's own law, whatever the flags said
        "law_changes": life.law_changes,
        "motors": life.motors,
        "layout": life.layout,
        "per": life.senses.per,
        "setpoints": life.senses.setpoints,
        "compass": life.senses.compass,
        "ground": life.senses.ground,
        "blocks": rows,
    }
    if args.save and life.brain is not None:
        summary["saved"] = life.save()
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--moments", type=int, default=3000)
    parser.add_argument("--policy", choices=("brain", "random", "frozen"), default="brain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, default=BLOCK)
    parser.add_argument("--brain", type=Path, default=None, help="a saved brain to continue, or to freeze")
    parser.add_argument("--save", type=Path, default=None, help="save the brain and its world here after the run (.npz)")
    parser.add_argument("--gene", type=parse_gene, action="append", default=[],
                        help="name=value, e.g. modules=48,24 or arousal.need=0 (repeatable)")
    parser.add_argument("--per", type=int, default=None, help="tuned inputs per reading (default 8)")
    parser.add_argument("--setpoints", action="store_true",
                        help="sense the servos' setpoints too (needs --motors target; sixteen readings more)")
    parser.add_argument("--compass", action="store_true",
                        help="sense the heading too, its cosine and sine (two readings more): the sense the course law is read by")
    parser.add_argument("--ground", action="store_true",
                        help="sense the velocity over the ground in the world's frame too (two readings more): what the course law pays, read from the world")
    parser.add_argument("--stance-pay", type=float, default=STANCE_PAY,
                        help=f"reward taken per foot short of half the feet planted (default {STANCE_PAY}; 0 = off)")
    parser.add_argument("--yaw-pay", type=float, default=YAW_PAY,
                        help=f"reward taken per radian turned in a moment (default {YAW_PAY}; 0 = off)")
    parser.add_argument("--course", type=checked_course, default=COURSE,
                        help="pay progress along this fixed heading of the world (radians; 'birth' or 0 is the heading at "
                             "birth) instead of along the body's own heading (default off)")
    parser.add_argument("--window", type=int, default=PROGRESS_WINDOW,
                        help=f"moments the progress term is averaged over (default {PROGRESS_WINDOW})")
    parser.add_argument("--high-water", action="store_true", default=HIGH_WATER,
                        help="pay new ground only: progress beyond the farthest point reached (default off)")
    parser.add_argument("--old-ground", type=float, default=OLD_GROUND,
                        help=f"under --high-water, the share paid for progress on old ground, 0 to 1 (default {OLD_GROUND})")
    parser.add_argument("--pay-per-metre", type=float, default=PROGRESS_PAY,
                        help=f"reward per metre of progress, under the cap of 0.5 a moment (default {PROGRESS_PAY})")
    parser.add_argument("--backward-pay", type=float, default=BACKWARD_PAY,
                        help=f"share of the pay per metre charged for backward progress, 0 to 1 (default {BACKWARD_PAY}; 0 = free)")
    parser.add_argument("--layout", choices=LAYOUTS, default=DEFAULT_LAYOUT,
                        help=f"how a newborn brain's answer is read: a slot per joint (three answers) or per leg "
                             f"(nine, the pairs of its two joints' answers); a saved brain keeps its own (default {DEFAULT_LAYOUT})")
    parser.add_argument("--motors", choices=MOTORS, default=DEFAULT_MOTORS,
                        help=f"the body's motor model: rate (the founder until 2026-11-12, a command moves the joint for one "
                             f"moment) or target, position servos (default {DEFAULT_MOTORS})")
    parser.add_argument("--backend", default=None, help="auto (default), cpu, torch or torch:<device>")
    parser.add_argument("--json", type=Path, default=None, help="write the summary here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.save and args.json and args.json.resolve() == args.save.with_suffix(".json").resolve():
        parser.error("--json would overwrite the saved brain's sidecar; choose another name")
    summary = run(args)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=1))
    if not args.quiet:
        print(json.dumps({k: v for k, v in summary.items() if k != "blocks"}, indent=1))
    return summary


if __name__ == "__main__":
    main()
    sys.exit(0)
