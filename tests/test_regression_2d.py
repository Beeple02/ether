"""Golden-baseline regression harness for 2D mode.

Run:  python -m pytest tests/test_regression_2d.py -v
  or: python tests/test_regression_2d.py

Guards that any future change — including 3D migration — leaves the 2D
simulation path (SIM_3D=False, RENDER_3D=False) producing the exact same
RunStats field values as when this baseline was recorded.
"""
import json
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from swarm_sim import config

# ── ensure flags are in 2D baseline state ────────────────────────────────────
config.RENDER_3D = False
config.SIM_3D    = False

from swarm_sim.environment import Environment
from swarm_sim.swarm import Swarm

# ── baseline file location ───────────────────────────────────────────────────
BASELINE_PATH = pathlib.Path(__file__).parent / "baseline_2d.json"

# ── run parameters ───────────────────────────────────────────────────────────
SEED      = 7
MAX_TICKS = 60 * 60 * 2   # 2 sim-minutes (same default as sweep.py)

# Fields captured in the golden snapshot
STATS_FIELDS = [
    "end_reason",
    "relay_survived",
    "target_reached",
    "drones_survived",
    "drones_total",
    "phase_reached",
    "relay_successions",
    "intercepted_projectiles",
    "intercepted_enemy_drones",
    "emp_turrets_disabled",
    "smokescreen_deployments",
    "convergence_used",
    "enemy_swarm_eliminated",
]


def _run_sim(seed: int) -> dict:
    np.random.seed(seed)
    config.RENDER_3D = False
    config.SIM_3D    = False

    env   = Environment()
    swarm = Swarm()
    swarm.set_environment(env)

    for _ in range(MAX_TICKS):
        swarm.tick()
        if swarm.stats.ended:
            break

    if not swarm.stats.ended:
        swarm.end_run_manual()

    s = swarm.stats
    return {f: getattr(s, f) for f in STATS_FIELDS}


def record_baseline():
    """Write a fresh golden baseline. Run once manually after a confirmed-good state."""
    result = _run_sim(SEED)
    with open(BASELINE_PATH, "w") as f:
        json.dump({"seed": SEED, "max_ticks": MAX_TICKS, "stats": result}, f, indent=2)
    print(f"Baseline recorded → {BASELINE_PATH}")
    for k, v in result.items():
        print(f"  {k}: {v}")


# ── pytest entry point ───────────────────────────────────────────────────────
def test_2d_baseline_unchanged():
    """Verify that seed-7 2D run produces the exact same RunStats as the golden baseline."""
    if not BASELINE_PATH.exists():
        record_baseline()
        # First run records the baseline; mark as expected pass.
        return

    with open(BASELINE_PATH) as f:
        saved = json.load(f)

    assert saved["seed"]      == SEED,      "Baseline seed mismatch"
    assert saved["max_ticks"] == MAX_TICKS, "Baseline max_ticks mismatch"

    actual   = _run_sim(SEED)
    expected = saved["stats"]

    mismatches = []
    for field in STATS_FIELDS:
        if actual[field] != expected[field]:
            mismatches.append(f"  {field}: expected={expected[field]!r}  actual={actual[field]!r}")

    assert not mismatches, "2D baseline regression failures:\n" + "\n".join(mismatches)


# ── standalone entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true",
                    help="Record a new golden baseline (overwrites existing)")
    args = ap.parse_args()

    if args.record or not BASELINE_PATH.exists():
        record_baseline()
    else:
        test_2d_baseline_unchanged()
        print("OK — 2D baseline unchanged")
