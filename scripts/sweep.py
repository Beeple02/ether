"""Headless batch experiment runner for the relay-swarm simulation.

Usage examples
--------------
# 20 seeded runs with default settings, output to sweep_results/
python scripts/sweep.py --runs 20

# Sweep mesh on vs. off, 10 runs each
python scripts/sweep.py --runs 10 --sweep mesh

# Sweep all three loadout presets, 15 runs each
python scripts/sweep.py --runs 15 --sweep loadout

# Full sweep (mesh × loadout × weight variants), 5 runs per cell
python scripts/sweep.py --runs 5 --sweep all --out results/

# Single reproducible run (seed 42, assault loadout, no mesh)
python scripts/sweep.py --runs 1 --seed-start 42 --preset 1 --no-mesh
"""
import argparse
import csv
import json
import os
import sys
import time

# Allow running from repo root or from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ── simulation imports (no pygame) ───────────────────────────────────────────
from swarm_sim import config
from swarm_sim.swarm import Swarm
from swarm_sim.environment import Environment


# ── constants ────────────────────────────────────────────────────────────────
MAX_TICKS  = 60 * 60 * 2    # 2 sim-minutes at 60 fps (overridable via --max-ticks)
_PRESETS   = config.LOADOUT_PRESET_NAMES


# ── single headless run ───────────────────────────────────────────────────────
def run_once(seed, loadout=None, weights=None, mesh=None, phases=None,
             swarm_size=None, env=None, max_ticks=None):
    """Run one deterministic simulation and return the RunStats object."""
    np.random.seed(seed)

    # Apply overrides to live config (config module is a singleton)
    if loadout      is not None:  config.LOADOUT             = dict(loadout)
    if weights      is not None:  config.WEIGHTS             = dict(weights)
    if mesh         is not None:  config.MESH_SIGNAL_ENABLED = bool(mesh)
    if phases       is not None:  config.MISSION_PHASES      = list(phases)
    if swarm_size   is not None:  config.NUM_DRONES          = int(swarm_size)

    if env is None:
        env = Environment()

    swarm = Swarm()
    swarm.set_environment(env)
    _max = max_ticks if max_ticks is not None else MAX_TICKS

    for _ in range(_max):
        swarm.tick()
        if swarm.stats.ended:
            break

    if not swarm.stats.ended:
        swarm.end_run_manual()

    return swarm.stats


# ── result → dict ─────────────────────────────────────────────────────────────
def stats_to_dict(stats, seed, label, params):
    return {
        "seed":                     seed,
        "label":                    label,
        **params,
        "end_reason":               stats.end_reason,
        "time_elapsed":             round(stats.time_elapsed, 3),
        "relay_survived":           stats.relay_survived,
        "target_reached":           stats.target_reached,
        "drones_survived":          stats.drones_survived,
        "drones_total":             stats.drones_total,
        "attrition_pct":            round(1 - stats.drones_survived / max(stats.drones_total, 1), 3),
        "phase_reached":            stats.phase_reached,
        "relay_successions":        stats.relay_successions,
        "peak_signal_coverage":     round(stats.peak_signal_coverage, 3),
        "intercepted_projectiles":  stats.intercepted_projectiles,
        "intercepted_enemy_drones": stats.intercepted_enemy_drones,
        "emp_turrets_disabled":     stats.emp_turrets_disabled,
        "smokescreen_deployments":  stats.smokescreen_deployments,
        "convergence_used":         stats.convergence_used,
        "enemy_swarm_eliminated":   stats.enemy_swarm_eliminated,
    }


# ── scoring function (doctrine-weighted) ─────────────────────────────────────
def doctrine_score(row, weights=None):
    """Higher = better doctrine adherence."""
    w = weights or {
        "relay_survived":           3.0,
        "target_reached":           2.0,
        "attrition_pct":           -1.0,   # fraction dead — lower is better
        "peak_signal_coverage":     1.0,
        "relay_successions":       -0.5,   # succession is costly
        "intercepted_projectiles":  0.3,
        "intercepted_enemy_drones": 0.4,
    }
    score = 0.0
    score += w["relay_survived"]          * int(row["relay_survived"])
    score += w["target_reached"]          * int(row["target_reached"])
    score += w["attrition_pct"]           * float(row["attrition_pct"])
    score += w["peak_signal_coverage"]    * float(row["peak_signal_coverage"])
    score += w["relay_successions"]       * int(row["relay_successions"])
    score += w["intercepted_projectiles"] * int(row["intercepted_projectiles"])
    score += w["intercepted_enemy_drones"]* int(row["intercepted_enemy_drones"])
    return round(score, 4)


# ── sweep cell builder ────────────────────────────────────────────────────────
def build_cells(sweep_mode, preset_idx):
    """Return list of (label, params_dict) tuples defining the sweep space."""
    cells = []

    base_loadout = dict(config.LOADOUT_PRESETS[preset_idx])
    base_weights = dict(config.WEIGHTS)

    if sweep_mode in ("mesh", "all"):
        for mesh_val in (True, False):
            tag = f"mesh={'on' if mesh_val else 'off'}"
            cells.append((tag, {"mesh": mesh_val,
                                 "loadout": base_loadout,
                                 "weights": base_weights}))
    elif sweep_mode in ("loadout", "all"):
        pass  # handled below

    if sweep_mode in ("loadout", "all"):
        for pi, pname in enumerate(_PRESETS):
            ld = dict(config.LOADOUT_PRESETS[pi])
            cells.append((f"preset={pname}", {"mesh": config.MESH_SIGNAL_ENABLED,
                                               "loadout": ld,
                                               "weights": base_weights}))

    if sweep_mode == "weights":
        # Sweep separation weight ±0.5 in 0.25 steps
        for sep in [1.0, 1.5, 2.0, 2.5, 3.0]:
            w = dict(base_weights)
            w["separation"] = sep
            cells.append((f"sep={sep}", {"mesh": config.MESH_SIGNAL_ENABLED,
                                          "loadout": base_loadout,
                                          "weights": w}))

    if not cells:
        # No sweep — single cell with current defaults
        cells.append(("default", {"mesh": config.MESH_SIGNAL_ENABLED,
                                   "loadout": base_loadout,
                                   "weights": base_weights}))

    return cells


# ── aggregate summary ─────────────────────────────────────────────────────────
def summarize(rows, label):
    if not rows:
        return {}
    relay_surv  = [int(r["relay_survived"]) for r in rows]
    tgt_reached = [int(r["target_reached"])  for r in rows]
    times       = [float(r["time_elapsed"])  for r in rows]
    scores      = [doctrine_score(r) for r in rows]
    return {
        "label":                  label,
        "n":                      len(rows),
        "relay_survival_rate":    round(sum(relay_surv)  / len(rows), 3),
        "target_reach_rate":      round(sum(tgt_reached) / len(rows), 3),
        "mean_time_elapsed":      round(sum(times) / len(rows), 2),
        "mean_doctrine_score":    round(sum(scores) / len(scores), 4),
        "min_doctrine_score":     round(min(scores), 4),
        "max_doctrine_score":     round(max(scores), 4),
        "mean_signal_coverage":   round(sum(float(r["peak_signal_coverage"]) for r in rows) / len(rows), 3),
        "mean_attrition_pct":     round(sum(float(r["attrition_pct"]) for r in rows) / len(rows), 3),
        "total_relay_successions":sum(int(r["relay_successions"]) for r in rows),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Headless swarm experiment runner")
    ap.add_argument("--runs",        type=int,   default=10,
                    help="Number of seeded runs per sweep cell (default: 10)")
    ap.add_argument("--seed-start",  type=int,   default=0,
                    help="First RNG seed; subsequent runs use seed+1, seed+2, … (default: 0)")
    ap.add_argument("--preset",      type=int,   default=0,
                    choices=range(len(_PRESETS)),
                    help=f"Loadout preset index 0={_PRESETS[0]} 1={_PRESETS[1]} "
                         f"2={_PRESETS[2]} (default: 0)")
    ap.add_argument("--sweep",       default="none",
                    choices=["none", "mesh", "loadout", "weights", "all"],
                    help="Parameter dimension(s) to sweep (default: none)")
    ap.add_argument("--no-mesh",     action="store_true",
                    help="Force MESH_SIGNAL_ENABLED=False for this run")
    ap.add_argument("--swarm-size",  type=int,   default=None,
                    help="Override NUM_DRONES")
    ap.add_argument("--max-ticks",   type=int,   default=MAX_TICKS,
                    help=f"Max ticks per run (default: {MAX_TICKS} = 2 sim-min)")
    ap.add_argument("--out",         default="sweep_results",
                    help="Output directory (default: sweep_results/)")
    args = ap.parse_args()

    if args.no_mesh:
        config.MESH_SIGNAL_ENABLED = False

    cells = build_cells(args.sweep if args.sweep != "none" else "", args.preset)
    os.makedirs(args.out, exist_ok=True)

    all_rows     = []
    summaries    = []
    total_runs   = len(cells) * args.runs
    run_num      = 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(args.out, f"sweep_{ts}.csv")
    json_path = os.path.join(args.out, f"sweep_{ts}_summary.json")

    print(f"Sweep: {len(cells)} cell(s) × {args.runs} run(s) = {total_runs} total")
    print(f"Output: {csv_path}\n")

    csv_file   = open(csv_path, "w", newline="")
    csv_writer = None

    for label, params in cells:
        cell_rows = []
        for i in range(args.runs):
            seed = args.seed_start + run_num
            run_num += 1

            t0 = time.time()
            stats = run_once(
                seed        = seed,
                loadout     = params.get("loadout"),
                weights     = params.get("weights"),
                mesh        = params.get("mesh"),
                swarm_size  = args.swarm_size,
                max_ticks   = args.max_ticks,
            )
            elapsed = time.time() - t0

            row = stats_to_dict(stats, seed, label, {
                k: v for k, v in params.items()
                if k not in ("loadout", "weights")
            })
            row["wall_time_s"] = round(elapsed, 3)

            if csv_writer is None:
                csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                csv_writer.writeheader()
            csv_writer.writerow(row)
            csv_file.flush()

            cell_rows.append(row)
            all_rows.append(row)

            score = doctrine_score(row)
            relay_sym = "✓" if row["relay_survived"] else "✗"
            tgt_sym   = "✓" if row["target_reached"] else "✗"
            print(f"  [{label}] seed={seed:4d}  relay={relay_sym}  target={tgt_sym}"
                  f"  score={score:+.2f}  {stats.end_reason:<14}  "
                  f"sig={row['peak_signal_coverage']:.2f}  wall={elapsed:.1f}s")

        cell_sum = summarize(cell_rows, label)
        summaries.append(cell_sum)
        print(f"\n  ── {label}: relay_surv={cell_sum['relay_survival_rate']:.0%}  "
              f"tgt_reach={cell_sum['target_reach_rate']:.0%}  "
              f"score_avg={cell_sum['mean_doctrine_score']:+.2f}\n")

    csv_file.close()

    # JSON summary
    total_sum = summarize(all_rows, "TOTAL")
    output = {
        "timestamp":  ts,
        "args":       vars(args),
        "cells":      summaries,
        "total":      total_sum,
        "ranking":    sorted(summaries,
                             key=lambda s: s["mean_doctrine_score"],
                             reverse=True),
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSummary written → {json_path}")
    print(f"\nRanking by doctrine score:")
    for rank, s in enumerate(output["ranking"], 1):
        print(f"  #{rank}  {s['label']:<30}  score={s['mean_doctrine_score']:+.4f}"
              f"  relay={s['relay_survival_rate']:.0%}"
              f"  tgt={s['target_reach_rate']:.0%}")


if __name__ == "__main__":
    main()
