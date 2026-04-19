import time
import numpy as np
from . import config
from .agent import Agent
from .pathfinder import PathFinder
from .physics import normalize
from .environment import Projectile


class RunStats:
    """Per-run metrics tracker."""
    def __init__(self):
        self.start_time           = time.time()
        self.time_elapsed         = 0.0
        self.drones_total         = 0
        self.drones_survived      = 0
        self.relay_survived       = True
        self.target_reached       = False
        self.peak_signal_coverage = 0.0
        self.ended                = False
        self.end_reason           = ""

    def finalise(self, relay_alive, alive_count, total, target_reached, reason):
        if self.ended:
            return
        self.ended           = True
        self.time_elapsed    = time.time() - self.start_time
        self.drones_total    = total
        self.drones_survived = alive_count
        self.relay_survived  = relay_alive
        self.target_reached  = target_reached
        self.end_reason      = reason


class Swarm:
    def __init__(self):
        self.relay         = None
        self.drones        = []
        self.projectiles   = []
        self._waypoints    = []
        self._wp_index     = 0
        self._environment  = None
        self._pf           = PathFinder()
        self._relay_path   = []
        self._path_idx     = 0
        self._path_target  = None
        self._replan_in    = 0
        self.stats         = RunStats()
        self.last_stats    = None
        self._spawn()

    # ── spawn ─────────────────────────────────────────────────────────────────
    def _spawn(self):
        env = self._environment
        w, h = config.WORLD_SIZE

        base_pos = None
        if env is not None and env.base is not None:
            base_pos = env.base.position

        if base_pos is not None:
            self.relay = Agent(base_pos.copy(), role="relay")
            self.drones = [
                Agent(base_pos + np.random.uniform(-22, 22, 2),
                      role="drone", drone_type=dt)
                for dt in _spawn_types(config.NUM_DRONES, config.LOADOUT)
            ]
        else:
            self.relay = Agent([w / 2, h / 2], role="relay")
            self.drones = [
                Agent(np.random.uniform([0, 0], [w, h]),
                      role="drone", drone_type=dt)
                for dt in _spawn_types(config.NUM_DRONES, config.LOADOUT)
            ]

        self.projectiles  = []
        self._relay_path  = []
        self._path_idx    = 0
        self._path_target = None
        self._replan_in   = 0
        self.stats        = RunStats()
        self.stats.drones_total = config.NUM_DRONES

    def set_environment(self, env):
        self._environment = env
        self._pf.rebuild(env)
        self._relay_path  = []
        self._path_target = None
        self._replan_in   = 0
        self._spawn()

    def set_waypoints(self, waypoints):
        self._waypoints   = list(waypoints)
        self._wp_index    = 0
        self._relay_path  = []
        self._path_target = None

    # ── main tick ─────────────────────────────────────────────────────────────
    def tick(self):
        if self.stats.ended:
            return

        self._update_relay()
        self._update_signals()
        self._update_noflyzone()

        alive = [d for d in self.drones if d.alive]
        shield_forces = self._shield_forces(alive) if self._environment else {}

        for i, drone in enumerate(alive):
            ft = self._formation_target(i, len(alive))
            sf = shield_forces.get(id(drone))
            drone.tick(self._neighbors(drone, alive), ft,
                       self._environment, shield_force=sf)

        self._update_threats()
        self._check_end_conditions()

    # ── signal ────────────────────────────────────────────────────────────────
    def _update_signals(self):
        env = self._environment
        relay_pos = self.relay.position
        R = config.SIGNAL_RADIUS
        R2 = R * 2

        strong = 0
        alive_count = 0
        for d in self.drones:
            if not d.alive:
                continue
            alive_count += 1
            dist = float(np.linalg.norm(d.position - relay_pos))
            if dist <= R:
                s = 1.0
            elif dist >= R2:
                s = 0.0
            else:
                s = 1.0 - (dist - R) / R
            if env is not None and not env.has_line_of_sight(d.position, relay_pos):
                s *= config.SIGNAL_LOS_PEN
            d.signal = float(np.clip(s, 0.0, 1.0))
            if d.signal > config.SIGNAL_TIER_HIGH:
                strong += 1

        if alive_count > 0:
            coverage = strong / alive_count
            if coverage > self.stats.peak_signal_coverage:
                self.stats.peak_signal_coverage = coverage

    # ── NOFLYZONE kills ───────────────────────────────────────────────────────
    def _update_noflyzone(self):
        env = self._environment
        if not env or not config.NOFLYZONE_KILLS:
            return
        nfz = [z for z in env.zones if z.zone_type == "NOFLYZONE"]
        if not nfz:
            return
        for d in self.drones:
            if d.alive and any(z.contains(d.position) for z in nfz):
                d.alive = False

    # ── formation target ──────────────────────────────────────────────────────
    def _formation_target(self, drone_idx, n_alive):
        mode      = config.FORMATION_MODE
        relay_pos = self.relay.position

        if mode == "ring":
            angle = 2 * np.pi * drone_idx / max(n_alive, 1)
            r     = config.FORMATION_RING_RADIUS
            return relay_pos + np.array([np.cos(angle), np.sin(angle)]) * r

        if mode == "V":
            rv    = self.relay.velocity
            speed = float(np.linalg.norm(rv))
            fwd   = rv / speed if speed > 0.1 else np.array([0.0, -1.0])
            perp  = np.array([-fwd[1], fwd[0]])
            row   = drone_idx // 2 + 1
            side  = 1 if drone_idx % 2 == 0 else -1
            return relay_pos - fwd * row * 25 + perp * (side * row * 20)

        # flock — existing path-following target
        return self._drone_follow_target()

    # ── shield intercept ──────────────────────────────────────────────────────
    def _shield_forces(self, alive):
        env = self._environment
        out = {}
        if env is None or not env.turrets:
            return out
        relay_pos = self.relay.position
        for d in alive:
            if d.drone_type != "shield":
                continue
            nearest = None
            nd = 1e18
            for t in env.turrets:
                dd = float(np.linalg.norm(t.position - relay_pos))
                if dd < nd:
                    nd = dd
                    nearest = t
            if nearest is None:
                continue
            slot = relay_pos + (nearest.position - relay_pos) * 0.45
            out[id(d)] = normalize(slot - d.position)
        return out

    # ── relay navigation ──────────────────────────────────────────────────────
    def _update_relay(self):
        env = self._environment
        if not self.relay.alive:
            return

        raw_goal = self._relay_goal()
        if raw_goal is None:
            self.relay._move_relay(env)
            self._relay_path = []
            return

        self._replan_in -= 1
        goal_shifted = (
            self._path_target is None or
            np.linalg.norm(raw_goal - self._path_target) > 50
        )
        if goal_shifted or self._replan_in <= 0:
            self._relay_path  = self._pf.find_path(self.relay.position, raw_goal)
            self._path_idx    = 0
            self._path_target = raw_goal.copy()
            self._replan_in   = 90

        path = self._relay_path
        while self._path_idx < len(path) - 1:
            if np.linalg.norm(path[self._path_idx] - self.relay.position) < 20:
                self._path_idx += 1
            else:
                break

        step = path[self._path_idx] if path else raw_goal
        self.relay.move_relay_toward(step, env)

        if self._waypoints and self._path_idx >= len(path) - 1:
            if np.linalg.norm(raw_goal - self.relay.position) < 15:
                self._wp_index    = (self._wp_index + 1) % len(self._waypoints)
                self._relay_path  = []
                self._path_target = None

    def _relay_goal(self):
        if self._waypoints:
            return np.array(self._waypoints[self._wp_index], dtype=float)

        env = self._environment
        if env:
            targets = [z for z in env.zones if z.zone_type == "TARGET"]
            if targets:
                centers = [np.array([z.rect[0] + z.rect[2] / 2,
                                     z.rect[1] + z.rect[3] / 2])
                           for z in targets]
                com = np.mean(centers, axis=0)
                self.relay._t += 0.008
                return com + np.array([
                    np.cos(self.relay._t) * 70,
                    np.sin(self.relay._t * 0.7) * 35,
                ])
        return None

    # ── drone follow target ───────────────────────────────────────────────────
    def _drone_follow_target(self):
        path = self._relay_path
        if not path or self._path_idx >= len(path):
            return self.relay.position.copy()

        LOOKAHEAD = 80.0
        pos       = self.relay.position.copy()
        remaining = LOOKAHEAD

        for i in range(self._path_idx, len(path)):
            wp   = path[i]
            diff = wp - pos
            d    = np.linalg.norm(diff)
            if d >= remaining:
                return pos + normalize(diff) * remaining
            remaining -= d
            pos = wp.copy()

        return path[-1].copy()

    # ── threats ───────────────────────────────────────────────────────────────
    def _update_threats(self):
        env = self._environment
        dt  = 1.0 / config.FPS
        speed_per_frame = config.PROJECTILE_SPEED / config.FPS
        W, H = config.WORLD_SIZE

        if env and env.turrets:
            candidates = [d for d in self.drones if d.alive]
            if self.relay.alive:
                candidates.append(self.relay)
            for turret in env.turrets:
                turret._cooldown -= dt
                if turret._cooldown > 0:
                    continue
                nearest = None
                nd = turret.range
                for t in candidates:
                    d = float(np.linalg.norm(t.position - turret.position))
                    if d < nd:
                        nd = d
                        nearest = t
                if nearest is not None:
                    direction = normalize(nearest.position - turret.position)
                    vel = direction * speed_per_frame
                    self.projectiles.append(Projectile(turret.position.copy(), vel))
                    turret._cooldown = 1.0 / max(turret.fire_rate, 1e-3)

        for p in self.projectiles:
            if not p.alive:
                continue
            p.position = p.position + p.velocity
            if (p.position[0] < 0 or p.position[0] > W or
                p.position[1] < 0 or p.position[1] > H):
                p.alive = False
                continue
            hit = False
            for d in self.drones:
                if d.alive and np.linalg.norm(p.position - d.position) < config.DRONE_HIT_RADIUS:
                    d.alive = False
                    p.alive = False
                    hit = True
                    break
            if not hit and self.relay.alive:
                if np.linalg.norm(p.position - self.relay.position) < config.RELAY_HIT_RADIUS:
                    self.relay.alive = False
                    p.alive = False

        self.projectiles = [p for p in self.projectiles if p.alive]

    # ── end conditions ────────────────────────────────────────────────────────
    def _check_end_conditions(self):
        if self.stats.ended:
            return
        env = self._environment

        if not self.relay.alive:
            self._finalise("relay_dead")
            return

        if env:
            targets = [z for z in env.zones if z.zone_type == "TARGET"]
            if targets:
                alive = [d for d in self.drones if d.alive]
                if alive:
                    inside = sum(1 for d in alive
                                 if any(t.contains(d.position) for t in targets))
                    if inside / len(alive) >= config.TARGET_REACH_FRACTION:
                        self._finalise("target_reached")

    def _finalise(self, reason):
        alive = [d for d in self.drones if d.alive]
        target_reached = (reason == "target_reached")
        self.stats.finalise(
            relay_alive=self.relay.alive,
            alive_count=len(alive),
            total=len(self.drones),
            target_reached=target_reached,
            reason=reason,
        )
        self.last_stats = self.stats
        _append_csv_log(self.stats)

    def end_run_manual(self):
        if not self.stats.ended:
            self._finalise("manual")

    # ── utils ─────────────────────────────────────────────────────────────────
    def _neighbors(self, agent, alive):
        perc_mult = config.DRONE_TYPES.get(agent.drone_type or "fast", {}).get("perception_mult", 1.0)
        r2 = (config.PERCEPTION_RADIUS * perc_mult) ** 2
        return [
            o for o in alive
            if o is not agent and np.sum((o.position - agent.position) ** 2) < r2
        ]

    def kill_random(self):
        alive = [d for d in self.drones if d.alive]
        if alive:
            np.random.choice(alive).alive = False

    def respawn(self, n):
        config.NUM_DRONES = n
        self._spawn()

    @property
    def alive_count(self):
        return sum(1 for d in self.drones if d.alive)

    @property
    def run_ended(self):
        return self.stats.ended


# ── helpers ──────────────────────────────────────────────────────────────────
def _spawn_types(count, loadout):
    types = []
    for dt, frac in loadout.items():
        types += [dt] * int(round(count * frac))
    while len(types) < count:
        types.append("fast")
    types = types[:count]
    np.random.shuffle(types)
    return types


def _append_csv_log(stats):
    import csv
    import os
    path = config.CSV_LOG_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "time_elapsed", "drones_survived", "drones_total",
                "relay_survived", "target_reached", "peak_signal_coverage",
                "end_reason",
            ])
        w.writerow([
            f"{stats.time_elapsed:.2f}",
            stats.drones_survived,
            stats.drones_total,
            stats.relay_survived,
            stats.target_reached,
            f"{stats.peak_signal_coverage:.3f}",
            stats.end_reason,
        ])
