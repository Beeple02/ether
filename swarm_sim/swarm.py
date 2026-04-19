import time
import numpy as np
from . import config
from .agent import Agent
from .pathfinder import PathFinder
from .physics import normalize
from .environment import Projectile
from .mission import MissionFSM, FormationFSM
from .effects import EMPBlast, SmokeCloud, NetDeploy, Explosion


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
        self._mission   = MissionFSM()
        self._formation = FormationFSM()
        self.effects    = []
        # assign per-FAST convergence angles
        fast_drones = [d for d in self.drones if d.drone_type == "fast"]
        for k, d in enumerate(fast_drones):
            d._convergence_angle = 2 * np.pi * k / max(len(fast_drones), 1)

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

        dt  = 1.0 / config.FPS
        env = self._environment

        # tick FSMs
        self._mission.tick(dt, self)
        self._formation.tick(dt, self._mission.phase)

        # tick and cull dead effects
        for fx in self.effects:
            fx.tick(dt)
        self.effects = [fx for fx in self.effects if fx.alive]

        self._update_relay()
        self._update_signals()
        self._update_noflyzone()

        alive = [d for d in self.drones if d.alive]
        n_alive = len(alive)

        target_center = self._target_center()
        n_interceptors = sum(
            1 for d in alive
            if d.drone_type in ("interceptor_net", "interceptor_fuse")
        )

        # shared context written-to by all drone ticks this frame
        ctx = {
            "phase":               self._mission.phase,
            "formation_mode":      self._formation.mode,
            "relay_pos":           self.relay.position.copy(),
            "relay_vel":           self.relay.velocity.copy(),
            "turrets":             env.turrets if env else [],
            "projectiles":         self.projectiles,
            "enemy_drones":        [],
            "player_drones":       self.drones,
            "all_agents":          self.drones + [self.relay],
            "target_center":       target_center,
            "target_radius":       120.0,
            "threats":             [],
            "events":              [],
            "n_interceptors":      n_interceptors,
            "convergence_enabled": config.CONVERGENCE_ENABLED,
            "t_zero":              self._mission.total_elapsed + config.CONVERGENCE_T_LEAD,
            "total_elapsed":       self._mission.total_elapsed,
        }

        interceptor_idx = 0
        for i, drone in enumerate(alive):
            ctx["drone_idx"] = i
            if drone.drone_type in ("interceptor_net", "interceptor_fuse"):
                ctx["drone_idx"] = interceptor_idx
                interceptor_idx += 1
            ft = self._formation_target(drone, i, n_alive)
            drone.tick(self._neighbors(drone, alive), ft, env, context=ctx)

        self._process_events(ctx["events"])
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
    def _formation_target(self, drone, drone_idx, n_alive):
        mode      = self._formation.mode
        phase     = self._mission.phase
        relay_pos = self.relay.position
        relay_vel = self.relay.velocity

        # PERSISTENCE: return non-loiter drones to base
        if phase == "PERSISTENCE" and drone.drone_type != "loiter":
            base = self._base_pos()
            if base is not None:
                return base

        if mode == "COLUMN":
            speed = float(np.linalg.norm(relay_vel))
            fwd   = relay_vel / speed if speed > 0.1 else np.array([0.0, -1.0])
            perp  = np.array([-fwd[1], fwd[0]])
            col   = (drone_idx % 4) - 1.5
            row   = drone_idx // 4 + 1
            return relay_pos - fwd * (row * 18) + perp * (col * 14)

        if mode == "DENSE":
            ft    = self._drone_follow_target()
            angle = 2 * np.pi * drone_idx / max(n_alive, 1)
            return ft + np.array([np.cos(angle), np.sin(angle)]) * 35

        if mode == "DISPERSED":
            angle = 2 * np.pi * drone_idx / max(n_alive, 1)
            return relay_pos + np.array([np.cos(angle), np.sin(angle)]) * 130

        if mode == "PURSUIT":
            tc = self._target_center()
            if tc is not None:
                fwd  = normalize(tc - relay_pos)
                perp = np.array([-fwd[1], fwd[0]])
                side = 1 if drone_idx % 2 == 0 else -1
                row  = drone_idx // 2 + 1
                return relay_pos + fwd * (row * 12) + perp * (side * row * 20)
            return relay_pos.copy()

        if mode == "BUBBLE":
            angle = 2 * np.pi * drone_idx / max(n_alive, 1)
            return relay_pos + np.array([np.cos(angle), np.sin(angle)]) * config.BUBBLE_RADIUS

        return self._drone_follow_target()

    # ── event processing ──────────────────────────────────────────────────────
    def _process_events(self, events):
        env = self._environment
        for evt in events:
            etype = evt["type"]

            if etype == "emp_detonate":
                pos    = evt["position"]
                radius = evt.get("radius", config.EMP_EFFECT_RADIUS)
                self.effects.append(EMPBlast(pos, radius))
                self._mission.on_emp_detonated()
                if env:
                    for t in env.turrets:
                        if np.linalg.norm(t.position - pos) <= radius:
                            t._disabled_timer = config.EMP_DISABLE_TIME
                for d in self.drones:
                    if d.alive and d.side == "enemy":
                        if np.linalg.norm(d.position - pos) <= radius:
                            d._stun_timer = config.EMP_STUN_TIME

            elif etype == "smoke_deploy":
                self.effects.append(SmokeCloud(
                    evt["position"],
                    config.SMOKESCREEN_RADIUS,
                    config.SMOKESCREEN_PERSIST,
                ))

            elif etype == "net_deploy":
                pos    = evt["position"]
                radius = evt.get("radius", config.INTERCEPTOR_NET_RADIUS)
                self.effects.append(NetDeploy(pos, radius))
                for p in self.projectiles:
                    if p.alive and np.linalg.norm(p.position - pos) <= radius:
                        p.alive = False
                for d in self.drones:
                    if d.alive and d.side == "enemy":
                        if np.linalg.norm(d.position - pos) <= radius:
                            d.alive = False
                            self.effects.append(Explosion(d.position.copy(), 12))

            elif etype == "fuse_detonate":
                pos    = evt["position"]
                radius = evt.get("radius", config.INTERCEPTOR_FUSE_RANGE * 3)
                self.effects.append(Explosion(pos, radius * 0.5, (255, 180, 30)))
                for p in self.projectiles:
                    if p.alive and np.linalg.norm(p.position - pos) <= radius:
                        p.alive = False
                for d in self.drones:
                    if d.alive and d.side == "enemy":
                        if np.linalg.norm(d.position - pos) <= radius:
                            d.alive = False

            elif etype == "explosion":
                self.effects.append(Explosion(
                    evt["position"], 18,
                    evt.get("color", (255, 140, 40)),
                ))

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
            jammer_pos = [
                d.position.copy() for d in self.drones
                if d.alive and d.drone_type == "jammer"
            ]
            candidates = [d for d in self.drones if d.alive]
            if self.relay.alive:
                candidates.append(self.relay)

            for turret in env.turrets:
                # EMP disable countdown
                if turret._disabled_timer > 0:
                    turret._disabled_timer -= dt
                    continue

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
                if nearest is None:
                    continue

                # jammer miss chance
                jammed = any(
                    np.linalg.norm(jp - turret.position) <= config.JAMMER_RANGE
                    for jp in jammer_pos
                )
                if jammed and np.random.random() < config.JAMMER_MISS_CHANCE:
                    turret._cooldown = 1.0 / max(turret.fire_rate, 1e-3)
                    continue

                direction = normalize(nearest.position - turret.position)
                self.projectiles.append(Projectile(turret.position.copy(),
                                                   direction * speed_per_frame))
                turret._cooldown = 1.0 / max(turret.fire_rate, 1e-3)

        smoke_clouds = [fx for fx in self.effects
                        if isinstance(fx, SmokeCloud) and fx.alive]

        for p in self.projectiles:
            if not p.alive:
                continue
            p.position = p.position + p.velocity
            if (p.position[0] < 0 or p.position[0] > W or
                    p.position[1] < 0 or p.position[1] > H):
                p.alive = False
                continue

            # smoke cloud miss chance
            in_smoke = any(
                np.linalg.norm(p.position - sc.position) < sc.radius
                for sc in smoke_clouds
            )
            if in_smoke and np.random.random() < config.SMOKESCREEN_MISS_CH:
                p.alive = False
                continue

            hit = False
            for d in self.drones:
                if d.alive and np.linalg.norm(p.position - d.position) < config.DRONE_HIT_RADIUS:
                    d.alive = False
                    p.alive = False
                    self.effects.append(Explosion(d.position.copy(), 12, (255, 120, 30)))
                    hit = True
                    break
            if not hit and self.relay.alive:
                if np.linalg.norm(p.position - self.relay.position) < config.RELAY_HIT_RADIUS:
                    self.relay.alive = False
                    p.alive = False
                    self.effects.append(Explosion(self.relay.position.copy(), 20, (255, 60, 20)))

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

    # ── phase control ─────────────────────────────────────────────────────────
    def advance_phase(self):
        if self._mission.advance():
            self._formation.set_for_phase(self._mission.phase)

    # ── environment helpers ───────────────────────────────────────────────────
    def _base_pos(self):
        env = self._environment
        if env and env.base is not None:
            return env.base.position.copy()
        return None

    def _target_center(self):
        env = self._environment
        if not env:
            return None
        targets = [z for z in env.zones if z.zone_type == "TARGET"]
        if not targets:
            return None
        return np.mean([z.center() for z in targets], axis=0)

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
