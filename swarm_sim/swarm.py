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
        self._mission          = MissionFSM()
        self._formation        = FormationFSM()
        self.effects           = []
        self._succession_active = False   # True while mimicry window is open
        self._last_threats      = []      # RECON reports from last tick (for renderer)

        # assign per-FAST convergence angles
        fast_drones = [d for d in self.drones if d.drone_type == "fast"]
        for k, d in enumerate(fast_drones):
            d._convergence_angle = 2 * np.pi * k / max(len(fast_drones), 1)

        # assign succession ranks 1..N to relay_backup drones
        backups = [d for d in self.drones if d.drone_type == "relay_backup"]
        for rank, d in enumerate(backups, start=1):
            d.succession_rank = rank

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

        self._update_succession()
        self._update_relay()
        self._update_signals()
        self._update_noflyzone()

        # Exclude promoted drones (_is_relay) from the flock loop — they are
        # moved by _update_relay() and drawn via _draw_relay(), not as drones.
        alive = [d for d in self.drones if d.alive and not d._is_relay]
        n_alive = len(alive)

        target_center = self._target_center()

        # pre-compute per-interceptor indices (stable, dict-keyed by id)
        _INT_TYPES = ("interceptor_net", "interceptor_fuse")
        int_alive   = [d for d in alive if d.drone_type in _INT_TYPES]
        int_indices = {id(d): i for i, d in enumerate(int_alive)}
        n_interceptors = len(int_alive)

        # BUBBLE trigger — enemy projectile or drone within threshold of relay
        relay_p  = self.relay.position
        trig_r2  = config.BUBBLE_TRIGGER_RADIUS ** 2
        _threat  = any(np.sum((p.position - relay_p) ** 2) < trig_r2
                       for p in self.projectiles if p.alive)
        if not _threat:
            _threat = any(np.sum((d.position - relay_p) ** 2) < trig_r2
                          for d in self.drones if d.alive and d.side == "enemy")
        if _threat:
            self._formation.set_bubble(True)

        # all_agents for collision avoidance — relay counted exactly once
        _relay_in_drones = any(d is self.relay for d in self.drones)
        all_agents = alive + ([] if _relay_in_drones else [self.relay])

        # shared context written-to by all drone ticks this frame
        ctx = {
            "phase":               self._mission.phase,
            "formation_mode":      self._formation.mode,
            "relay_pos":           relay_p.copy(),
            "relay_vel":           self.relay.velocity.copy(),
            "turrets":             env.turrets if env else [],
            "projectiles":         self.projectiles,
            "enemy_drones":        [],
            "player_drones":       self.drones,
            "all_agents":          all_agents,
            "target_center":       target_center,
            "target_radius":       120.0,
            "smoke_clouds":        [fx for fx in self.effects
                                    if isinstance(fx, SmokeCloud) and fx.alive],
            "threats":             [],
            "events":              [],
            "n_interceptors":      n_interceptors,
            "convergence_enabled": config.CONVERGENCE_ENABLED,
            "t_zero":              self._mission.total_elapsed + config.CONVERGENCE_T_LEAD,
            "total_elapsed":       self._mission.total_elapsed,
        }

        for i, drone in enumerate(alive):
            is_int  = drone.drone_type in _INT_TYPES
            int_idx = int_indices.get(id(drone), 0) if is_int else 0
            ctx["drone_idx"] = int_idx if is_int else i
            ft = self._formation_target(drone, i, n_alive, int_idx, n_interceptors)
            drone.tick(self._neighbors(drone, alive), ft, env, context=ctx)

        self._process_events(ctx["events"])
        self._process_threat_reports(ctx["threats"])
        self._last_threats = ctx["threats"]
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
    def _formation_target(self, drone, drone_idx, n_alive,
                          interceptor_idx=0, n_interceptors=0):
        mode      = self._formation.mode
        phase     = self._mission.phase
        relay_pos = self.relay.position
        relay_vel = self.relay.velocity
        is_int    = drone.drone_type in ("interceptor_net", "interceptor_fuse")

        # Mimicry backups navigate to relay goal (waypoint or target zone),
        # making them visually indistinguishable from the real relay in motion.
        if drone.drone_type == "relay_backup" and drone._mimicry_active:
            goal = self._relay_goal()
            return goal if goal is not None else relay_pos.copy()

        # PERSISTENCE: non-loiter drones return to base
        if phase == "PERSISTENCE" and drone.drone_type != "loiter":
            base = self._base_pos()
            if base is not None:
                return base

        # ── BUBBLE positioning rules ──────────────────────────────────────────
        # Applies when: (a) mode == BUBBLE  OR (b) SATURATION phase (interceptors
        # always hold the perimeter even while the main formation is DENSE).
        # Interceptors → equal-angle ring at BUBBLE_RADIUS.
        # All other drones → hold tight DENSE around relay.
        want_bubble = (mode == "BUBBLE") or (phase == "SATURATION" and is_int)
        if want_bubble:
            if is_int:
                n     = max(n_interceptors, 1)
                angle = 2 * np.pi * interceptor_idx / n
                return relay_pos + np.array([np.cos(angle), np.sin(angle)]) * config.BUBBLE_RADIUS
            else:
                # non-interceptors: DENSE ring around relay
                angle = 2 * np.pi * drone_idx / max(n_alive, 1)
                return relay_pos + np.array([np.cos(angle), np.sin(angle)]) * (
                    config.SEPARATION_RADIUS * 2)

        # ── COLUMN: staggered double-file behind relay heading ─────────────────
        # Spec: 2-wide, spaced by SEPARATION_RADIUS.
        if mode == "COLUMN":
            speed = float(np.linalg.norm(relay_vel))
            fwd   = relay_vel / speed if speed > 0.1 else np.array([0.0, -1.0])
            perp  = np.array([-fwd[1], fwd[0]])
            col   = (drone_idx % 2) - 0.5          # left/right file: -0.5 or +0.5
            row   = drone_idx // 2 + 1             # rows: 1 1 2 2 3 3 …
            sp    = float(config.SEPARATION_RADIUS)
            return relay_pos - fwd * (row * sp) + perp * (col * sp)

        # ── DENSE: tight cluster around relay ─────────────────────────────────
        if mode == "DENSE":
            angle = 2 * np.pi * drone_idx / max(n_alive, 1)
            return relay_pos + np.array([np.cos(angle), np.sin(angle)]) * (
                config.SEPARATION_RADIUS * 2)

        # ── DISPERSED: multi-ring layout, ~PERCEPTION_RADIUS between rings ─────
        if mode == "DISPERSED":
            return self._dispersed_target(drone_idx, relay_pos)

        # ── PURSUIT: each drone heads directly to target zone ─────────────────
        if mode == "PURSUIT":
            tc = self._target_center()
            return tc if tc is not None else relay_pos.copy()

        return self._drone_follow_target()

    def _dispersed_target(self, drone_idx, center):
        """Assign drone_idx a slot in expanding rings spaced PERCEPTION_RADIUS apart."""
        per_r = float(config.PERCEPTION_RADIUS)
        ring  = 1
        total = 0
        while True:
            n_in_ring = max(1, int(2 * np.pi * ring))   # ≈ 6, 12, 19, 25, 31 …
            if drone_idx < total + n_in_ring:
                slot  = drone_idx - total
                angle = 2 * np.pi * slot / n_in_ring
                return center + np.array([np.cos(angle), np.sin(angle)]) * (ring * per_r)
            total += n_in_ring
            ring  += 1
            if ring > 25:   # safety cap — beyond 2000 px
                break
        return center.copy()

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

            elif etype == "mimicry_expired":
                self._try_promote_or_revert(evt["drone"])

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
            # End only when no succession can save the mission:
            # succession is inactive AND no live backups remain.
            if not self._succession_active:
                alive_backups = [d for d in self.drones
                                 if d.alive and d.drone_type == "relay_backup"]
                if not alive_backups:
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

    # ── RECON threat aggregation ──────────────────────────────────────────────
    def _process_threat_reports(self, threats):
        """Aggregate per-tick RECON reports and trigger formation responses.

        Priority (highest→lowest):
          PROJECTILE_THREAT           → BUBBLE
          DRONE_THREAT or ≥3 turrets  → DISPERSED
          1-2 TURRET_THREAT+SATURATION→ DENSE
        """
        if not threats:
            return

        # Deduplicate by object identity; multiple RECON drones may see the same object.
        seen_turrets = {}
        seen_projs   = {}
        seen_enemies = {}
        for t in threats:
            oid = id(t["obj"])
            if   t["kind"] == "TURRET_THREAT":      seen_turrets[oid] = t
            elif t["kind"] == "PROJECTILE_THREAT":  seen_projs[oid]   = t
            elif t["kind"] == "DRONE_THREAT":        seen_enemies[oid] = t

        n_turrets = len(seen_turrets)
        has_proj  = bool(seen_projs)
        has_drone = bool(seen_enemies)
        phase     = self._mission.phase

        if has_proj:
            self._formation.set_bubble(True)
        elif has_drone or n_turrets >= 3:
            self._formation.set_recon_override("DISPERSED")
        elif n_turrets > 0 and phase == "SATURATION":
            self._formation.set_recon_override("DENSE")

    # ── relay succession ──────────────────────────────────────────────────────
    def _update_succession(self):
        """Open mimicry window when relay dies; close it if all backups die."""
        if self.relay.alive:
            return

        if not self._succession_active:
            backups = [d for d in self.drones
                       if d.alive and d.drone_type == "relay_backup"]
            if backups:
                self._succession_active = True
                for d in backups:
                    d._mimicry_active = True
                    d._mimicry_timer  = np.random.uniform(
                        config.MIMICRY_DELAY_MIN, config.MIMICRY_DELAY_MAX
                    )
        else:
            # Check if all backups died mid-window — succession fails
            alive_backups = [d for d in self.drones
                             if d.alive and d.drone_type == "relay_backup"]
            if not alive_backups:
                self._succession_active = False   # triggers run-end next frame

    def _try_promote_or_revert(self, drone):
        """Called when a relay_backup's mimicry timer expires."""
        if not drone.alive:
            return
        # Find alive backups by rank to determine if this drone should promote
        alive_backups = sorted(
            [d for d in self.drones
             if d.alive and d.drone_type == "relay_backup"],
            key=lambda x: x.succession_rank,
        )
        if alive_backups and alive_backups[0] is drone:
            self._promote_relay(drone)
        else:
            # Not the designated heir — revert to fast behavior
            drone._mimicry_active = False
            drone.drone_type      = "fast"
            drone.succession_rank = 0

    def _promote_relay(self, drone):
        """Promote drone to full relay status."""
        drone._mimicry_active = False
        drone._is_relay       = True
        drone.role            = "relay"
        self.relay            = drone          # swap relay reference

        # Reset pathfinding from the new relay's current position
        self._relay_path  = []
        self._path_target = None
        self._replan_in   = 0

        # Immediately end mimicry for all remaining backups → revert to fast
        for d in self.drones:
            if d is not drone and d.alive and d.drone_type == "relay_backup":
                d._mimicry_active = False
                d.drone_type      = "fast"
                d.succession_rank = 0

        # Re-rank any surviving relay_backup drones for future succession
        new_backups = [d for d in self.drones
                       if d.alive and d.drone_type == "relay_backup"]
        for rank, d in enumerate(new_backups, start=1):
            d.succession_rank = rank

        self._succession_active = False

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
