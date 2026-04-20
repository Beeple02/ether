import numpy as np
from . import config
from .physics import normalize, limit, seek, euler_integrate


class Agent:
    def __init__(self, position, role="drone", drone_type="fast", side="player"):
        self.position   = np.array(position, dtype=float)
        angle           = np.random.uniform(0, 2 * np.pi)
        self.velocity   = np.array([np.cos(angle), np.sin(angle)]) * np.random.uniform(0.5, 1.5)
        self.role       = role
        self.drone_type = drone_type if role == "drone" else None
        self.side       = side          # "player" or "enemy"
        self.alive      = True
        self._t         = np.random.uniform(0, 2 * np.pi)
        self.signal     = 1.0

        # debug
        self.last_forces             = {}
        self.last_neighbor_positions = []
        self.active_reflexes         = set()  # for debug overlay

        # relay succession (relay_backup only)
        self.succession_rank       = 0
        self._mimicry_active       = False
        self._mimicry_timer        = 0.0
        self._is_relay             = (role == "relay")
        self._intercept_target_agent = None   # tracked target for interceptors

        # per-type runtime state
        self._smoke_charges   = config.SMOKESCREEN_CHARGES
        self._smoke_cd        = 0.0
        self._net_cd          = 0.0
        self._emp_stage       = "attached"   # attached/detaching/arming/detonated
        self._emp_arming_t    = 0.0
        self._emp_target_pos  = None
        self._loiter_angle    = np.random.uniform(0, 2 * np.pi)
        self._convergence_angle = 0.0       # assigned by swarm
        self._stun_timer      = 0.0         # enemy stun from EMP

    # ── properties ────────────────────────────────────────────────────────────
    @property
    def speed_mult(self):
        if self._is_relay or self.role == "relay":
            return 1.0
        if self.side == "enemy":
            return config.ENEMY_DRONE_TYPES.get(self.drone_type, {}).get("speed_mult", 1.0)
        return config.DRONE_TYPES.get(self.drone_type or "fast", {}).get("speed_mult", 1.0)

    @property
    def type_color(self):
        if self._is_relay or self.role == "relay":
            return (255, 220, 40)
        if self._mimicry_active:
            t = (pygame_time() % 0.6) / 0.6
            pulse = int(abs(t - 0.5) * 2 * 215)
            return (255, 220, pulse)
        if self.side == "enemy":
            return config.ENEMY_DRONE_TYPES.get(self.drone_type, {}).get("color", (200, 50, 50))
        return config.DRONE_TYPES.get(self.drone_type or "fast", {}).get("color", (205, 215, 230))

    @property
    def type_size(self):
        if self._is_relay or self.role == "relay":
            return 10
        if self.side == "enemy":
            return config.ENEMY_DRONE_TYPES.get(self.drone_type, {}).get("size", 4)
        return config.DRONE_TYPES.get(self.drone_type or "fast", {}).get("size", 4)

    @property
    def airframe(self):
        if self._is_relay or self.role == "relay":
            return "large"
        dt = self.drone_type or "fast"
        return config.DRONE_TYPES.get(dt, {}).get("airframe", "small")

    # ── main tick ──────────────────────────────────────────────────────────────
    def tick(self, neighbors, formation_target, environment=None, context=None):
        if not self.alive:
            return
        if context is None:
            context = {}
        dt = 1.0 / config.FPS

        # decrement cooldowns
        self._smoke_cd = max(0.0, self._smoke_cd - dt)
        self._net_cd   = max(0.0, self._net_cd   - dt)
        if self._stun_timer > 0:
            self._stun_timer -= dt

        # relay role
        if self._is_relay or self.role == "relay":
            return  # movement handled by Swarm._update_relay

        # relay_backup mimicry — move like relay toward relay goal
        if self.drone_type == "relay_backup" and self._mimicry_active:
            self._mimicry_timer -= dt
            if self._mimicry_timer <= 0:
                # Timer expired: emit event so swarm can decide promote vs. revert
                self._mimicry_active = False
                context.setdefault("events", []).append({
                    "type":  "mimicry_expired",
                    "drone": self,
                })
                # Fall through to normal relay_backup behavior this tick
            else:
                # Still in window: navigate toward relay's goal (formation_target
                # is set to relay_goal by swarm._formation_target for these drones)
                self._move_toward(
                    formation_target if formation_target is not None
                    else np.array(config.WORLD_SIZE, dtype=float) / 2,
                    environment,
                )
                self._finalize_reflex(context, environment)
                return

        # relay_backup that just got promoted
        if self._is_relay:
            return

        # enemy drones
        if self.side == "enemy":
            self._enemy_tick(context, environment, dt)
            self._finalize_reflex(context, environment)
            return

        # player drones — type-specific then boids fallback
        phase = context.get("phase", "TRANSIT")
        handled = self._special_tick(context, environment, dt, phase, neighbors, formation_target)
        if not handled:
            self._apply_boids(neighbors, formation_target, environment)
        self._finalize_reflex(context, environment)

    # ── type-specific behaviors ────────────────────────────────────────────────
    def _special_tick(self, context, environment, dt, phase, neighbors, formation_target):
        dt_name = self.drone_type
        if dt_name == "emp":
            return self._tick_emp(context, environment, dt, phase)
        if dt_name == "jammer":
            return self._tick_jammer(context, environment, dt, phase)
        if dt_name == "recon":
            return self._tick_recon(context, environment, dt)
        if dt_name == "loiter":
            return self._tick_loiter(context, environment, dt, phase)
        if dt_name == "smokescreen":
            return self._tick_smokescreen(context, environment, dt, neighbors, formation_target)
        if dt_name == "decoy":
            return self._tick_decoy(context, environment, dt, phase, formation_target)
        if dt_name in ("interceptor_net", "interceptor_fuse"):
            return self._tick_interceptor(context, environment, dt, neighbors, formation_target)
        if dt_name == "relay_backup":
            return self._tick_relay_backup(context, environment, dt, phase, formation_target)
        if dt_name == "fast":
            return self._tick_fast(context, environment, dt, phase, neighbors, formation_target)
        if dt_name == "heavy":
            return self._tick_heavy(context, environment, dt, phase)
        return False

    # EMP ──────────────────────────────────────────────────────────────────────
    def _tick_emp(self, context, environment, dt, phase):
        if phase != "SUPPRESSION":
            return False

        relay_pos  = context.get("relay_pos")
        relay_vel  = context.get("relay_vel", np.zeros(2))
        if relay_pos is None:
            return False

        heading = normalize(relay_vel) if np.linalg.norm(relay_vel) > 0.1 else np.array([0, -1.0])
        target_pos = relay_pos + heading * config.EMP_LEAD_DIST

        if self._emp_stage == "attached":
            self._emp_stage    = "detaching"
            self._emp_target_pos = target_pos.copy()

        if self._emp_stage == "detaching":
            self._emp_target_pos = target_pos.copy()
            d = np.linalg.norm(self._emp_target_pos - self.position)
            self._move_toward(self._emp_target_pos, environment)
            if d < 25:
                self._emp_stage   = "arming"
                self._emp_arming_t = 0.0
            return True

        if self._emp_stage == "arming":
            self._emp_arming_t += dt
            # hover in place
            self.velocity *= 0.85
            if self._emp_arming_t >= config.EMP_ARMING_DELAY:
                self._emp_stage = "detonating"
                context.setdefault("events", []).append({
                    "type": "emp_detonate",
                    "position": self.position.copy(),
                    "radius": config.EMP_EFFECT_RADIUS,
                    "agent": self,
                })
                self.alive = False
            return True

        return True

    # JAMMER ───────────────────────────────────────────────────────────────────
    def _tick_jammer(self, context, environment, dt, phase):
        if phase not in ("SUPPRESSION", "SATURATION"):
            return False
        relay_pos = context.get("relay_pos")
        relay_vel = context.get("relay_vel", np.zeros(2))
        if relay_pos is None:
            return False
        heading    = normalize(relay_vel) if np.linalg.norm(relay_vel) > 0.1 else np.array([0, -1.0])
        orbit_center = relay_pos + heading * (config.EMP_LEAD_DIST * 0.6)
        self._t   += 0.04
        orbit_r    = 60.0
        tgt = orbit_center + np.array([np.cos(self._t), np.sin(self._t)]) * orbit_r
        self._move_toward(tgt, environment)
        return True

    # RECON ────────────────────────────────────────────────────────────────────
    def _tick_recon(self, context, environment, dt):
        relay_pos = context.get("relay_pos")
        relay_vel = context.get("relay_vel", np.zeros(2))
        if relay_pos is None:
            return False
        heading = normalize(relay_vel) if np.linalg.norm(relay_vel) > 0.1 else np.array([0, -1.0])
        tgt = relay_pos + heading * config.RECON_LEAD_DIST
        self._move_toward(tgt, environment)

        # scan for threats
        threats = context.setdefault("threats", [])
        turrets  = context.get("turrets", [])
        for t in turrets:
            d = np.linalg.norm(t.position - self.position)
            if d < config.RECON_RANGE:
                threats.append({"kind": "TURRET_THREAT", "pos": t.position.copy(), "obj": t})

        projectiles = context.get("projectiles", [])
        for p in projectiles:
            if relay_pos is not None:
                d = np.linalg.norm(p.position - relay_pos)
                if d < config.RECON_PROJ_ALERT:
                    threats.append({"kind": "PROJECTILE_THREAT", "pos": p.position.copy(), "obj": p})

        enemy_drones = context.get("enemy_drones", [])
        for e in enemy_drones:
            if e.alive:
                d = np.linalg.norm(e.position - self.position)
                if d < config.RECON_RANGE:
                    threats.append({"kind": "DRONE_THREAT", "pos": e.position.copy(), "obj": e})

        return True

    # LOITER ───────────────────────────────────────────────────────────────────
    def _tick_loiter(self, context, environment, dt, phase):
        if phase not in ("PROSECUTION", "PERSISTENCE"):
            return False
        target_center = context.get("target_center")
        if target_center is None:
            return False
        self._loiter_angle += 0.015 * (config.MAX_SPEED * config.DRONE_TYPES["loiter"]["speed_mult"])
        orbit_r = 80.0
        tgt = target_center + np.array([np.cos(self._loiter_angle), np.sin(self._loiter_angle)]) * orbit_r
        self._move_toward(tgt, environment)
        return True

    # SMOKESCREEN ──────────────────────────────────────────────────────────────
    def _tick_smokescreen(self, context, environment, dt, neighbors, formation_target):
        if self._smoke_charges <= 0:
            return False
        relay_pos    = context.get("relay_pos")
        projectiles  = context.get("projectiles", [])
        if relay_pos is None:
            return False
        for p in projectiles:
            to_relay = relay_pos - p.position
            if np.dot(p.velocity, to_relay) > 0:
                d = np.linalg.norm(to_relay)
                if d < config.SMOKESCREEN_RANGE:
                    interpose = p.position + normalize(to_relay) * (d * 0.5)
                    self._move_toward(interpose, environment)
                    if self._smoke_cd <= 0 and np.linalg.norm(self.position - interpose) < 40:
                        self._smoke_charges -= 1
                        self._smoke_cd = config.SMOKESCREEN_CD
                        context.setdefault("events", []).append({
                            "type": "smoke_deploy",
                            "position": self.position.copy(),
                        })
                    return True
        return False

    # DECOY ────────────────────────────────────────────────────────────────────
    def _tick_decoy(self, context, environment, dt, phase, formation_target):
        if phase not in ("SATURATION", "PROSECUTION"):
            return False
        relay_pos = context.get("relay_pos")
        if relay_pos is None:
            return False
        # mimic relay movement: follow relay closely with slight offset
        self._t += 0.02
        offset = np.array([np.cos(self._t) * 30, np.sin(self._t) * 30])
        self._move_toward(relay_pos + offset, environment)
        return True

    # INTERCEPTOR ──────────────────────────────────────────────────────────────
    def _tick_interceptor(self, context, environment, dt, neighbors, formation_target):
        relay_pos    = context.get("relay_pos")
        threats      = context.get("threats", [])
        projectiles  = context.get("projectiles", [])
        enemy_drones = context.get("enemy_drones", [])
        form_mode    = context.get("formation_mode", "DENSE")

        # Activation gate: only hunt when BUBBLE is active OR DRONE_THREAT reported.
        # BUBBLE is triggered by proximity (≤500px) or by RECON PROJECTILE_THREAT.
        bubble_active    = (form_mode == "BUBBLE")
        has_drone_threat = any(t["kind"] == "DRONE_THREAT" for t in threats)
        should_hunt      = bubble_active or has_drone_threat

        if should_hunt:
            best_target_pos = None
            best_d          = 1e18

            # enemy drones first
            for e in enemy_drones:
                if e.alive and not _is_iff_safe(self, e):
                    d = np.linalg.norm(e.position - self.position)
                    if d < best_d:
                        best_d                       = d
                        best_target_pos              = e.position.copy()
                        self._intercept_target_agent = e

            # then incoming projectiles
            if best_target_pos is None:
                for p in projectiles:
                    if p.alive:
                        d = np.linalg.norm(p.position - self.position)
                        if d < best_d:
                            best_d                       = d
                            best_target_pos              = p.position.copy()
                            self._intercept_target_agent = p

            if best_target_pos is not None:
                self._move_toward(best_target_pos, environment)
                trigger_r = (config.INTERCEPTOR_NET_RANGE
                             if self.drone_type == "interceptor_net"
                             else config.INTERCEPTOR_FUSE_RANGE)

                if best_d < trigger_r and self._net_cd <= 0:
                    if self.drone_type == "interceptor_net":
                        self._net_cd = config.INTERCEPTOR_NET_CD
                        context.setdefault("events", []).append({
                            "type":     "net_deploy",
                            "position": self.position.copy(),
                            "radius":   config.INTERCEPTOR_NET_RADIUS,
                            "drone":    self,
                        })
                    else:  # fuse — close-range area detonation, destroys self
                        context.setdefault("events", []).append({
                            "type":     "fuse_detonate",
                            "position": self.position.copy(),
                            "radius":   config.INTERCEPTOR_FUSE_RANGE * 3,
                            "drone":    self,
                        })
                        self.alive = False
                return True

        # No active threat (or hunt conditions not met) — hold BUBBLE perimeter.
        # Always maintain perimeter regardless of current formation mode so
        # interceptors stay positioned to react instantly when threats appear.
        if relay_pos is not None:
            idx   = context.get("drone_idx", 0)
            n     = max(context.get("n_interceptors", 1), 1)
            angle = 2 * np.pi * idx / n
            tgt   = relay_pos + np.array([np.cos(angle), np.sin(angle)]) * config.BUBBLE_RADIUS
            self._move_toward(tgt, environment)
            return True
        return False

    # RELAY_BACKUP ─────────────────────────────────────────────────────────────
    def _tick_relay_backup(self, context, environment, dt, phase, formation_target):
        # behaves like fast drone but stays near relay
        return False  # fall through to boids

    # FAST ─────────────────────────────────────────────────────────────────────
    def _tick_fast(self, context, environment, dt, phase, neighbors, formation_target):
        conv = context.get("convergence_enabled", False)
        if conv and phase in ("SATURATION", "PROSECUTION"):
            target_center = context.get("target_center")
            if target_center is not None:
                angle = self._convergence_angle
                radius = context.get("target_radius", 120.0)
                entry  = target_center + np.array([np.cos(angle), np.sin(angle)]) * radius
                t_zero = context.get("t_zero", 0.0)
                now    = context.get("total_elapsed", 0.0)
                remaining = max(0.01, t_zero - now)
                d = np.linalg.norm(entry - self.position)
                if remaining > 0:
                    desired_speed = min(d / remaining, config.MAX_SPEED * self.speed_mult)
                    if d > 5:
                        self.velocity = normalize(entry - self.position) * desired_speed
                        if environment:
                            self.velocity += limit(environment.repulsion_force(self.position) * 2.0,
                                                   config.MAX_SPEED * 0.5)
                        self.velocity = limit(self.velocity, config.MAX_SPEED * self.speed_mult)
                        return True
        return False

    # HEAVY ────────────────────────────────────────────────────────────────────
    def _tick_heavy(self, context, environment, dt, phase):
        if phase not in ("PROSECUTION", "PERSISTENCE"):
            return False
        target_center = context.get("target_center")
        if target_center is None:
            return False
        self._move_toward(target_center, environment)
        return True

    # enemy AI ─────────────────────────────────────────────────────────────────
    def _enemy_tick(self, context, environment, dt):
        if self._stun_timer > 0:
            # random wander
            self.velocity += np.random.uniform(-0.3, 0.3, 2)
            self.velocity  = limit(self.velocity, config.MAX_SPEED * self.speed_mult)
            return

        relay_pos    = context.get("relay_pos")
        player_drones = context.get("player_drones", [])
        backups = [d for d in player_drones
                   if d.alive and d.drone_type == "relay_backup"]

        if self.drone_type == "hunter":
            tgt = relay_pos
            if tgt is None and backups:
                tgt = backups[0].position
            if tgt is None and player_drones:
                alive_p = [d for d in player_drones if d.alive]
                if alive_p:
                    tgt = alive_p[0].position
            if tgt is not None:
                self._move_toward(tgt, environment)

        elif self.drone_type == "kamikaze":
            alive_p = [d for d in player_drones if d.alive]
            if alive_p:
                nearest = min(alive_p, key=lambda d: np.linalg.norm(d.position - self.position))
                self._move_toward(nearest.position, environment)
                if np.linalg.norm(nearest.position - self.position) < 12:
                    nearest.alive = False
                    context.setdefault("events", []).append({
                        "type": "explosion",
                        "position": self.position.copy(),
                        "color": (155, 25, 25),
                    })
                    self.alive = False

    # ── movement helpers ───────────────────────────────────────────────────────
    def _move_toward(self, target, environment=None):
        target = np.array(target, dtype=float)
        desired = normalize(target - self.position) * config.MAX_SPEED
        steer   = limit(desired - self.velocity, config.MAX_FORCE * 2)
        self.velocity += steer
        if environment:
            self.velocity += limit(environment.repulsion_force(self.position) * 2.0,
                                   config.MAX_SPEED * 0.6)
        self.velocity = limit(self.velocity, config.MAX_SPEED * self.speed_mult)

    def _move_relay(self, environment=None):
        self._t += 0.008
        cx, cy = config.WORLD_SIZE[0] / 2, config.WORLD_SIZE[1] / 2
        r = min(config.WORLD_SIZE) * 0.3
        target = np.array([cx + r * np.sin(self._t),
                           cy + r * np.sin(self._t * 2) * 0.5])
        self.velocity = limit(target - self.position, config.MAX_SPEED)
        if environment:
            rep = environment.repulsion_force(self.position)
            self.velocity += limit(rep * 3.0, config.MAX_SPEED * 0.7)
        self.velocity = limit(self.velocity, config.MAX_SPEED)
        self.position = euler_integrate(self.position, self.velocity)

    def move_relay_toward(self, target, environment=None):
        desired = normalize(target - self.position) * config.MAX_SPEED
        self.velocity = limit(desired, config.MAX_SPEED)
        if environment:
            rep = environment.repulsion_force(self.position)
            self.velocity += limit(rep * 3.0, config.MAX_SPEED * 0.7)
        self.velocity = limit(self.velocity, config.MAX_SPEED)
        self.position = euler_integrate(self.position, self.velocity)

    # ── boids ──────────────────────────────────────────────────────────────────
    def _apply_boids(self, neighbors, relay_pos, environment, shield_force=None):
        sep   = self._separation(neighbors)
        aln   = self._alignment(neighbors)
        coh   = self._cohesion(neighbors)
        rel   = (seek(self.position, relay_pos, self.velocity, config.MAX_SPEED, config.MAX_FORCE)
                 if relay_pos is not None else np.zeros(2))
        env_f = environment.repulsion_force(self.position) if environment else np.zeros(2)

        w = config.WEIGHTS
        s = self.signal
        if s > config.SIGNAL_TIER_HIGH:
            w_sep, w_aln, w_coh, w_rel = w["separation"], w["alignment"], w["cohesion"], w["relay"]
            wander = np.zeros(2)
        elif s > config.SIGNAL_TIER_LOW:
            w_sep, w_aln, w_coh, w_rel = w["separation"], w["alignment"]*0.5, w["cohesion"]*0.5, 0.0
            wander = np.zeros(2)
            self.velocity *= 0.97
        else:
            w_sep, w_aln, w_coh, w_rel = w["separation"], w["alignment"], w["cohesion"], 0.0
            wander = np.random.uniform(-1, 1, 2) * 0.08

        force = sep*w_sep + aln*w_aln + coh*w_coh + rel*w_rel + env_f + wander
        self.last_forces = {
            "sep": sep*w_sep, "aln": aln*w_aln, "coh": coh*w_coh,
            "rel": rel*w_rel, "env": env_f,
        }
        self.last_neighbor_positions = [nb.position.copy() for nb in neighbors]

        max_speed = config.MAX_SPEED * self.speed_mult
        self.velocity += limit(force, config.MAX_FORCE)
        self.velocity  = limit(self.velocity, max_speed)

    def _separation(self, neighbors):
        steer = np.zeros(2)
        count = 0
        for nb in neighbors:
            diff = self.position - nb.position
            d    = np.linalg.norm(diff)
            if d < config.SEPARATION_RADIUS:
                if d < 0.5:
                    diff = np.random.uniform(-1, 1, 2)
                    d    = max(np.linalg.norm(diff), 1e-4)
                steer += normalize(diff) / d
                count += 1
        if count:
            steer /= count
        return limit(steer, config.MAX_FORCE)

    def _alignment(self, neighbors):
        if not neighbors:
            return np.zeros(2)
        avg = np.mean([nb.velocity for nb in neighbors], axis=0)
        return limit(limit(avg, config.MAX_SPEED) - self.velocity, config.MAX_FORCE)

    def _cohesion(self, neighbors):
        if not neighbors:
            return np.zeros(2)
        avg = np.mean([nb.position for nb in neighbors], axis=0)
        return seek(self.position, avg, self.velocity, config.MAX_SPEED, config.MAX_FORCE)

    # ── reflex layer (always runs last, integrates position) ───────────────────
    def _finalize_reflex(self, context, environment):
        self.active_reflexes = set()
        W, H = config.WORLD_SIZE

        # GEOFENCE
        margin = config.GEOFENCE_MARGIN
        f_geo  = np.zeros(2)
        if self.position[0] < margin:
            f_geo[0] += config.GEOFENCE_FORCE * (margin - self.position[0]) / margin
        if self.position[0] > W - margin:
            f_geo[0] -= config.GEOFENCE_FORCE * (self.position[0] - (W - margin)) / margin
        if self.position[1] < margin:
            f_geo[1] += config.GEOFENCE_FORCE * (margin - self.position[1]) / margin
        if self.position[1] > H - margin:
            f_geo[1] -= config.GEOFENCE_FORCE * (self.position[1] - (H - margin)) / margin
        if np.linalg.norm(f_geo) > 0.001:
            self.velocity += f_geo
            self.active_reflexes.add("GEOFENCE")

        # COLLISION_AVOID
        all_agents = context.get("all_agents", [])
        for other in all_agents:
            if other is self or not other.alive:
                continue
            diff = self.position - other.position
            d    = np.linalg.norm(diff)
            if d < config.MIN_SAFE_DISTANCE and d > 1e-4:
                self.velocity += normalize(diff) * (config.MIN_SAFE_DISTANCE - d) * 0.5
                self.active_reflexes.add("COLLISION_AVOID")

        self.velocity = limit(self.velocity, config.MAX_SPEED * self.speed_mult)
        self.position = euler_integrate(self.position, self.velocity)
        # clamp to world
        self.position[0] = np.clip(self.position[0], 0, W)
        self.position[1] = np.clip(self.position[1], 0, H)


# ── helpers ────────────────────────────────────────────────────────────────────
def _is_iff_safe(agent, target):
    """Return True if target is friendly (same side) — never engage."""
    return target.side == agent.side


def pygame_time():
    try:
        import pygame
        return pygame.time.get_ticks() / 1000.0
    except Exception:
        return 0.0
