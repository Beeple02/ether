"""Mission FSM and Formation FSM."""
import numpy as np
from . import config


class MissionFSM:
    def __init__(self, phases=None):
        self._phases   = phases or config.MISSION_PHASES
        self._idx      = 0
        self.phase     = self._phases[0]
        self.phase_elapsed  = 0.0   # seconds in current phase
        self.total_elapsed  = 0.0

        # per-phase state
        self._emp_detonated          = False
        self._emp_detonation_elapsed = 0.0
        self.furthest_phase          = self.phase

    # ── advance ───────────────────────────────────────────────────────────────
    def advance(self):
        if self._idx < len(self._phases) - 1:
            self._idx        += 1
            self.phase        = self._phases[self._idx]
            self.phase_elapsed = 0.0
            _pi = config.MISSION_PHASES.index(self.phase) if self.phase in config.MISSION_PHASES else 0
            _fi = config.MISSION_PHASES.index(self.furthest_phase) if self.furthest_phase in config.MISSION_PHASES else 0
            if _pi > _fi:
                self.furthest_phase = self.phase
            return True
        return False

    def on_emp_detonated(self):
        self._emp_detonated          = True
        self._emp_detonation_elapsed = 0.0

    # ── tick ─────────────────────────────────────────────────────────────────
    def tick(self, dt, swarm):
        self.phase_elapsed += dt
        self.total_elapsed += dt

        if self.phase == "SUPPRESSION" and self._emp_detonated:
            self._emp_detonation_elapsed += dt
            if self._emp_detonation_elapsed >= 10.0:
                self.advance()
                return

        self._check_auto(dt, swarm)

    def _check_auto(self, dt, swarm):
        phase = self.phase

        if phase == "TRANSIT":
            wp = swarm._waypoints
            if wp and swarm.relay.alive:
                dist = float(np.linalg.norm(
                    swarm.relay.position - np.array(wp[0], dtype=float)
                ))
                if dist < 35:
                    self.advance()

        elif phase == "SATURATION":
            fast = [d for d in swarm.drones
                    if d.drone_type == "fast" and d.side == "player"]
            if fast:
                env = swarm._environment
                targets = [z for z in env.zones if z.zone_type == "TARGET"] if env else []
                if targets:
                    in_zone = sum(1 for d in fast if d.alive and
                                  any(t.contains(d.position) for t in targets))
                    dead    = sum(1 for d in fast if not d.alive)
                    total   = len(fast)
                    if total and (in_zone + dead) / total >= 0.5:
                        self.advance()

        elif phase == "PROSECUTION":
            heavy = [d for d in swarm.drones
                     if d.drone_type == "heavy" and d.side == "player"]
            if heavy:
                env = swarm._environment
                targets = [z for z in env.zones if z.zone_type == "TARGET"] if env else []
                all_done = all(
                    not d.alive or (targets and any(t.contains(d.position) for t in targets))
                    for d in heavy
                )
                if all_done:
                    self.advance()


class FormationFSM:
    def __init__(self):
        self.mode            = "COLUMN"
        self._override       = None    # None or a mode string
        self._override_timer = 0.0     # seconds left on override

    # ── called when mission phase changes ────────────────────────────────────
    def set_for_phase(self, phase):
        default = config.PHASE_FORMATION.get(phase, "DENSE")
        if self._override is None:
            self.mode = default

    # ── BUBBLE activates from threat detection (auto-expires) ─────────────
    def set_bubble(self, active):
        if active:
            self._override       = "BUBBLE"
            self._override_timer = 3.0
            self.mode            = "BUBBLE"

    # ── RECON-driven timed override (auto-expires like BUBBLE) ────────────
    def set_recon_override(self, mode, duration=2.0):
        """Temporarily force a formation mode from RECON threat reports.
        BUBBLE always takes priority; refreshes timer if same mode already active."""
        if self._override == "BUBBLE" and self._override_timer > 0:
            return  # BUBBLE is highest priority
        self._override       = mode
        self._override_timer = max(self._override_timer, duration)
        self.mode            = mode

    # ── operator manual override ──────────────────────────────────────────
    def manual_set(self, mode):
        self.mode      = mode
        self._override = mode

    # ── tick ─────────────────────────────────────────────────────────────
    def tick(self, dt, phase):
        if self._override_timer > 0:
            self._override_timer -= dt
            if self._override_timer <= 0:
                self._override = None
                self.set_for_phase(phase)
        elif self._override is None:
            self.set_for_phase(phase)
