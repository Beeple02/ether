"""Operator ↔ Relay uplink state model.

Feature-flagged via config.UPLINK_ENABLED:
  False (default) → quality always 1.0, state always GOOD, zero behavioural change.
  True            → quality can be degraded externally; commands are gated.

The relay continues local coordination regardless of uplink state — the uplink
only gates operator-issued phase/formation commands flowing DOWN to the relay.
"""
from . import config


class UplinkState:
    GOOD     = "GOOD"
    DEGRADED = "DEGRADED"
    LOST     = "LOST"

    def __init__(self):
        self.quality = 1.0          # 0.0–1.0
        self.state   = self.GOOD

    # ── state derivation ──────────────────────────────────────────────────────
    def _refresh_state(self):
        if self.quality >= config.SIGNAL_TIER_HIGH:
            self.state = self.GOOD
        elif self.quality >= config.SIGNAL_TIER_LOW:
            self.state = self.DEGRADED
        else:
            self.state = self.LOST

    # ── external setters (for future jamming / occlusion hooks) ───────────────
    def set_quality(self, q):
        self.quality = float(max(0.0, min(1.0, q)))
        self._refresh_state()

    # ── command gate ──────────────────────────────────────────────────────────
    def can_receive_command(self):
        """Return True if an operator-issued phase/formation command goes through.

        When UPLINK_ENABLED is False this always returns True (no change to existing
        behaviour).  When enabled, DEGRADED has a probabilistic drop rate.
        """
        if not config.UPLINK_ENABLED:
            return True
        if self.state == self.GOOD:
            return True
        if self.state == self.DEGRADED:
            import numpy as np
            return np.random.random() < self.quality
        return False   # LOST — no commands through

    # ── tick (placeholder for future latency / jamming simulation) ────────────
    def tick(self, dt):
        if not config.UPLINK_ENABLED:
            self.quality = 1.0
            self.state   = self.GOOD
