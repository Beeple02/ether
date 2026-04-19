"""Transient gameplay and visual effects."""
import numpy as np


class SmokeCloud:
    def __init__(self, position, radius=80, persist=6.0):
        self.position = np.array(position, dtype=float)
        self.radius   = float(radius)
        self.ttl      = float(persist)
        self.alive    = True

    def tick(self, dt):
        self.ttl -= dt
        if self.ttl <= 0:
            self.alive = False

    @property
    def alpha(self):
        """0‥1 opacity — fades in then out."""
        return max(0.0, min(1.0, self.ttl / 1.0)) * 0.45

    def blocks_los(self, a, b):
        """Return True if segment a→b passes within self.radius of center."""
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        ab = b - a
        denom = np.dot(ab, ab)
        if denom < 1e-8:
            return np.linalg.norm(a - self.position) < self.radius
        t  = np.clip(np.dot(self.position - a, ab) / denom, 0, 1)
        cp = a + t * ab
        return float(np.linalg.norm(cp - self.position)) < self.radius


class EMPBlast:
    DURATION = 1.5  # seconds for visual expansion

    def __init__(self, position, effect_radius=400):
        self.position      = np.array(position, dtype=float)
        self.effect_radius = float(effect_radius)
        self.radius        = 0.0          # visual — grows to effect_radius
        self.ttl           = self.DURATION
        self.alive         = True

    def tick(self, dt):
        self.ttl -= dt
        t = max(0.0, 1.0 - self.ttl / self.DURATION)  # 0→1
        self.radius = self.effect_radius * t
        if self.ttl <= 0:
            self.alive = False

    @property
    def alpha(self):
        return max(0.0, self.ttl / self.DURATION) * 0.5


class NetDeploy:
    DURATION = 0.5

    def __init__(self, position, effect_radius=60):
        self.position      = np.array(position, dtype=float)
        self.effect_radius = float(effect_radius)
        self.radius        = 0.0
        self.ttl           = self.DURATION
        self.alive         = True

    def tick(self, dt):
        self.ttl -= dt
        t = max(0.0, 1.0 - self.ttl / self.DURATION)
        self.radius = self.effect_radius * t
        if self.ttl <= 0:
            self.alive = False

    @property
    def alpha(self):
        return max(0.0, self.ttl / self.DURATION) * 0.7


class Explosion:
    DURATION = 0.35

    def __init__(self, position, max_radius=18, color=(255, 140, 40)):
        self.position   = np.array(position, dtype=float)
        self.max_radius = float(max_radius)
        self.radius     = 0.0
        self.color      = color
        self.ttl        = self.DURATION
        self.alive      = True

    def tick(self, dt):
        self.ttl -= dt
        t = max(0.0, 1.0 - self.ttl / self.DURATION)
        self.radius = self.max_radius * t
        if self.ttl <= 0:
            self.alive = False

    @property
    def alpha(self):
        return max(0.0, self.ttl / self.DURATION) * 0.85
