import json
import numpy as np
from .physics import normalize

ZONE_TYPES = ["TARGET", "OBSTACLE", "NOFLYZONE"]
ZONE_COLORS = {
    "TARGET": (0, 200, 100, 60),
    "OBSTACLE": (200, 100, 0, 80),
    "NOFLYZONE": (220, 30, 30, 100),
}
ZONE_REPULSION = {
    "TARGET": 0.0,
    "OBSTACLE": 1.0,
    "NOFLYZONE": 3.0,
}


class Wall:
    def __init__(self, start, end):
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)

    def to_dict(self):
        return {"start": self.start.tolist(), "end": self.end.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["start"], d["end"])

    def closest_point(self, pos):
        ab = self.end - self.start
        ap = pos - self.start
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
        t = np.clip(t, 0, 1)
        return self.start + t * ab


class Zone:
    def __init__(self, rect, zone_type="OBSTACLE"):
        self.rect = rect  # (x, y, w, h)
        self.zone_type = zone_type

    def to_dict(self):
        return {"rect": list(self.rect), "type": self.zone_type}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(d["rect"]), d["type"])

    def contains(self, pos):
        x, y, w, h = self.rect
        return x <= pos[0] <= x + w and y <= pos[1] <= y + h


class Environment:
    def __init__(self):
        self.walls = []
        self.zones = []
        self.waypoints = []

    def repulsion_force(self, pos):
        force = np.zeros(2)
        half_r = __import__('swarm_sim.config', fromlist=['config']).PERCEPTION_RADIUS / 2

        for wall in self.walls:
            cp = wall.closest_point(pos)
            diff = pos - cp
            d = np.linalg.norm(diff)
            if 0 < d < half_r:
                force += normalize(diff) * (half_r - d) / half_r * 0.5

        for zone in self.zones:
            strength = ZONE_REPULSION.get(zone.zone_type, 0)
            if strength == 0:
                continue
            x, y, w, h = zone.rect
            cx, cy = x + w / 2, y + h / 2
            diff = pos - np.array([cx, cy])
            d = np.linalg.norm(diff)
            diag = (w ** 2 + h ** 2) ** 0.5 / 2
            if d < diag + half_r:
                force += normalize(diff) * strength * max(0, (diag + half_r - d) / (diag + half_r))

        return force

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "walls": [w.to_dict() for w in self.walls],
            "zones": [z.to_dict() for z in self.zones],
            "waypoints": [list(p) for p in self.waypoints],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.walls = [Wall.from_dict(w) for w in data.get("walls", [])]
        self.zones = [Zone.from_dict(z) for z in data.get("zones", [])]
        self.waypoints = [tuple(p) for p in data.get("waypoints", [])]
