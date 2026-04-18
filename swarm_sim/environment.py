import json
import numpy as np
from .physics import normalize

ZONE_TYPES = ["TARGET", "NOFLYZONE"]
ZONE_COLORS = {
    "TARGET":    (40, 200, 100, 55),
    "NOFLYZONE": (210, 35,  35,  70),
}
ZONE_REPULSION = {
    "TARGET":    0.0,
    "NOFLYZONE": 3.0,
}


class Wall:
    def __init__(self, start, end):
        self.start = np.array(start, dtype=float)
        self.end   = np.array(end,   dtype=float)

    def to_dict(self):
        return {"start": self.start.tolist(), "end": self.end.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["start"], d["end"])

    def closest_point(self, pos):
        ab = self.end - self.start
        t  = np.dot(pos - self.start, ab) / (np.dot(ab, ab) + 1e-8)
        return self.start + np.clip(t, 0, 1) * ab


class Tree:
    def __init__(self, position, radius=18):
        self.position = np.array(position, dtype=float)
        self.radius   = float(radius)

    def to_dict(self):
        return {"position": self.position.tolist(), "radius": self.radius}

    @classmethod
    def from_dict(cls, d):
        return cls(d["position"], d.get("radius", 18))


class Building:
    """Rectangular building — acts as a hard obstacle."""
    def __init__(self, rect):
        self.rect = tuple(float(v) for v in rect)   # (x, y, w, h)

    def to_dict(self):
        return {"rect": list(self.rect)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["rect"])

    def closest_point(self, pos):
        x, y, w, h = self.rect
        return np.array([np.clip(pos[0], x, x + w),
                         np.clip(pos[1], y, y + h)])

    def contains(self, pos):
        x, y, w, h = self.rect
        return x <= pos[0] <= x + w and y <= pos[1] <= y + h


class Zone:
    """Semantic area — TARGET or NOFLYZONE."""
    def __init__(self, rect, zone_type="TARGET"):
        self.rect      = tuple(float(v) for v in rect)
        self.zone_type = zone_type

    def to_dict(self):
        return {"rect": list(self.rect), "type": self.zone_type}

    @classmethod
    def from_dict(cls, d):
        return cls(d["rect"], d["type"])

    def contains(self, pos):
        x, y, w, h = self.rect
        return x <= pos[0] <= x + w and y <= pos[1] <= y + h


class Environment:
    def __init__(self):
        self.walls     = []
        self.trees     = []
        self.buildings = []
        self.zones     = []
        self.waypoints = []

    def repulsion_force(self, pos):
        import swarm_sim.config as cfg
        force  = np.zeros(2)
        half_r = cfg.PERCEPTION_RADIUS / 2

        for wall in self.walls:
            cp   = wall.closest_point(pos)
            diff = pos - cp
            d    = np.linalg.norm(diff)
            if 0 < d < half_r:
                force += normalize(diff) * (half_r - d) / half_r

        for tree in self.trees:
            diff = pos - tree.position
            d    = np.linalg.norm(diff)
            eff  = tree.radius + half_r * 0.6
            if d < eff:
                if d < 0.5:
                    diff = np.random.uniform(-1, 1, 2)
                    d    = max(np.linalg.norm(diff), 1e-4)
                force += normalize(diff) * (eff - d) / eff * 2.5

        for bld in self.buildings:
            cp   = bld.closest_point(pos)
            diff = pos - cp
            d    = np.linalg.norm(diff)
            if d < half_r:
                if d < 0.5:
                    diff = np.random.uniform(-1, 1, 2)
                    d    = max(np.linalg.norm(diff), 1e-4)
                force += normalize(diff) * (half_r - d) / half_r * 2.5

        for zone in self.zones:
            strength = ZONE_REPULSION.get(zone.zone_type, 0)
            if strength == 0:
                continue
            x, y, w, h = zone.rect
            cx, cy = x + w / 2, y + h / 2
            diff   = pos - np.array([cx, cy])
            d      = np.linalg.norm(diff)
            diag   = (w ** 2 + h ** 2) ** 0.5 / 2
            if d < diag + half_r:
                force += normalize(diff) * strength * max(0, (diag + half_r - d) / (diag + half_r))

        return force

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "walls":     [w.to_dict() for w in self.walls],
            "trees":     [t.to_dict() for t in self.trees],
            "buildings": [b.to_dict() for b in self.buildings],
            "zones":     [z.to_dict() for z in self.zones],
            "waypoints": [list(p)     for p in self.waypoints],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.walls     = [Wall.from_dict(w)     for w in data.get("walls",     [])]
        self.trees     = [Tree.from_dict(t)     for t in data.get("trees",     [])]
        self.buildings = [Building.from_dict(b) for b in data.get("buildings", [])]
        self.zones     = [Zone.from_dict(z)     for z in data.get("zones",     [])]
        self.waypoints = [tuple(p)              for p in data.get("waypoints", [])]
