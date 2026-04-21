import json
import numpy as np
from .physics import normalize, segments_intersect, segment_rect_intersect

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
    DEFAULT_HEIGHT = 50.0

    def __init__(self, start, end, height=None):
        self.start  = np.array(start, dtype=float)
        self.end    = np.array(end,   dtype=float)
        self.height = float(height) if height is not None else self.DEFAULT_HEIGHT

    def to_dict(self):
        return {"start": self.start.tolist(), "end": self.end.tolist(),
                "height": self.height}

    @classmethod
    def from_dict(cls, d):
        return cls(d["start"], d["end"], d.get("height"))

    def closest_point(self, pos):
        s = self.start[:2]
        e = self.end[:2]
        p = pos[:2]
        ab = e - s
        t  = np.dot(p - s, ab) / (np.dot(ab, ab) + 1e-8)
        return s + np.clip(t, 0, 1) * ab


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
    DEFAULT_HEIGHT = 60.0

    def __init__(self, rect, height=None):
        self.rect   = tuple(float(v) for v in rect)
        self.height = float(height) if height is not None else self.DEFAULT_HEIGHT

    def to_dict(self):
        return {"rect": list(self.rect), "height": self.height}

    @classmethod
    def from_dict(cls, d):
        return cls(d["rect"], d.get("height"))

    def closest_point(self, pos):
        x, y, w, h = self.rect
        return np.array([np.clip(pos[0], x, x + w),
                         np.clip(pos[1], y, y + h)])

    def contains(self, pos):
        x, y, w, h = self.rect
        return x <= pos[0] <= x + w and y <= pos[1] <= y + h

    def contains_2d(self, pos2):
        x, y, w, h = self.rect
        return x <= pos2[0] <= x + w and y <= pos2[1] <= y + h


class Zone:
    def __init__(self, rect, zone_type="TARGET", z_min=None, z_max=None):
        self.rect      = tuple(float(v) for v in rect)
        self.zone_type = zone_type
        self.z_min     = float(z_min) if z_min is not None else 0.0
        self.z_max     = float(z_max) if z_max is not None else None  # None → WORLD_DEPTH

    def to_dict(self):
        d = {"rect": list(self.rect), "type": self.zone_type,
             "z_min": self.z_min}
        if self.z_max is not None:
            d["z_max"] = self.z_max
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["rect"], d["type"], d.get("z_min"), d.get("z_max"))

    def contains(self, pos):
        x, y, w, h = self.rect
        return x <= pos[0] <= x + w and y <= pos[1] <= y + h

    def center(self):
        x, y, w, h = self.rect
        return np.array([x + w / 2, y + h / 2])


class Base:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)

    def to_dict(self):
        return {"position": self.position.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["position"])


class EnemyBase:
    """Enemy spawn point. Spawns enemy drones after a configurable delay."""
    def __init__(self, position):
        self.position = np.array(position, dtype=float)

    def to_dict(self):
        return {"position": self.position.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["position"])


class Turret:
    """Hostile stationary emplacement."""
    def __init__(self, position, fire_rate=None, range=None):
        import swarm_sim.config as cfg
        self.position  = np.array(position, dtype=float)
        self.fire_rate = float(fire_rate) if fire_rate is not None else cfg.TURRET_FIRE_RATE
        self.range     = float(range)     if range     is not None else cfg.TURRET_RANGE
        # runtime state (not serialised)
        self._cooldown       = 0.0
        self._disabled_timer = 0.0   # seconds remaining disabled (EMP)

    @property
    def disabled(self):
        return self._disabled_timer > 0

    def to_dict(self):
        return {
            "position":  self.position.tolist(),
            "fire_rate": self.fire_rate,
            "range":     self.range,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["position"], d.get("fire_rate"), d.get("range"))


class Projectile:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.alive    = True


class Environment:
    def __init__(self):
        self.walls             = []
        self.trees             = []
        self.buildings         = []
        self.zones             = []
        self.waypoints         = []
        self.base              = None
        self.turrets           = []
        self.enemy_base        = None
        self.enemy_swarm_config = None  # per-scenario override; None → use config defaults

    def repulsion_force(self, pos):
        import swarm_sim.config as cfg
        # All geometry is ground-based (XY plane), so work in 2D regardless of
        # pos dimensionality and zero-pad the result for 3D callers.
        pos2   = pos[:2]
        force2 = np.zeros(2)
        half_r = cfg.PERCEPTION_RADIUS / 2

        for wall in self.walls:
            cp   = wall.closest_point(pos2)
            diff = pos2 - cp
            d    = np.linalg.norm(diff)
            if 0 < d < half_r:
                force2 += normalize(diff) * (half_r - d) / half_r

        for tree in self.trees:
            diff = pos2 - tree.position[:2]
            d    = np.linalg.norm(diff)
            eff  = tree.radius + half_r * 0.6
            if d < eff:
                if d < 0.5:
                    diff = np.random.uniform(-1, 1, 2)
                    d    = max(np.linalg.norm(diff), 1e-4)
                force2 += normalize(diff) * (eff - d) / eff * 2.5

        for bld in self.buildings:
            cp   = bld.closest_point(pos2)
            diff = pos2 - cp
            d    = np.linalg.norm(diff)
            if d < half_r:
                if d < 0.5:
                    diff = np.random.uniform(-1, 1, 2)
                    d    = max(np.linalg.norm(diff), 1e-4)
                force2 += normalize(diff) * (half_r - d) / half_r * 2.5

        for zone in self.zones:
            strength = ZONE_REPULSION.get(zone.zone_type, 0)
            if strength == 0:
                continue
            x, y, w, h = zone.rect
            cx, cy = x + w / 2, y + h / 2
            diff   = pos2 - np.array([cx, cy])
            d      = np.linalg.norm(diff)
            diag   = (w ** 2 + h ** 2) ** 0.5 / 2
            if d < diag + half_r:
                force2 += normalize(diff) * strength * max(0, (diag + half_r - d) / (diag + half_r))

        if len(pos) == 3:
            return np.array([force2[0], force2[1], 0.0])
        return force2

    def has_line_of_sight(self, a, b, smoke_clouds=None):
        """Blocked by walls, buildings, NOFLYZONE, and optionally smoke clouds.

        When SIM_3D=True and both positions are 3D, building/wall height is
        considered: if both endpoints are above the obstacle's height, the
        obstacle does not block LOS (drones can fly over it).
        The 2D callers pass 2-element arrays and always use the original logic.
        """
        from . import config as _cfg
        use_3d = _cfg.SIM_3D and len(a) == 3 and len(b) == 3
        a2 = a[:2]
        b2 = b[:2]

        for wall in self.walls:
            ws = wall.start[:2] if len(wall.start) == 3 else wall.start
            we = wall.end[:2]   if len(wall.end)   == 3 else wall.end
            if segments_intersect(a2, b2, ws, we):
                if use_3d:
                    # Skip block if both endpoints are above wall height
                    if a[2] > wall.height and b[2] > wall.height:
                        continue
                return False

        for bld in self.buildings:
            if segment_rect_intersect(a2, b2, bld.rect):
                if use_3d:
                    if a[2] > bld.height and b[2] > bld.height:
                        continue
                return False

        for zone in self.zones:
            if zone.zone_type == "NOFLYZONE":
                if segment_rect_intersect(a2, b2, zone.rect):
                    return False

        if smoke_clouds:
            for sc in smoke_clouds:
                if sc.alive and sc.blocks_los(a, b):
                    return False
        return True

    def save(self, path):
        import os
        from . import config as _cfg
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        W, H = _cfg.WORLD_SIZE
        data = {
            "meta":      {"version": 2, "units": "world_units"},
            "world":     {"width": W, "height": H, "depth": _cfg.WORLD_DEPTH},
            "walls":     [w.to_dict() for w in self.walls],
            "trees":     [t.to_dict() for t in self.trees],
            "buildings": [b.to_dict() for b in self.buildings],
            "zones":     [z.to_dict() for z in self.zones],
            "waypoints": [list(p)     for p in self.waypoints],
            "base":      self.base.to_dict()       if self.base       else None,
            "turrets":   [t.to_dict() for t in self.turrets],
            "enemy_base":        self.enemy_base.to_dict() if self.enemy_base else None,
            "enemy_swarm":       self.enemy_swarm_config,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.walls      = [Wall.from_dict(w)      for w in data.get("walls",      [])]
        self.trees      = [Tree.from_dict(t)      for t in data.get("trees",      [])]
        self.buildings  = [Building.from_dict(b)  for b in data.get("buildings",  [])]
        self.zones      = [Zone.from_dict(z)      for z in data.get("zones",      [])]
        self.waypoints  = [tuple(p)               for p in data.get("waypoints",  [])]
        b_data          = data.get("base")
        self.base       = Base.from_dict(b_data)          if b_data else None
        eb_data               = data.get("enemy_base")
        self.enemy_base       = EnemyBase.from_dict(eb_data) if eb_data else None
        self.enemy_swarm_config = data.get("enemy_swarm")   # None if not present
        self.turrets          = [Turret.from_dict(t) for t in data.get("turrets", [])]
