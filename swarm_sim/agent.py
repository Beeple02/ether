import numpy as np
from . import config
from .physics import normalize, limit, seek, euler_integrate


class Agent:
    def __init__(self, position, role="drone"):
        self.position = np.array(position, dtype=float)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.role = role
        self.alive = True
        self._t = np.random.uniform(0, 2 * np.pi)

    def tick(self, neighbors, relay_pos, environment=None):
        if not self.alive:
            return
        if self.role == "relay":
            self._move_relay()
            return
        self._apply_boids(neighbors, relay_pos, environment)

    def _move_relay(self, waypoints=None):
        """Default figure-eight orbit. Overridden externally when waypoints exist."""
        self._t += 0.01
        cx, cy = config.WORLD_SIZE[0] / 2, config.WORLD_SIZE[1] / 2
        r = min(config.WORLD_SIZE) * 0.3
        nx = cx + r * np.sin(self._t)
        ny = cy + r * np.sin(self._t * 2) * 0.5
        target = np.array([nx, ny])
        self.velocity = limit(target - self.position, config.MAX_SPEED)
        self.position = euler_integrate(self.position, self.velocity)

    def move_relay_toward(self, target):
        desired = normalize(target - self.position) * config.MAX_SPEED
        self.velocity = limit(desired, config.MAX_SPEED)
        self.position = euler_integrate(self.position, self.velocity)

    def _apply_boids(self, neighbors, relay_pos, environment):
        sep = self._separation(neighbors)
        aln = self._alignment(neighbors)
        coh = self._cohesion(neighbors)
        rel = seek(
            self.position, relay_pos, self.velocity,
            config.MAX_SPEED, config.MAX_FORCE
        )

        w = config.WEIGHTS
        force = (
            sep * w["separation"] +
            aln * w["alignment"] +
            coh * w["cohesion"] +
            rel * w["relay"]
        )

        if environment:
            force += environment.repulsion_force(self.position)

        self.velocity += limit(force, config.MAX_FORCE)
        self.velocity = limit(self.velocity, config.MAX_SPEED)
        self.position = euler_integrate(self.position, self.velocity)
        self._wrap()

    def _separation(self, neighbors):
        steer = np.zeros(2)
        count = 0
        for nb in neighbors:
            diff = self.position - nb.position
            d = np.linalg.norm(diff)
            if 0 < d < config.SEPARATION_RADIUS:
                steer += normalize(diff) / d
                count += 1
        if count > 0:
            steer /= count
        return limit(steer, config.MAX_FORCE)

    def _alignment(self, neighbors):
        if not neighbors:
            return np.zeros(2)
        avg_vel = np.mean([nb.velocity for nb in neighbors], axis=0)
        desired = limit(avg_vel, config.MAX_SPEED)
        steer = desired - self.velocity
        return limit(steer, config.MAX_FORCE)

    def _cohesion(self, neighbors):
        if not neighbors:
            return np.zeros(2)
        avg_pos = np.mean([nb.position for nb in neighbors], axis=0)
        return seek(self.position, avg_pos, self.velocity, config.MAX_SPEED, config.MAX_FORCE)

    def _wrap(self):
        w, h = config.WORLD_SIZE
        self.position[0] %= w
        self.position[1] %= h
