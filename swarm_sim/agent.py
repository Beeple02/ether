import numpy as np
from . import config
from .physics import normalize, limit, seek, euler_integrate


class Agent:
    def __init__(self, position, role="drone"):
        self.position  = np.array(position, dtype=float)
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity  = np.array([np.cos(angle), np.sin(angle)]) * np.random.uniform(0.5, 1.5)
        self.role      = role
        self.alive     = True
        self._t        = np.random.uniform(0, 2 * np.pi)
        # populated each tick for debug overlay
        self.last_forces             = {}
        self.last_neighbor_positions = []

    def tick(self, neighbors, relay_pos, environment=None):
        if not self.alive:
            return
        if self.role == "relay":
            self._move_relay(environment)
            return
        self._apply_boids(neighbors, relay_pos, environment)

    def _move_relay(self, environment=None):
        """Figure-eight with obstacle avoidance. Called by Swarm when no waypoints/targets."""
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
        """Steer relay toward a point, avoiding obstacles."""
        desired = normalize(target - self.position) * config.MAX_SPEED
        self.velocity = limit(desired, config.MAX_SPEED)
        if environment:
            rep = environment.repulsion_force(self.position)
            self.velocity += limit(rep * 3.0, config.MAX_SPEED * 0.7)
        self.velocity = limit(self.velocity, config.MAX_SPEED)
        self.position = euler_integrate(self.position, self.velocity)

    def _apply_boids(self, neighbors, relay_pos, environment):
        sep  = self._separation(neighbors)
        aln  = self._alignment(neighbors)
        coh  = self._cohesion(neighbors)
        rel  = seek(self.position, relay_pos, self.velocity, config.MAX_SPEED, config.MAX_FORCE)
        env_f = environment.repulsion_force(self.position) if environment else np.zeros(2)

        w = config.WEIGHTS
        force = (
            sep * w["separation"] +
            aln * w["alignment"] +
            coh * w["cohesion"] +
            rel * w["relay"] +
            env_f
        )

        self.last_forces = {
            "sep": sep * w["separation"],
            "aln": aln * w["alignment"],
            "coh": coh * w["cohesion"],
            "rel": rel * w["relay"],
            "env": env_f,
        }
        self.last_neighbor_positions = [nb.position.copy() for nb in neighbors]

        self.velocity += limit(force, config.MAX_FORCE)
        self.velocity  = limit(self.velocity, config.MAX_SPEED)
        self.position  = euler_integrate(self.position, self.velocity)
        self._wrap()

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

    def _wrap(self):
        w, h = config.WORLD_SIZE
        self.position[0] %= w
        self.position[1] %= h
