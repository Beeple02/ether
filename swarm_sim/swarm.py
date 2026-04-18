import numpy as np
from . import config
from .agent import Agent


class Swarm:
    def __init__(self):
        self.relay = None
        self.drones = []
        self._waypoints = []
        self._wp_index = 0
        self._environment = None
        self._spawn()

    def _spawn(self):
        w, h = config.WORLD_SIZE
        self.relay = Agent(
            [w / 2, h / 2],
            role="relay"
        )
        self.drones = [
            Agent(np.random.uniform([0, 0], [w, h]))
            for _ in range(config.NUM_DRONES)
        ]

    def set_environment(self, env):
        self._environment = env

    def set_waypoints(self, waypoints):
        self._waypoints = waypoints
        self._wp_index = 0

    def tick(self):
        self._update_relay()
        relay_pos = self.relay.position.copy()
        alive = [d for d in self.drones if d.alive]
        for drone in alive:
            neighbors = self._neighbors(drone, alive)
            drone.tick(neighbors, relay_pos, self._environment)

    def _update_relay(self):
        if self._waypoints:
            target = np.array(self._waypoints[self._wp_index], dtype=float)
            if np.linalg.norm(target - self.relay.position) < 15:
                self._wp_index = (self._wp_index + 1) % len(self._waypoints)
            self.relay.move_relay_toward(target)
        else:
            self.relay.tick([], self.relay.position, None)

    def _neighbors(self, agent, alive):
        r2 = config.PERCEPTION_RADIUS ** 2
        return [
            other for other in alive
            if other is not agent and
            np.sum((other.position - agent.position) ** 2) < r2
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
