import numpy as np
from . import config
from .agent import Agent
from .pathfinder import PathFinder
from .physics import normalize


class Swarm:
    def __init__(self):
        self.relay         = None
        self.drones        = []
        self._waypoints    = []
        self._wp_index     = 0
        self._environment  = None
        self._pf           = PathFinder()
        self._relay_path   = []        # smoothed world-space waypoints
        self._path_idx     = 0         # index into relay_path
        self._path_target  = None      # last planned goal
        self._replan_in    = 0         # frames until next forced replan
        self._spawn()

    def _spawn(self):
        w, h = config.WORLD_SIZE
        self.relay  = Agent([w / 2, h / 2], role="relay")
        self.drones = [
            Agent(np.random.uniform([0, 0], [w, h]))
            for _ in range(config.NUM_DRONES)
        ]
        self._relay_path  = []
        self._path_idx    = 0
        self._path_target = None
        self._replan_in   = 0

    def set_environment(self, env):
        self._environment = env
        self._pf.rebuild(env)
        self._relay_path  = []
        self._path_target = None
        self._replan_in   = 0

    def set_waypoints(self, waypoints):
        self._waypoints   = list(waypoints)
        self._wp_index    = 0
        self._relay_path  = []
        self._path_target = None

    # ── main tick ─────────────────────────────────────────────────────────────
    def tick(self):
        self._update_relay()
        # Drones steer to a lookahead point on the relay's planned path so they
        # naturally route around the same obstacles the relay navigated around.
        follow_target = self._drone_follow_target()
        alive = [d for d in self.drones if d.alive]
        for drone in alive:
            drone.tick(self._neighbors(drone, alive), follow_target, self._environment)

    # ── relay navigation ──────────────────────────────────────────────────────
    def _update_relay(self):
        env = self._environment

        # Determine high-level goal
        raw_goal = self._relay_goal()
        if raw_goal is None:
            # No structured target — default orbit
            self.relay._move_relay(env)
            self._relay_path = []
            return

        # Replan when goal moved significantly or timer expired
        self._replan_in -= 1
        goal_shifted = (
            self._path_target is None or
            np.linalg.norm(raw_goal - self._path_target) > 50
        )
        if goal_shifted or self._replan_in <= 0:
            self._relay_path  = self._pf.find_path(self.relay.position, raw_goal)
            self._path_idx    = 0
            self._path_target = raw_goal.copy()
            self._replan_in   = 90   # ~1.5 s at 60 fps

        # Advance past waypoints we've reached
        path = self._relay_path
        while self._path_idx < len(path) - 1:
            if np.linalg.norm(path[self._path_idx] - self.relay.position) < 20:
                self._path_idx += 1
            else:
                break

        step = path[self._path_idx] if path else raw_goal
        self.relay.move_relay_toward(step, env)

        # Arrival at final goal (waypoint mode only)
        if self._waypoints and self._path_idx >= len(path) - 1:
            if np.linalg.norm(raw_goal - self.relay.position) < 15:
                self._wp_index    = (self._wp_index + 1) % len(self._waypoints)
                self._relay_path  = []
                self._path_target = None

    def _relay_goal(self):
        """
        Returns the world-space goal the relay should navigate toward,
        or None if the relay should use its default figure-eight.
        """
        if self._waypoints:
            return np.array(self._waypoints[self._wp_index], dtype=float)

        env = self._environment
        if env:
            targets = [z for z in env.zones if z.zone_type == "TARGET"]
            if targets:
                centers = [np.array([z.rect[0] + z.rect[2] / 2,
                                     z.rect[1] + z.rect[3] / 2])
                           for z in targets]
                com = np.mean(centers, axis=0)
                self.relay._t += 0.008
                return com + np.array([
                    np.cos(self.relay._t) * 70,
                    np.sin(self.relay._t * 0.7) * 35,
                ])
        return None

    # ── drone follow target ───────────────────────────────────────────────────
    def _drone_follow_target(self):
        """
        80 px lookahead on the relay's planned path.
        Drones steer here so they naturally take the same route around obstacles.
        Falls back to relay position when no path exists.
        """
        path = self._relay_path
        if not path or self._path_idx >= len(path):
            return self.relay.position.copy()

        LOOKAHEAD = 80.0
        pos       = self.relay.position.copy()
        remaining = LOOKAHEAD

        for i in range(self._path_idx, len(path)):
            wp   = path[i]
            diff = wp - pos
            d    = np.linalg.norm(diff)
            if d >= remaining:
                return pos + normalize(diff) * remaining
            remaining -= d
            pos = wp.copy()

        return path[-1].copy()

    # ── utils ─────────────────────────────────────────────────────────────────
    def _neighbors(self, agent, alive):
        r2 = config.PERCEPTION_RADIUS ** 2
        return [
            o for o in alive
            if o is not agent and np.sum((o.position - agent.position) ** 2) < r2
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
