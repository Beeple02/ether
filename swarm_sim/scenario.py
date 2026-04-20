"""Scenario JSON format: environment reference + swarm/mission configuration."""
import json
import os
from . import config


class Scenario:
    """Wraps a full scenario: referenced environment file + all sim config overrides."""

    def __init__(self):
        self.environment_path    = None
        self.swarm_size          = config.NUM_DRONES
        self.loadout             = dict(config.LOADOUT)
        self.weights             = dict(config.WEIGHTS)
        self.mission_phases      = list(config.MISSION_PHASES)
        self.convergence_enabled = config.CONVERGENCE_ENABLED
        self.enemy_swarm         = None  # None → use config / env defaults

    # ── serialisation ─────────────────────────────────────────────────────────
    def to_dict(self):
        d = {
            "swarm_size":          self.swarm_size,
            "loadout":             self.loadout,
            "weights":             self.weights,
            "mission_phases":      self.mission_phases,
            "convergence_enabled": self.convergence_enabled,
        }
        if self.environment_path is not None:
            d["environment"] = self.environment_path
        if self.enemy_swarm is not None:
            d["enemy_swarm"] = self.enemy_swarm
        return d

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_environment(cls, env, env_path=None):
        """Snapshot current config + an environment into a Scenario."""
        s = cls()
        s.environment_path = env_path
        s.enemy_swarm      = env.enemy_swarm_config if env else None
        return s

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        s = cls()
        s.environment_path    = data.get("environment")
        s.swarm_size          = int(data.get("swarm_size",  config.NUM_DRONES))
        s.loadout             = data.get("loadout",         dict(config.LOADOUT))
        s.weights             = data.get("weights",         dict(config.WEIGHTS))
        s.mission_phases      = data.get("mission_phases",  list(config.MISSION_PHASES))
        s.convergence_enabled = bool(data.get("convergence_enabled",
                                              config.CONVERGENCE_ENABLED))
        s.enemy_swarm         = data.get("enemy_swarm")
        return s

    # ── apply ─────────────────────────────────────────────────────────────────
    def apply_config(self):
        """Push scenario overrides into the live config module."""
        config.NUM_DRONES          = self.swarm_size
        # Filter to known keys only; unknown types are silently dropped
        config.LOADOUT             = {k: float(v) for k, v in self.loadout.items()
                                      if k in config.DRONE_TYPES}
        config.WEIGHTS             = {k: float(v) for k, v in self.weights.items()
                                      if k in config.WEIGHTS}
        config.MISSION_PHASES      = list(self.mission_phases)
        config.CONVERGENCE_ENABLED = self.convergence_enabled

    def load_environment(self):
        """Load the referenced environment file, applying enemy_swarm override.
        Returns the Environment or None if no path is set."""
        if not self.environment_path:
            return None
        from .environment import Environment
        env = Environment()
        env.load(self.environment_path)
        if self.enemy_swarm is not None:
            env.enemy_swarm_config = self.enemy_swarm
        return env
