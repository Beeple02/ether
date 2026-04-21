"""Camera and projection for 3D rendering.

Two modes:
  SIM_3D=False  — positions are 2D numpy arrays (pygame Y-down convention).
                  world_to_screen() is a pass-through; depth_key() uses world-Y.
  SIM_3D=True   — positions are 3D (X-east, Y-north, Z-up right-handed).
                  Full oblique-isometric projection applied.

The split is intentional: RENDER_3D and SIM_3D are orthogonal flags.  The Camera
never modifies simulation data; it only converts world coords to screen pixels.
"""
import numpy as np
from . import config


class Camera:
    """Oblique-isometric camera for the drone swarm simulator.

    Default parameters match a classic 2:1 isometric feel:
      y_squeeze = 0.5   — world Y projects at half height on screen
      z_squeeze = 1.0   — world Z lifts screen position 1:1
    """

    def __init__(self, screen_size: tuple):
        sw, sh = screen_size
        self.sw = sw
        self.sh = sh
        # Oblique projection parameters
        self.y_squeeze: float = 0.5   # world-Y compression on screen
        self.z_squeeze: float = 1.0   # world-Z lift on screen
        self.scale: float     = 1.0   # world-units per pixel
        # Pivot: the world point that maps to the screen centre
        W, H = config.WORLD_SIZE
        self._pivot = np.array([W / 2.0, H / 2.0, 0.0])

    # ── core projection ───────────────────────────────────────────────────────

    def world_to_screen(self, pos) -> tuple:
        """Map a world position (2D or 3D) to (sx, sy) integer screen pixels."""
        if not config.SIM_3D or len(pos) == 2:
            # 2D pass-through: positions use pygame Y-down convention already
            return int(pos[0]), int(pos[1])
        # 3D oblique projection (X-east, Y-north, Z-up → screen)
        dx = pos[0] - self._pivot[0]
        dy = pos[1] - self._pivot[1]
        dz = pos[2] - self._pivot[2]
        sx = self.sw / 2 + dx / self.scale
        # Y-north increases away from viewer → subtract dy*y_squeeze (goes up on screen)
        # Z-up → subtract dz*z_squeeze (also lifts items up on screen)
        sy = self.sh / 2 - (dy * self.y_squeeze + dz * self.z_squeeze) / self.scale
        return int(sx), int(sy)

    def project_batch(self, positions: np.ndarray) -> np.ndarray:
        """Vectorised projection: (N, 2|3) world positions → (N, 2) screen ints."""
        if not config.SIM_3D or positions.shape[1] == 2:
            return positions[:, :2].astype(int)
        d   = positions - self._pivot
        sx  = self.sw / 2 + d[:, 0] / self.scale
        sy  = self.sh / 2 - (d[:, 1] * self.y_squeeze + d[:, 2] * self.z_squeeze) / self.scale
        return np.stack([sx, sy], axis=1).astype(int)

    # ── depth sort key ────────────────────────────────────────────────────────

    def depth_key(self, pos) -> float:
        """Painter's algorithm sort key: larger value = drawn later = on top."""
        if not config.SIM_3D or len(pos) == 2:
            # 2D: objects further south (larger y) are drawn first; near objects
            # drawn last so they appear on top.  We negate so sort ascending = far first.
            return float(pos[1])
        # 3D: larger world-Y = further from viewer; larger Z = higher altitude (also
        # further from viewer in oblique projection, but only slightly).
        return float(pos[1]) + float(pos[2]) * 0.001

    # ── inverse projection (for editor / HUD clicks) ──────────────────────────

    def screen_to_world_xy(self, sx: int, sy: int, z: float = 0.0) -> np.ndarray:
        """Unproject a screen pixel to world (x, y) at a given altitude z.

        Only meaningful when SIM_3D=True; in 2D mode returns the pixel as world coords.
        """
        if not config.SIM_3D:
            return np.array([float(sx), float(sy)])
        dx    = (sx - self.sw / 2) * self.scale
        sy_adj = (sy - self.sh / 2) * self.scale
        # sy_adj = -(dy * y_squeeze + z * z_squeeze)
        # → dy = (-sy_adj - z * z_squeeze) / y_squeeze
        dy = (-sy_adj - z * self.z_squeeze) / self.y_squeeze
        return self._pivot[:2] + np.array([dx, dy])

    # ── ground projection (for drop shadows) ─────────────────────────────────

    def ground_screen_pos(self, pos) -> tuple:
        """Screen position of the ground point directly below `pos`."""
        if not config.SIM_3D or len(pos) == 2:
            return int(pos[0]), int(pos[1])
        ground = np.array([pos[0], pos[1], 0.0])
        return self.world_to_screen(ground)

    def altitude(self, pos) -> float:
        """Return world-Z of pos (0 when SIM_3D=False or pos is 2D)."""
        if config.SIM_3D and len(pos) == 3:
            return float(pos[2])
        return 0.0

    # ── runtime parameter update ──────────────────────────────────────────────

    def on_resize(self, screen_size: tuple):
        self.sw, self.sh = screen_size
