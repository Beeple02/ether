"""3D renderer for the relay-swarm simulation.

Activated when config.RENDER_3D = True.  Subclasses the canonical Renderer
and overrides only the methods that benefit from depth ordering and altitude
cues.  The 2D renderer (renderer.py) is never modified.

Phase 1 behaviour (SIM_3D=False):
  - Positions are still 2D; camera.world_to_screen() is a pass-through.
  - Drones are depth-sorted by world-Y (painter's algorithm).
  - Drop shadows are invisible (altitude = 0).
  - HUD shows [3D VIEW] badge.
  - RunStats and CSV output are identical to 2D mode.

Phase 3+ behaviour (SIM_3D=True):
  - Positions become 3D; camera projects them through oblique isometric math.
  - Altitude drop-shadows are visible and scale with height.
  - Building vertical faces are rendered.
"""
import pygame
import numpy as np
from . import config
from .renderer import (
    Renderer, _signal_color, _draw_diamond, _draw_triangle,
    C_RELAY_RING, C_RELAY, C_BLD_FILL, C_BLD_BDR, C_HUD_VAL,
)
from .effects import SmokeCloud, EMPBlast, NetDeploy, Explosion

# Shadow palette
_SHADOW_COLOR = (0, 0, 0)


class Renderer3D(Renderer):
    """Renderer subclass that routes draw calls through a Camera instance."""

    def __init__(self, swarm, camera):
        super().__init__(swarm)
        self.camera = camera

    # ── helpers ───────────────────────────────────────────────────────────────

    def _w2s(self, pos) -> tuple:
        """Shorthand: world position → screen (sx, sy) int tuple."""
        return self.camera.world_to_screen(pos)

    def _draw_shadow(self, pos, size: int):
        """Draw an altitude drop-shadow ellipse under a drone."""
        alt      = self.camera.altitude(pos)
        if alt <= 0.0:
            return
        max_z    = config.WORLD_DEPTH
        alt_frac = min(1.0, alt / max_z)
        alpha    = int(110 * (1.0 - alt_frac * 0.7))
        if alpha < 8:
            return
        gsx, gsy  = self.camera.ground_screen_pos(pos)
        rx        = max(2, size + 1)
        ry        = max(1, int(rx * self.camera.y_squeeze))
        sz        = (rx * 2 + 4, ry * 2 + 4)
        surf      = pygame.Surface(sz, pygame.SRCALPHA)
        pygame.draw.ellipse(surf, (*_SHADOW_COLOR, alpha),
                            (0, 0, sz[0], sz[1]))
        self._screen.blit(surf, (gsx - rx - 2, gsy - ry - 2))

    # ── overridden draw methods ───────────────────────────────────────────────

    def _draw_drones(self):
        """Depth-sorted drone rendering with optional altitude shadows."""
        all_drones = list(self.swarm.drones) + list(self.swarm.enemy_drones)
        alive = [d for d in all_drones if d.alive and not d._is_relay]

        # Depth sort: far objects drawn first so near objects appear on top.
        # In 2D mode (SIM_3D=False) this sorts by world-Y, giving a slight
        # painter's-algorithm feel even without full 3D.
        alive.sort(key=lambda d: self.camera.depth_key(d.position))

        for d in alive:
            color = (_signal_color(d.type_color, d.signal)
                     if d.side == "player" else d.type_color)
            size  = d.type_size
            pos   = d.position
            af    = d.airframe
            sx, sy = self._w2s(pos)
            ipos   = (sx, sy)

            # altitude drop shadow (invisible in Phase 1 / SIM_3D=False)
            self._draw_shadow(pos, size)

            # mimicry ring
            if d._mimicry_active:
                pygame.draw.circle(self._screen, C_RELAY_RING, ipos, size + 6, 1)

            # airframe shape
            if af == "medium":
                _draw_diamond(self._screen, color, (sx, sy), size)
            elif af == "large":
                _draw_triangle(self._screen, color, (sx, sy), size, d.velocity)
            else:
                pygame.draw.circle(self._screen, color, ipos, size)

            # mesh_relay comm-radius overlay
            if d.drone_type == "mesh_relay" and config.MESH_SIGNAL_ENABLED:
                mr   = config.MESH_COMM_RADIUS
                ms   = pygame.Surface((mr * 2 + 2, mr * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(ms, (130, 195, 255, 14), (mr + 1, mr + 1), mr)
                pygame.draw.circle(ms, (130, 195, 255, 40), (mr + 1, mr + 1), mr, 1)
                self._screen.blit(ms, (sx - mr - 1, sy - mr - 1))

            # EMP arming pulse
            if d.drone_type == "emp" and getattr(d, "_emp_stage", None) == "arming":
                t_frac = min(1.0, d._emp_arming_t / config.EMP_ARMING_DELAY)
                ring_r = int(8 + t_frac * 14)
                ring_a = int(80 + t_frac * 160)
                rsurf  = pygame.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(rsurf, (175, 70, 220, ring_a),
                                   (ring_r + 2, ring_r + 2), ring_r, 2)
                self._screen.blit(rsurf, (sx - ring_r - 2, sy - ring_r - 2))

    def _draw_relay(self):
        sx, sy = self._w2s(self.swarm.relay.position)
        self._draw_shadow(self.swarm.relay.position, 12)
        pygame.draw.circle(self._screen, C_RELAY_RING, (sx, sy), 12)
        pygame.draw.circle(self._screen, C_RELAY,      (sx, sy), 10)

    def _draw_projectiles(self):
        for p in self.swarm.projectiles:
            if p.alive:
                sx, sy = self._w2s(p.position)
                pygame.draw.circle(self._screen, (255, 215, 55),
                                   (sx, sy), config.PROJECTILE_RADIUS)

    def _draw_effects(self):
        for fx in self.swarm.effects:
            if not fx.alive:
                continue
            r = max(1, int(getattr(fx, 'radius', 0)))
            a = int(getattr(fx, 'alpha', 0) * 255)
            if r < 1 or a < 1:
                continue
            sx, sy = self._w2s(fx.position)
            sz = r * 2 + 4

            if isinstance(fx, SmokeCloud):
                surf = pygame.Surface((sz, sz), pygame.SRCALPHA)
                pygame.draw.circle(surf, (155, 160, 165, a), (r + 2, r + 2), r)
                self._screen.blit(surf, (sx - r - 2, sy - r - 2))
            elif isinstance(fx, EMPBlast):
                surf = pygame.Surface((sz, sz), pygame.SRCALPHA)
                pygame.draw.circle(surf, (120, 60, 220, a),      (r + 2, r + 2), r, 3)
                pygame.draw.circle(surf, (200, 150, 255, a // 2), (r + 2, r + 2), max(1, r // 3))
                self._screen.blit(surf, (sx - r - 2, sy - r - 2))
            elif isinstance(fx, NetDeploy):
                surf = pygame.Surface((sz, sz), pygame.SRCALPHA)
                pygame.draw.circle(surf, (70, 215, 215, a), (r + 2, r + 2), r, 2)
                self._screen.blit(surf, (sx - r - 2, sy - r - 2))
            elif isinstance(fx, Explosion):
                col  = (*fx.color[:3], a)
                surf = pygame.Surface((sz, sz), pygame.SRCALPHA)
                pygame.draw.circle(surf, col, (r + 2, r + 2), r)
                self._screen.blit(surf, (sx - r - 2, sy - r - 2))

    def _draw_relay_lines(self):
        rsx, rsy = self._w2s(self.swarm.relay.position)
        for d in self.swarm.drones:
            if d.alive and np.linalg.norm(d.position - self.swarm.relay.position) < config.PERCEPTION_RADIUS:
                sx, sy = self._w2s(d.position)
                pygame.draw.line(self._screen, (38, 48, 72), (sx, sy), (rsx, rsy), 1)

    # ── HUD badge ─────────────────────────────────────────────────────────────

    def _draw_hud(self, fps, paused):
        super()._draw_hud(fps, paused)
        # 3D VIEW badge — top-right corner
        W, _ = config.WORLD_SIZE
        badge_text = "[3D VIEW]" if not config.SIM_3D else "[3D SIM+VIEW]"
        badge = self._font_sm.render(badge_text, True, (100, 200, 255))
        self._screen.blit(badge, (W - badge.get_width() - 8, 8))
