"""3D renderer for the relay-swarm simulation.

Activated when config.RENDER_3D = True.  Subclasses the canonical Renderer
and overrides the draw methods that benefit from camera projection, depth
ordering, and altitude cues.  The 2D renderer (renderer.py) is never
modified — it remains the canonical 2D path.

Phase 1 behaviour (SIM_3D=False):
  - Positions are still 2D; camera.world_to_screen() is a pass-through.
  - Drones are depth-sorted by world-Y.
  - Drop shadows are invisible (altitude = 0).
  - HUD shows [3D VIEW] badge.

Phase 3+ behaviour (SIM_3D=True):
  - Positions become 3D; camera projects them through oblique isometric math.
  - Ground grid lines show the tilted ground plane.
  - Buildings render as 3D volumes (top face + two visible side faces).
  - Walls render as tall quads between the ground and wall.height.
  - Trees show a trunk and foliage lifted to canopy height.
  - Altitude drop-shadows separate from drones by world-Z.
  - HUD shows [3D SIM+VIEW] badge.
"""
import pygame
import numpy as np
from . import config
from .renderer import (
    Renderer, _signal_color, _draw_diamond, _draw_triangle, _draw_pentagon,
    C_BG, C_RELAY_RING, C_RELAY, C_BLD_FILL, C_BLD_BDR, C_BLD_WIN,
    C_HUD_VAL, C_WALL, C_WP,
    C_TREE_OUT, C_TREE_MID, C_TREE_HI, C_TREE_TRUNK,
    C_LINE,
)
from .environment import ZONE_COLORS
from .effects import SmokeCloud, EMPBlast, NetDeploy, Explosion

# Ground grid + shadow palette
_SHADOW_COLOR = (0, 0, 0)
_GRID_COLOR   = (26, 32, 48)
_GRID_STEP    = 80   # world units between grid lines
_HORIZON_COL  = (24, 28, 42)


class Renderer3D(Renderer):
    """Renderer subclass that routes draw calls through a Camera instance."""

    def __init__(self, swarm, camera):
        super().__init__(swarm)
        self.camera = camera

    # ── helpers ───────────────────────────────────────────────────────────────

    def _w2s(self, pos) -> tuple:
        """Shorthand: world position → screen (sx, sy) int tuple."""
        return self.camera.world_to_screen(pos)

    def _g2s(self, x, y) -> tuple:
        """Ground (z=0) world (x, y) → screen (sx, sy)."""
        return self.camera.world_to_screen((x, y, 0.0))

    def _t2s(self, x, y, z) -> tuple:
        """Explicit 3D world point → screen."""
        return self.camera.world_to_screen((x, y, z))

    def _draw_shadow(self, pos, size: int):
        """Draw an altitude drop-shadow ellipse under a drone."""
        alt = self.camera.altitude(pos)
        if alt <= 0.0:
            return
        max_z    = config.WORLD_DEPTH
        alt_frac = min(1.0, alt / max_z)
        alpha    = int(110 * (1.0 - alt_frac * 0.5))
        if alpha < 8:
            return
        gsx, gsy = self.camera.ground_screen_pos(pos)
        rx = max(2, size + 1)
        ry = max(1, int(rx * self.camera.y_squeeze))
        sz = (rx * 2 + 4, ry * 2 + 4)
        surf = pygame.Surface(sz, pygame.SRCALPHA)
        pygame.draw.ellipse(surf, (*_SHADOW_COLOR, alpha),
                            (0, 0, sz[0], sz[1]))
        self._screen.blit(surf, (gsx - rx - 2, gsy - ry - 2))

    # ── override top-level draw to inject ground grid before env ─────────────

    def draw(self, fps, paused):
        s = self._screen
        s.fill(C_BG)
        self._draw_ground_grid()
        env = self.swarm._environment
        if env:
            self._draw_env(env)
        if self.debug_mode:
            self._draw_debug(env)
        elif self.show_lines:
            self._draw_relay_lines()
        self._draw_effects()
        self._draw_drones()
        self._draw_relay()
        self._draw_projectiles()
        self._draw_hud(fps, paused)
        self._draw_type_legend()
        if self.debug_mode:
            self._draw_debug_legend()
        self._tick_end_overlay(1.0 / config.FPS)
        if self.help_mode:
            self._draw_help_overlay()

    # ── ground plane grid (big 3D visual cue) ────────────────────────────────

    def _draw_ground_grid(self):
        if not config.SIM_3D:
            return
        W, H = config.WORLD_SIZE
        # Outer boundary parallelogram (world rect at z=0)
        tl = self._g2s(0, 0)
        tr = self._g2s(W, 0)
        br = self._g2s(W, H)
        bl = self._g2s(0, H)
        # Fill the ground plane with a slightly lighter tone for contrast
        ground = pygame.Surface((self.camera.sw, self.camera.sh), pygame.SRCALPHA)
        pygame.draw.polygon(ground, (18, 22, 34, 200), [tl, tr, br, bl])
        self._screen.blit(ground, (0, 0))

        # Grid lines parallel to world-X (constant y)
        for y in range(0, H + 1, _GRID_STEP):
            a = self._g2s(0, y)
            b = self._g2s(W, y)
            pygame.draw.line(self._screen, _GRID_COLOR, a, b, 1)
        # Grid lines parallel to world-Y (constant x)
        for x in range(0, W + 1, _GRID_STEP):
            a = self._g2s(x, 0)
            b = self._g2s(x, H)
            pygame.draw.line(self._screen, _GRID_COLOR, a, b, 1)

        # Outline
        pygame.draw.polygon(self._screen, (48, 60, 90), [tl, tr, br, bl], 1)

    # ── environment draw (camera-projected) ──────────────────────────────────

    def _draw_env(self, env):
        # Depth-order: zones first (ground paint), then buildings/trees/walls
        # (volumes) sorted back-to-front by world-Y, then overlays.
        for z in env.zones:
            self._zone(z)

        # Collect upright volumes so we can depth-sort them.
        volumes = []
        for b in env.buildings:
            x, y, w, h = b.rect
            cy = y + h / 2.0
            volumes.append((cy, "building", b))
        for t in env.trees:
            volumes.append((t.position[1], "tree", t))
        for w in env.walls:
            cy = (w.start[1] + w.end[1]) / 2.0
            volumes.append((cy, "wall", w))
        # Painter's algorithm: far (small Y) first, near (large Y) last
        volumes.sort(key=lambda v: v[0])
        for _, kind, obj in volumes:
            if kind == "building":
                self._building(obj)
            elif kind == "tree":
                self._tree(obj)
            elif kind == "wall":
                self._wall(obj)

        for i, wp in enumerate(env.waypoints):
            self._waypoint(wp, i + 1)
        for t in env.turrets:
            self._turret(t)
        if env.enemy_base:
            self._enemy_base(env.enemy_base)

    def _zone(self, zone):
        x, y, w, h = zone.rect
        col = ZONE_COLORS.get(zone.zone_type, (128, 128, 128, 55))
        pts = [self._g2s(x, y), self._g2s(x + w, y),
               self._g2s(x + w, y + h), self._g2s(x, y + h)]
        # Fill parallelogram on a per-area surface for alpha
        surf = pygame.Surface((self.camera.sw, self.camera.sh), pygame.SRCALPHA)
        pygame.draw.polygon(surf, col, pts)
        self._screen.blit(surf, (0, 0))
        border = (min(col[0] + 50, 255), min(col[1] + 50, 255), min(col[2] + 50, 255))
        pygame.draw.polygon(self._screen, border, pts, 1)

    def _building(self, bld):
        if not config.SIM_3D:
            # 2D fallback — mirror parent implementation
            super()._building(bld)
            return
        x, y, w, h = bld.rect
        bz = float(bld.height)
        # 8 corners of the box
        g_tl = self._t2s(x,     y,     0.0);  t_tl = self._t2s(x,     y,     bz)
        g_tr = self._t2s(x + w, y,     0.0);  t_tr = self._t2s(x + w, y,     bz)
        g_br = self._t2s(x + w, y + h, 0.0);  t_br = self._t2s(x + w, y + h, bz)
        g_bl = self._t2s(x,     y + h, 0.0);  t_bl = self._t2s(x,     y + h, bz)

        # Two "visible" side faces depending on oblique angle: south face (y+h)
        # and east face (x+w) are the near faces; both go UP on screen.
        south_face = [g_bl, g_br, t_br, t_bl]
        east_face  = [g_tr, g_br, t_br, t_tr]
        top_face   = [t_tl, t_tr, t_br, t_bl]

        # Shade by face orientation so the box reads as volume
        FILL    = C_BLD_FILL
        SIDE_S  = (max(FILL[0] - 6, 0),  max(FILL[1] - 6, 0),  max(FILL[2] - 6, 0))
        SIDE_E  = (max(FILL[0] - 14, 0), max(FILL[1] - 14, 0), max(FILL[2] - 14, 0))
        TOP     = (min(FILL[0] + 18, 255), min(FILL[1] + 18, 255), min(FILL[2] + 20, 255))

        pygame.draw.polygon(self._screen, SIDE_E, east_face)
        pygame.draw.polygon(self._screen, SIDE_S, south_face)
        pygame.draw.polygon(self._screen, TOP,    top_face)

        # Edge outlines
        for face in (south_face, east_face, top_face):
            pygame.draw.polygon(self._screen, C_BLD_BDR, face, 1)

        # Window dots on the south face (visual texture)
        if w > 28 and h > 20 and bz > 20:
            cols = min(8, max(1, int(w // 28)))
            rows = min(4, max(1, int(bz // 15)))
            for c in range(cols):
                for r in range(rows):
                    wx = x + (c + 0.5) * (w / cols)
                    wz = (r + 0.5) * (bz / rows)
                    wsp = self._t2s(wx, y + h, wz)
                    pygame.draw.circle(self._screen, C_BLD_WIN[:3], wsp, 1)

    def _wall(self, wall):
        if not config.SIM_3D:
            super()._wall(wall)
            return
        wz = float(getattr(wall, "height", 50.0))
        sx, sy = float(wall.start[0]), float(wall.start[1])
        ex, ey = float(wall.end[0]),   float(wall.end[1])
        g0 = self._t2s(sx, sy, 0.0)
        g1 = self._t2s(ex, ey, 0.0)
        t0 = self._t2s(sx, sy, wz)
        t1 = self._t2s(ex, ey, wz)
        # Filled quad side
        face = [g0, g1, t1, t0]
        fill = (max(C_WALL[0] - 50, 0), max(C_WALL[1] - 55, 0), max(C_WALL[2] - 30, 0))
        pygame.draw.polygon(self._screen, fill, face)
        pygame.draw.polygon(self._screen, C_WALL, face, 1)
        # Top edge thicker for readability
        pygame.draw.line(self._screen, C_WALL, t0, t1, 2)

    def _tree(self, tree):
        if not config.SIM_3D:
            super()._tree(tree)
            return
        tx, ty = float(tree.position[0]), float(tree.position[1])
        r = int(tree.radius)
        canopy_h = r * 1.6   # treat radius as rough canopy scale
        # Trunk line
        g = self._t2s(tx, ty, 0.0)
        cbot = self._t2s(tx, ty, canopy_h * 0.35)
        ctop = self._t2s(tx, ty, canopy_h)
        pygame.draw.line(self._screen, C_TREE_TRUNK, g, cbot, 3)
        # Foliage at canopy top (projected as ellipse to match ground squeeze)
        ry = max(2, int(r * (1.0 - 0.15)))   # slight vertical elongation
        rx = r
        sx, sy = ctop
        base = pygame.Surface((rx * 2 + 4, ry * 2 + 4), pygame.SRCALPHA)
        pygame.draw.ellipse(base, C_TREE_OUT, (0, 0, rx * 2, ry * 2))
        pygame.draw.ellipse(base, C_TREE_MID, (3, 3, rx * 2 - 6, ry * 2 - 6))
        self._screen.blit(base, (sx - rx - 2, sy - ry - 2))
        # Small highlight
        hi = (sx - rx // 3, sy - ry // 3)
        pygame.draw.circle(self._screen, C_TREE_HI, hi, max(r // 3, 2))

    def _waypoint(self, wp, idx):
        sx, sy = self._g2s(float(wp[0]), float(wp[1]))
        pygame.draw.circle(self._screen, C_WP, (sx, sy), 7, 2)
        txt = self._font.render(str(idx), True, C_WP)
        self._screen.blit(txt, (sx + 9, sy - 9))

    def _turret(self, turret):
        pos = turret.position
        tx, ty = float(pos[0]), float(pos[1])
        disabled = turret._disabled_timer > 0
        body_col = (70, 72, 88) if disabled else (195, 55, 40)
        ring_col = (55, 58, 75) if disabled else (230, 90, 65)
        # Faint range ring on the ground plane (ellipse, matches y-squeeze)
        r  = int(turret.range)
        ry = max(1, int(r * self.camera.y_squeeze))
        gsx, gsy = self._g2s(tx, ty)
        ring_surf = pygame.Surface((r * 2 + 2, ry * 2 + 2), pygame.SRCALPHA)
        pygame.draw.ellipse(ring_surf, (*ring_col, 18), (0, 0, r * 2, ry * 2), 1)
        self._screen.blit(ring_surf, (gsx - r, gsy - ry))
        # Small vertical mast (0 → ~22 world units)
        mast_top = self._t2s(tx, ty, 22.0)
        pygame.draw.line(self._screen, (50, 55, 72), (gsx, gsy), mast_top, 2)
        # Body sits on top of mast
        pygame.draw.circle(self._screen, body_col, mast_top, 6)
        pygame.draw.circle(self._screen, ring_col, mast_top, 6, 2)
        if disabled:
            mx, my = mast_top
            pygame.draw.line(self._screen, (120, 60, 220), (mx - 4, my - 4), (mx + 4, my + 4), 1)
            pygame.draw.line(self._screen, (120, 60, 220), (mx + 4, my - 4), (mx - 4, my + 4), 1)

    def _enemy_base(self, eb):
        pos = eb.position
        sx, sy = self._g2s(float(pos[0]), float(pos[1]))
        _draw_pentagon(self._screen, (150, 25, 25), (sx, sy), 13)
        _draw_pentagon(self._screen, (255, 60, 60), (sx, sy), 13, width=2)
        txt = self._font_sm.render("ENEMY BASE", True, (255, 80, 80))
        self._screen.blit(txt, (sx - txt.get_width() // 2, sy + 17))

    # ── overridden draw methods ───────────────────────────────────────────────

    def _draw_drones(self):
        """Depth-sorted drone rendering with altitude shadows."""
        all_drones = list(self.swarm.drones) + list(self.swarm.enemy_drones)
        alive = [d for d in all_drones if d.alive and not d._is_relay]

        alive.sort(key=lambda d: self.camera.depth_key(d.position))

        for d in alive:
            color = (_signal_color(d.type_color, d.signal)
                     if d.side == "player" else d.type_color)
            size  = d.type_size
            pos   = d.position
            af    = d.airframe
            sx, sy = self._w2s(pos)
            ipos   = (sx, sy)

            self._draw_shadow(pos, size)

            if d._mimicry_active:
                pygame.draw.circle(self._screen, C_RELAY_RING, ipos, size + 6, 1)

            if af == "medium":
                _draw_diamond(self._screen, color, (sx, sy), size)
            elif af == "large":
                _draw_triangle(self._screen, color, (sx, sy), size, d.velocity)
            else:
                pygame.draw.circle(self._screen, color, ipos, size)

            if d.drone_type == "mesh_relay" and config.MESH_SIGNAL_ENABLED:
                mr  = config.MESH_COMM_RADIUS
                mry = max(1, int(mr * self.camera.y_squeeze))
                gsx, gsy = self.camera.ground_screen_pos(pos)
                ms = pygame.Surface((mr * 2 + 2, mry * 2 + 2), pygame.SRCALPHA)
                pygame.draw.ellipse(ms, (130, 195, 255, 14), (0, 0, mr * 2, mry * 2))
                pygame.draw.ellipse(ms, (130, 195, 255, 40), (0, 0, mr * 2, mry * 2), 1)
                self._screen.blit(ms, (gsx - mr, gsy - mry))

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
                pygame.draw.line(self._screen, C_LINE, (sx, sy), (rsx, rsy), 1)

    # ── HUD badge ─────────────────────────────────────────────────────────────

    def _draw_hud(self, fps, paused):
        super()._draw_hud(fps, paused)
        W, _ = config.WORLD_SIZE
        badge_text = "[3D VIEW]" if not config.SIM_3D else "[3D SIM+VIEW]"
        badge = self._font_sm.render(badge_text, True, (100, 200, 255))
        self._screen.blit(badge, (W - badge.get_width() - 8, 8))
