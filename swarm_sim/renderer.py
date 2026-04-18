import pygame
import numpy as np
from . import config
from .environment import ZONE_COLORS

# ── palette ──────────────────────────────────────────────────────────────────
C_BG            = (14,  15,  24)
C_DRONE         = (205, 215, 230)
C_RELAY_RING    = (160, 120,  10)
C_RELAY         = (255, 220,  40)
C_LINE          = ( 38,  48,  72)
C_WALL          = (190, 150,  65)
C_WP            = ( 75, 185, 255)
C_HUD_BG        = ( 18,  20,  34, 210)
C_HUD_BORDER    = ( 50,  62,  98)
C_HUD_LABEL     = ( 95, 125, 165)
C_HUD_VAL       = (155, 215, 175)
C_TREE_OUTER    = ( 28,  85,  38)
C_TREE_MID      = ( 42, 125,  52)
C_TREE_HI       = ( 68, 155,  70)
C_TREE_TRUNK    = ( 75,  48,  18)
C_BLD_FILL      = ( 48,  50,  64)
C_BLD_BORDER    = ( 78,  82, 106)
C_BLD_WIN       = ( 88, 118, 162, 160)
C_SIDEBAR_BG    = ( 16,  18,  28, 230)
C_SIDEBAR_BDR   = ( 44,  54,  88)
C_BTN           = ( 30,  34,  52)
C_BTN_ACTIVE    = ( 55,  80, 140)
C_BTN_TEXT      = (185, 200, 220)
C_BTN_KEY       = (110, 160, 220)
C_SECTION       = ( 70,  82, 120)
C_COUNT_TEXT    = (120, 145, 175)

SIDEBAR_W = config.EDITOR_SIDEBAR_W


# ── Sim renderer ──────────────────────────────────────────────────────────────
class Renderer:
    def __init__(self, swarm):
        self.swarm      = swarm
        self.show_lines = False
        self._font      = None
        self._screen    = None

    def init(self, screen):
        self._screen = screen
        self._font   = pygame.font.SysFont("monospace", 14)

    def draw(self, fps, paused):
        s = self._screen
        s.fill(C_BG)

        env = self.swarm._environment
        if env:
            self._draw_env(env)

        if self.show_lines:
            self._draw_lines()

        self._draw_drones()
        self._draw_relay()
        self._draw_hud(fps, paused)

    # ── environment ──────────────────────────────────────────────────────────
    def _draw_env(self, env):
        for z  in env.zones:     self._draw_zone(z)
        for b  in env.buildings: self._draw_building(b)
        for t  in env.trees:     self._draw_tree(t)
        for w  in env.walls:     self._draw_wall(w)
        for i, wp in enumerate(env.waypoints):
            self._draw_waypoint(wp, i + 1)

    def _draw_zone(self, zone):
        x, y, w, h = (int(v) for v in zone.rect)
        col  = ZONE_COLORS.get(zone.zone_type, (128, 128, 128, 55))
        surf = pygame.Surface((max(w, 1), max(h, 1)), pygame.SRCALPHA)
        surf.fill(col)
        self._screen.blit(surf, (x, y))
        border = (min(col[0] + 50, 255), min(col[1] + 50, 255), min(col[2] + 50, 255))
        pygame.draw.rect(self._screen, border, (x, y, w, h), 1)

    def _draw_building(self, bld):
        x, y, w, h = (int(v) for v in bld.rect)
        pygame.draw.rect(self._screen, C_BLD_FILL, (x, y, w, h))
        pygame.draw.rect(self._screen, C_BLD_BORDER, (x, y, w, h), 2)
        # windows
        wsurf = pygame.Surface((max(w, 1), max(h, 1)), pygame.SRCALPHA)
        ww, wh, gx, gy = 6, 5, 9, 9
        for wx in range(gx, w - ww, ww + gx):
            for wy in range(gy, h - wh, wh + gy):
                pygame.draw.rect(wsurf, C_BLD_WIN, (wx, wy, ww, wh))
        self._screen.blit(wsurf, (x, y))

    def _draw_tree(self, tree):
        pos = tree.position.astype(int)
        r   = int(tree.radius)
        pygame.draw.circle(self._screen, C_TREE_OUTER, pos, r)
        pygame.draw.circle(self._screen, C_TREE_MID,   pos, max(r - 4, 2))
        hi = (pos[0] - r // 3, pos[1] - r // 3)
        pygame.draw.circle(self._screen, C_TREE_HI, hi, max(r // 3, 2))
        pygame.draw.circle(self._screen, C_TREE_TRUNK, pos, 3)

    def _draw_wall(self, wall):
        pygame.draw.line(self._screen, C_WALL,
                         wall.start.astype(int), wall.end.astype(int), 3)

    def _draw_waypoint(self, wp, idx):
        px, py = int(wp[0]), int(wp[1])
        pygame.draw.circle(self._screen, C_WP, (px, py), 7, 2)
        txt = self._font.render(str(idx), True, C_WP)
        self._screen.blit(txt, (px + 9, py - 9))

    # ── agents ───────────────────────────────────────────────────────────────
    def _draw_lines(self):
        rp = self.swarm.relay.position
        for d in self.swarm.drones:
            if d.alive and np.linalg.norm(d.position - rp) < config.PERCEPTION_RADIUS:
                pygame.draw.line(self._screen, C_LINE,
                                 d.position.astype(int), rp.astype(int), 1)

    def _draw_drones(self):
        for d in self.swarm.drones:
            if d.alive:
                pygame.draw.circle(self._screen, C_DRONE, d.position.astype(int), 4)

    def _draw_relay(self):
        rp = self.swarm.relay.position.astype(int)
        pygame.draw.circle(self._screen, C_RELAY_RING, rp, 12)
        pygame.draw.circle(self._screen, C_RELAY,      rp, 10)

    # ── HUD ──────────────────────────────────────────────────────────────────
    def _draw_hud(self, fps, paused):
        w   = config.WEIGHTS
        rows = [
            ("FPS",  f"{fps:.0f}" + ("  ■ PAUSED" if paused else "")),
            ("DRONES", f"{self.swarm.alive_count} / {config.NUM_DRONES}"),
            ("SPEED",  f"{config.MAX_SPEED:.1f}"),
            ("SEP",    f"{w['separation']:.1f}"),
            ("ALN",    f"{w['alignment']:.1f}"),
            ("COH",    f"{w['cohesion']:.1f}"),
            ("REL",    f"{w['relay']:.1f}"),
        ]

        lh, pad, pw = 18, 9, 168
        ph = pad * 2 + len(rows) * lh

        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill(C_HUD_BG)
        self._screen.blit(panel, (6, 6))
        pygame.draw.rect(self._screen, C_HUD_BORDER, (6, 6, pw, ph), 1)

        for i, (label, val) in enumerate(rows):
            y  = 6 + pad + i * lh
            ls = self._font.render(label, True, C_HUD_LABEL)
            vs = self._font.render(val,   True, C_HUD_VAL)
            self._screen.blit(ls, (14, y))
            self._screen.blit(vs, (82, y))

        # keybind reminder – bottom-left
        hints = "[SPC]Pause [R]Reset [E]Editor [L]Lines [K]Kill [↑↓]Drones [+−]Speed [1-4/S1-4]Weights"
        hs = pygame.font.SysFont("monospace", 11).render(hints, True, (60, 75, 105))
        self._screen.blit(hs, (6, config.WORLD_SIZE[1] - 16))


# ── Editor renderer ───────────────────────────────────────────────────────────
class EditorRenderer:
    GRID = 40

    def __init__(self, editor):
        self.editor    = editor
        self.show_grid = True
        self._font     = None
        self._font_sm  = None
        self._screen   = None

    def init(self, screen):
        self._screen  = screen
        self._font    = pygame.font.SysFont("monospace", 14)
        self._font_sm = pygame.font.SysFont("monospace", 12)

    def draw(self):
        s = self._screen
        s.fill(C_BG)

        if self.show_grid:
            self._draw_grid()

        env = self.editor.environment
        # world objects (behind sidebar visually but same coords)
        for z  in env.zones:     self._draw_zone(z)
        for b  in env.buildings: self._draw_building(b)
        for t  in env.trees:     self._draw_tree(t)
        for w  in env.walls:     self._draw_wall(w)
        for i, wp in enumerate(env.waypoints):
            self._draw_waypoint(wp, i + 1)

        self._draw_drag_preview()
        self._draw_sidebar()

    # ── world objects (same helpers as sim renderer) ──────────────────────────
    def _draw_grid(self):
        w, h = config.WORLD_SIZE
        for x in range(0, w, self.GRID):
            pygame.draw.line(self._screen, (22, 24, 38), (x, 0), (x, h))
        for y in range(0, h, self.GRID):
            pygame.draw.line(self._screen, (22, 24, 38), (0, y), (w, y))

    def _draw_zone(self, zone):
        x, y, w, h = (int(v) for v in zone.rect)
        col  = ZONE_COLORS.get(zone.zone_type, (128, 128, 128, 55))
        surf = pygame.Surface((max(w, 1), max(h, 1)), pygame.SRCALPHA)
        surf.fill(col)
        self._screen.blit(surf, (x, y))
        border = (min(col[0] + 60, 255), min(col[1] + 60, 255), min(col[2] + 60, 255))
        pygame.draw.rect(self._screen, border, (x, y, w, h), 1)
        lbl = self._font_sm.render(zone.zone_type, True, border)
        self._screen.blit(lbl, (x + 4, y + 4))

    def _draw_building(self, bld):
        x, y, w, h = (int(v) for v in bld.rect)
        pygame.draw.rect(self._screen, C_BLD_FILL,   (x, y, w, h))
        pygame.draw.rect(self._screen, C_BLD_BORDER, (x, y, w, h), 2)
        wsurf = pygame.Surface((max(w, 1), max(h, 1)), pygame.SRCALPHA)
        ww, wh, gx, gy = 6, 5, 9, 9
        for wx in range(gx, w - ww, ww + gx):
            for wy in range(gy, h - wh, wh + gy):
                pygame.draw.rect(wsurf, C_BLD_WIN, (wx, wy, ww, wh))
        self._screen.blit(wsurf, (x, y))
        lbl = self._font_sm.render("BLD", True, C_BLD_BORDER)
        self._screen.blit(lbl, (x + 4, y + 4))

    def _draw_tree(self, tree):
        pos = tree.position.astype(int)
        r   = int(tree.radius)
        pygame.draw.circle(self._screen, C_TREE_OUTER, pos, r)
        pygame.draw.circle(self._screen, C_TREE_MID,   pos, max(r - 4, 2))
        hi = (pos[0] - r // 3, pos[1] - r // 3)
        pygame.draw.circle(self._screen, C_TREE_HI, hi, max(r // 3, 2))
        pygame.draw.circle(self._screen, C_TREE_TRUNK, pos, 3)

    def _draw_wall(self, wall):
        pygame.draw.line(self._screen, C_WALL,
                         wall.start.astype(int), wall.end.astype(int), 3)
        # endpoint handles
        for ep in (wall.start, wall.end):
            pygame.draw.circle(self._screen, C_WALL, ep.astype(int), 5, 2)

    def _draw_waypoint(self, wp, idx):
        px, py = int(wp[0]), int(wp[1])
        pygame.draw.circle(self._screen, C_WP, (px, py), 8, 2)
        txt = self._font.render(str(idx), True, C_WP)
        self._screen.blit(txt, (px + 10, py - 10))

    def _draw_drag_preview(self):
        ed = self.editor
        if not ed.drag_start or ed.current_tool not in ("W", "B", "Z"):
            return
        mx, my = pygame.mouse.get_pos()
        sx, sy = ed.drag_start
        if ed.current_tool == "W":
            pygame.draw.line(self._screen, (255, 200, 80), (sx, sy), (mx, my), 2)
        elif ed.current_tool == "B":
            x, y = min(sx, mx), min(sy, my)
            w, h = abs(mx - sx), abs(my - sy)
            if w > 1 and h > 1:
                pygame.draw.rect(self._screen, C_BLD_FILL,   (x, y, w, h))
                pygame.draw.rect(self._screen, C_BLD_BORDER, (x, y, w, h), 2)
        elif ed.current_tool == "Z":
            x, y = min(sx, mx), min(sy, my)
            w, h = abs(mx - sx), abs(my - sy)
            if w > 1 and h > 1:
                col  = ZONE_COLORS.get(ed.zone_type, (128, 128, 128, 55))
                surf = pygame.Surface((w, h), pygame.SRCALPHA)
                surf.fill(col)
                self._screen.blit(surf, (x, y))

    # ── sidebar ───────────────────────────────────────────────────────────────
    def _draw_sidebar(self):
        h  = config.WORLD_SIZE[1]
        sw = SIDEBAR_W

        panel = pygame.Surface((sw, h), pygame.SRCALPHA)
        panel.fill(C_SIDEBAR_BG)
        self._screen.blit(panel, (0, 0))
        pygame.draw.line(self._screen, C_SIDEBAR_BDR, (sw, 0), (sw, h), 1)

        ed  = self.editor
        env = ed.environment
        y   = 10

        # title
        title = self._font.render("EDITOR", True, (140, 170, 220))
        self._screen.blit(title, (sw // 2 - title.get_width() // 2, y))
        y += 22
        pygame.draw.line(self._screen, C_SECTION, (6, y), (sw - 6, y)); y += 8

        # tool buttons
        self._section_label("TOOLS", y); y += 16
        for key, label in ed.TOOLS:
            active = ed.current_tool == key
            bg     = C_BTN_ACTIVE if active else C_BTN
            rect   = pygame.Rect(6, y, sw - 12, 26)
            pygame.draw.rect(self._screen, bg, rect, border_radius=4)
            if active:
                pygame.draw.rect(self._screen, C_BTN_KEY, rect, 1, border_radius=4)
            ks = self._font_sm.render(f"[{key}]", True, C_BTN_KEY)
            ls = self._font_sm.render(label,      True, C_BTN_TEXT)
            self._screen.blit(ks, (10,  y + 6))
            self._screen.blit(ls, (34,  y + 6))
            y += 30

        y += 4
        pygame.draw.line(self._screen, C_SECTION, (6, y), (sw - 6, y)); y += 8

        # zone type selector (only when Z active)
        if ed.current_tool == "Z":
            self._section_label("ZONE TYPE", y); y += 16
            from .environment import ZONE_TYPES
            for zt in ZONE_TYPES:
                active = ed.zone_type == zt
                bg     = C_BTN_ACTIVE if active else C_BTN
                col    = ZONE_COLORS.get(zt, (128, 128, 128, 55))
                rect   = pygame.Rect(6, y, sw - 12, 22)
                pygame.draw.rect(self._screen, bg,   rect, border_radius=3)
                dot_col = col[:3]
                pygame.draw.circle(self._screen, dot_col, (16, y + 11), 6)
                ztxt = self._font_sm.render(zt, True, C_BTN_TEXT)
                self._screen.blit(ztxt, (26, y + 5))
                y += 26
            y += 4
            pygame.draw.line(self._screen, C_SECTION, (6, y), (sw - 6, y)); y += 8

        # object counts
        self._section_label("OBJECTS", y); y += 16
        counts = [
            ("Walls",     len(env.walls)),
            ("Trees",     len(env.trees)),
            ("Buildings", len(env.buildings)),
            ("Zones",     len(env.zones)),
            ("Waypoints", len(env.waypoints)),
        ]
        for name, cnt in counts:
            ns = self._font_sm.render(name, True, C_COUNT_TEXT)
            cs = self._font_sm.render(str(cnt), True, C_HUD_VAL)
            self._screen.blit(ns, (10,  y))
            self._screen.blit(cs, (sw - 10 - cs.get_width(), y))
            y += 17

        y += 6
        pygame.draw.line(self._screen, C_SECTION, (6, y), (sw - 6, y)); y += 8

        # actions
        self._section_label("ACTIONS", y); y += 16
        actions = [("[G]", "Grid"), ("^S", "Save"), ("^O", "Load"), ("[E]", "Exit")]
        for key, lbl in actions:
            ks = self._font_sm.render(key, True, C_BTN_KEY)
            ls = self._font_sm.render(lbl, True, C_COUNT_TEXT)
            self._screen.blit(ks, (10,  y))
            self._screen.blit(ls, (42,  y))
            y += 16

    def _section_label(self, text, y):
        s = pygame.font.SysFont("monospace", 11).render(text, True, C_SECTION)
        self._screen.blit(s, (config.EDITOR_SIDEBAR_W // 2 - s.get_width() // 2, y))
