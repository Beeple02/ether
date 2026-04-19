import pygame
import numpy as np
from . import config
from .environment import (Environment, Wall, Tree, Building, Zone,
                          Base, Turret, ZONE_TYPES)

SIDEBAR_W = config.EDITOR_SIDEBAR_W

TOOLS = [
    ("W", "Wall"),
    ("T", "Tree"),
    ("H", "Building"),
    ("B", "Base"),
    ("U", "Turret"),
    ("Z", "Zone"),
    ("P", "Waypoint"),
    ("D", "Drag"),
    ("X", "Delete"),
]


class Editor:
    TOOLS = TOOLS

    def __init__(self, environment=None):
        self.environment  = environment or Environment()
        self.current_tool = "W"
        self.zone_type    = "TARGET"
        self.drag_start   = None
        self.drag_object  = None

    # ── event router ─────────────────────────────────────────────────────────
    def handle_event(self, event, screen):
        if event.type == pygame.KEYDOWN:
            return self._handle_key(event, screen)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self._handle_down(event.pos)
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            return self._handle_up(event.pos)
        return None

    # ── keyboard ─────────────────────────────────────────────────────────────
    def _handle_key(self, event, screen):
        ctrl = pygame.key.get_mods() & pygame.KMOD_CTRL
        if ctrl and event.key == pygame.K_s:
            self._save_prompt(screen); return None
        if ctrl and event.key == pygame.K_o:
            self._load_prompt(screen); return None

        key_map = {
            pygame.K_w: "W", pygame.K_t: "T", pygame.K_h: "H",
            pygame.K_b: "B", pygame.K_u: "U", pygame.K_z: "Z",
            pygame.K_p: "P", pygame.K_d: "D",
            pygame.K_x: "X", pygame.K_DELETE: "X",
        }
        if event.key in key_map:
            self.current_tool = key_map[event.key]

        if event.key == pygame.K_g:
            return "toggle_grid"
        if event.key == pygame.K_e:
            return "exit_editor"
        return None

    # ── mouse down ───────────────────────────────────────────────────────────
    def _handle_down(self, pos):
        if pos[0] < SIDEBAR_W:
            self._sidebar_click(pos)
            return None

        tool = self.current_tool

        if tool in ("W", "H", "Z"):
            self.drag_start = pos

        elif tool == "T":
            self.environment.trees.append(Tree(pos, radius=18))

        elif tool == "B":
            # Replace any existing base (only one allowed)
            self.environment.base = Base(pos)

        elif tool == "U":
            self.environment.turrets.append(Turret(pos))

        elif tool == "P":
            self.environment.waypoints.append(tuple(pos))

        elif tool == "D":
            self.drag_object = self._find_object(pos)

        elif tool == "X":
            self._delete_at(pos)

        return None

    # ── mouse up ─────────────────────────────────────────────────────────────
    def _handle_up(self, pos):
        if not self.drag_start and not self.drag_object:
            return None

        tool = self.current_tool

        if tool == "W" and self.drag_start:
            sx, sy = self.drag_start
            if abs(pos[0] - sx) > 3 or abs(pos[1] - sy) > 3:
                self.environment.walls.append(Wall(self.drag_start, pos))
            self.drag_start = None

        elif tool == "H" and self.drag_start:
            sx, sy = self.drag_start
            x, y = min(sx, pos[0]), min(sy, pos[1])
            w, h = abs(pos[0] - sx), abs(pos[1] - sy)
            if w > 8 and h > 8:
                self.environment.buildings.append(Building((x, y, w, h)))
            self.drag_start = None

        elif tool == "Z" and self.drag_start:
            sx, sy = self.drag_start
            x, y = min(sx, pos[0]), min(sy, pos[1])
            w, h = abs(pos[0] - sx), abs(pos[1] - sy)
            if w > 8 and h > 8:
                self.environment.zones.append(Zone((x, y, w, h), self.zone_type))
            self.drag_start = None

        elif tool == "D" and self.drag_object:
            self._apply_drag(pos)
            self.drag_object = None

        return None

    # ── sidebar click ─────────────────────────────────────────────────────────
    def _sidebar_click(self, pos):
        y = 10 + 22 + 8 + 16
        for key, label in TOOLS:
            rect = pygame.Rect(6, y, SIDEBAR_W - 12, 26)
            if rect.collidepoint(pos):
                self.current_tool = key
                return
            y += 30

        if self.current_tool == "Z":
            y += 4 + 8 + 16
            for zt in ZONE_TYPES:
                rect = pygame.Rect(6, y, SIDEBAR_W - 12, 22)
                if rect.collidepoint(pos):
                    self.zone_type = zt
                    return
                y += 26

    # ── drag helpers ──────────────────────────────────────────────────────────
    def _find_object(self, pos):
        pv = np.array(pos, dtype=float)

        if self.environment.base is not None:
            if np.linalg.norm(pv - self.environment.base.position) < 16:
                return ("base",)

        for i, t in enumerate(self.environment.turrets):
            if np.linalg.norm(pv - t.position) < 14:
                return ("turret", i)

        for i, wp in enumerate(self.environment.waypoints):
            if np.linalg.norm(pv - np.array(wp)) < 14:
                return ("wp", i)

        for i, t in enumerate(self.environment.trees):
            if np.linalg.norm(pv - t.position) < t.radius + 6:
                return ("tree", i)

        for i, w in enumerate(self.environment.walls):
            for j, ep in enumerate([w.start, w.end]):
                if np.linalg.norm(pv - ep) < 12:
                    return ("wall_ep", i, j)

        for i, b in enumerate(self.environment.buildings):
            if b.contains(pos):
                return ("building", i, pos)

        return None

    def _apply_drag(self, pos):
        obj = self.drag_object
        if obj is None:
            return
        kind = obj[0]
        if kind == "wp":
            self.environment.waypoints[obj[1]] = tuple(pos)
        elif kind == "tree":
            self.environment.trees[obj[1]].position = np.array(pos, dtype=float)
        elif kind == "wall_ep":
            wall = self.environment.walls[obj[1]]
            if obj[2] == 0:
                wall.start = np.array(pos, dtype=float)
            else:
                wall.end = np.array(pos, dtype=float)
        elif kind == "building":
            bld    = self.environment.buildings[obj[1]]
            ox, oy = obj[2]
            x, y, w, h = bld.rect
            dx, dy = pos[0] - ox, pos[1] - oy
            bld.rect = (x + dx, y + dy, w, h)
        elif kind == "base":
            self.environment.base.position = np.array(pos, dtype=float)
        elif kind == "turret":
            self.environment.turrets[obj[1]].position = np.array(pos, dtype=float)

    # ── delete ───────────────────────────────────────────────────────────────
    def _delete_at(self, pos):
        pv = np.array(pos, dtype=float)

        if self.environment.base is not None:
            if np.linalg.norm(pv - self.environment.base.position) < 16:
                self.environment.base = None
                return

        for i, t in enumerate(self.environment.turrets):
            if np.linalg.norm(pv - t.position) < 14:
                self.environment.turrets.pop(i); return

        for i, wp in enumerate(self.environment.waypoints):
            if np.linalg.norm(pv - np.array(wp)) < 14:
                self.environment.waypoints.pop(i); return

        for i, t in enumerate(self.environment.trees):
            if np.linalg.norm(pv - t.position) < t.radius + 6:
                self.environment.trees.pop(i); return

        for i, w in enumerate(self.environment.walls):
            cp = w.closest_point(pv)
            if np.linalg.norm(pv - cp) < 10:
                self.environment.walls.pop(i); return

        for i, b in enumerate(self.environment.buildings):
            if b.contains(pos):
                self.environment.buildings.pop(i); return

        for i, z in enumerate(self.environment.zones):
            if z.contains(pos):
                self.environment.zones.pop(i); return

    # ── file prompts ──────────────────────────────────────────────────────────
    def _save_prompt(self, screen):
        name = text_prompt(screen, "Save as (no extension): ")
        if name:
            self.environment.save(f"environments/{name}.json")

    def _load_prompt(self, screen):
        name = text_prompt(screen, "Load file (no extension): ")
        if name:
            try:
                self.environment.load(f"environments/{name}.json")
            except FileNotFoundError:
                pass


def text_prompt(screen, prompt):
    font   = pygame.font.SysFont("monospace", 18)
    text   = ""
    w, h   = screen.get_size()
    box    = pygame.Rect(w // 2 - 220, h // 2 - 22, 440, 44)
    clock  = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return text
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                elif event.unicode.isprintable():
                    text += event.unicode

        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))
        pygame.draw.rect(screen, (40, 44, 64), box, border_radius=5)
        pygame.draw.rect(screen, (80, 96, 150), box, 2, border_radius=5)
        label = font.render(prompt + text + "▌", True, (210, 220, 230))
        screen.blit(label, (box.x + 8, box.y + 10))
        pygame.display.flip()
        clock.tick(30)


# backwards-compat alias
_text_prompt = text_prompt
