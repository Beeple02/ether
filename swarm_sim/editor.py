import pygame
import numpy as np
from .environment import Environment, Wall, Zone, ZONE_TYPES


class Editor:
    def __init__(self, environment=None):
        self.environment = environment or Environment()
        self.current_tool = "W"
        self.zone_type = "OBSTACLE"
        self.drag_start = None
        self.drag_object = None
        self.show_zone_menu = False
        self.zone_menu_pos = (0, 0)
        self._font = None
        self._text_input = None

    def init(self, font):
        self._font = font

    def handle_event(self, event, screen):
        if event.type == pygame.KEYDOWN:
            return self._handle_key(event, screen)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            return self._handle_mouse_down(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            return self._handle_mouse_up(event)
        return None

    def _handle_key(self, event, screen):
        ctrl = pygame.key.get_mods() & pygame.KMOD_CTRL

        if ctrl and event.key == pygame.K_s:
            self._save_prompt(screen)
            return None
        if ctrl and event.key == pygame.K_o:
            self._load_prompt(screen)
            return None

        key_map = {
            pygame.K_w: "W",
            pygame.K_z: "Z",
            pygame.K_p: "P",
            pygame.K_d: "D",
        }
        if event.key in key_map:
            self.current_tool = key_map[event.key]
            if self.current_tool == "Z":
                self.show_zone_menu = True
                mx, my = pygame.mouse.get_pos()
                self.zone_menu_pos = (mx, my)
            else:
                self.show_zone_menu = False

        if event.key == pygame.K_DELETE:
            self._delete_at(pygame.mouse.get_pos())

        if event.key == pygame.K_g:
            return "toggle_grid"

        if event.key == pygame.K_e:
            return "exit_editor"

        return None

    def _handle_mouse_down(self, event):
        pos = event.pos

        if self.show_zone_menu:
            if self._click_zone_menu(pos):
                self.show_zone_menu = False
                return None
            self.show_zone_menu = False

        if self.current_tool in ("W", "Z"):
            self.drag_start = pos
        elif self.current_tool == "P":
            self.environment.waypoints.append(pos)
        elif self.current_tool == "D":
            self.drag_object = self._find_object(pos)

        return None

    def _handle_mouse_up(self, event):
        pos = event.pos

        if self.current_tool == "W" and self.drag_start:
            sx, sy = self.drag_start
            if abs(pos[0] - sx) > 2 or abs(pos[1] - sy) > 2:
                self.environment.walls.append(Wall(self.drag_start, pos))
            self.drag_start = None

        elif self.current_tool == "Z" and self.drag_start:
            sx, sy = self.drag_start
            mx, my = pos
            x, y = min(sx, mx), min(sy, my)
            w, h = abs(mx - sx), abs(my - sy)
            if w > 4 and h > 4:
                self.environment.zones.append(Zone((x, y, w, h), self.zone_type))
            self.drag_start = None

        elif self.current_tool == "D" and self.drag_object:
            self._place_dragged(pos)
            self.drag_object = None

        return None

    def _click_zone_menu(self, pos):
        mx, my = self.zone_menu_pos
        for i, zt in enumerate(ZONE_TYPES):
            r = pygame.Rect(mx, my + i * 28, 130, 26)
            if r.collidepoint(pos):
                self.zone_type = zt
                return True
        return False

    def _find_object(self, pos):
        px, py = pos
        # Check waypoints
        for i, wp in enumerate(self.environment.waypoints):
            if abs(wp[0] - px) < 12 and abs(wp[1] - py) < 12:
                return ("waypoint", i)
        # Check walls (near endpoint)
        for i, wall in enumerate(self.environment.walls):
            for ep in (wall.start, wall.end):
                if np.linalg.norm(ep - np.array(pos)) < 12:
                    return ("wall_end", i, ep is wall.end)
        return None

    def _place_dragged(self, pos):
        if not self.drag_object:
            return
        kind = self.drag_object[0]
        if kind == "waypoint":
            idx = self.drag_object[1]
            self.environment.waypoints[idx] = pos
        elif kind == "wall_end":
            idx = self.drag_object[1]
            is_end = self.drag_object[2]
            wall = self.environment.walls[idx]
            if is_end:
                wall.end = np.array(pos, dtype=float)
            else:
                wall.start = np.array(pos, dtype=float)

    def _delete_at(self, pos):
        px, py = pos
        # Try waypoints
        for i, wp in enumerate(self.environment.waypoints):
            if abs(wp[0] - px) < 12 and abs(wp[1] - py) < 12:
                self.environment.waypoints.pop(i)
                return
        # Try walls
        for i, wall in enumerate(self.environment.walls):
            cp = wall.closest_point(np.array(pos, dtype=float))
            if np.linalg.norm(cp - np.array(pos)) < 10:
                self.environment.walls.pop(i)
                return
        # Try zones
        for i, zone in enumerate(self.environment.zones):
            if zone.contains(pos):
                self.environment.zones.pop(i)
                return

    def _save_prompt(self, screen):
        name = self._text_prompt(screen, "Save as (no extension): ")
        if name:
            path = f"environments/{name}.json"
            self.environment.save(path)

    def _load_prompt(self, screen):
        name = self._text_prompt(screen, "Load file (no extension): ")
        if name:
            path = f"environments/{name}.json"
            try:
                self.environment.load(path)
            except FileNotFoundError:
                pass

    def _text_prompt(self, screen, prompt):
        font = pygame.font.SysFont("monospace", 18)
        text = ""
        w, h = screen.get_size()
        box = pygame.Rect(w // 2 - 200, h // 2 - 20, 400, 40)
        active = True
        clock = pygame.time.Clock()
        while active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return text
                    elif event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        if event.unicode.isprintable():
                            text += event.unicode

            overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (0, 0))
            pygame.draw.rect(screen, (50, 50, 70), box, border_radius=4)
            pygame.draw.rect(screen, (100, 100, 150), box, 2, border_radius=4)
            plabel = font.render(prompt + text + "|", True, (220, 220, 220))
            screen.blit(plabel, (box.x + 6, box.y + 10))
            pygame.display.flip()
            clock.tick(30)
        return None
