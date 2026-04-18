import pygame
import numpy as np
from . import config
from .environment import ZONE_COLORS


class Renderer:
    def __init__(self, swarm, editor=None):
        self.swarm = swarm
        self.editor = editor
        self.show_lines = False
        self._font = None
        self._screen = None

    def init(self, screen):
        self._screen = screen
        self._font = pygame.font.SysFont("monospace", 14)

    def draw(self, fps, paused):
        screen = self._screen
        screen.fill((15, 15, 25))

        env = self.swarm._environment
        if env:
            self._draw_environment(env)

        if self.show_lines:
            self._draw_perception_lines()

        self._draw_drones()
        self._draw_relay()
        self._draw_hud(fps, paused)

    def _draw_environment(self, env):
        screen = self._screen
        for zone in env.zones:
            x, y, w, h = zone.rect
            color = ZONE_COLORS.get(zone.zone_type, (128, 128, 128, 60))
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            surf.fill(color)
            screen.blit(surf, (x, y))

        for wall in env.walls:
            pygame.draw.line(
                screen, (180, 140, 80),
                wall.start.astype(int), wall.end.astype(int), 3
            )

        for i, wp in enumerate(env.waypoints):
            pygame.draw.circle(screen, (80, 200, 255), (int(wp[0]), int(wp[1])), 6, 2)
            txt = self._font.render(str(i + 1), True, (80, 200, 255))
            screen.blit(txt, (int(wp[0]) + 8, int(wp[1]) - 8))

    def _draw_perception_lines(self):
        relay_pos = self.swarm.relay.position
        for d in self.swarm.drones:
            if not d.alive:
                continue
            dist = np.linalg.norm(d.position - relay_pos)
            if dist < config.PERCEPTION_RADIUS:
                pygame.draw.line(
                    self._screen, (40, 40, 60),
                    d.position.astype(int), relay_pos.astype(int), 1
                )

    def _draw_drones(self):
        for d in self.swarm.drones:
            if d.alive:
                pygame.draw.circle(
                    self._screen, (220, 220, 220),
                    d.position.astype(int), 4
                )

    def _draw_relay(self):
        pygame.draw.circle(
            self._screen, (255, 230, 50),
            self.swarm.relay.position.astype(int), 10
        )

    def _draw_hud(self, fps, paused):
        lines = [
            f"FPS: {fps:.0f}{'  [PAUSED]' if paused else ''}",
            f"Drones: {self.swarm.alive_count}/{config.NUM_DRONES}",
            f"Speed: {config.MAX_SPEED:.1f}",
            f"Sep:{config.WEIGHTS['separation']:.1f} "
            f"Aln:{config.WEIGHTS['alignment']:.1f} "
            f"Coh:{config.WEIGHTS['cohesion']:.1f} "
            f"Rel:{config.WEIGHTS['relay']:.1f}",
        ]
        for i, line in enumerate(lines):
            surf = self._font.render(line, True, (180, 220, 180))
            self._screen.blit(surf, (8, 8 + i * 18))


class EditorRenderer:
    GRID_SIZE = 40

    def __init__(self, editor):
        self.editor = editor
        self._font = None
        self._screen = None
        self.show_grid = True

    def init(self, screen):
        self._screen = screen
        self._font = pygame.font.SysFont("monospace", 14)

    def draw(self):
        screen = self._screen
        screen.fill((10, 10, 18))

        if self.show_grid:
            self._draw_grid()

        env = self.editor.environment
        self._draw_zones(env)
        self._draw_walls(env)
        self._draw_waypoints(env)
        self._draw_drag_preview()
        self._draw_hud()
        self._draw_zone_menu()

    def _draw_grid(self):
        w, h = config.WORLD_SIZE
        for x in range(0, w, self.GRID_SIZE):
            pygame.draw.line(self._screen, (25, 25, 35), (x, 0), (x, h))
        for y in range(0, h, self.GRID_SIZE):
            pygame.draw.line(self._screen, (25, 25, 35), (0, y), (w, y))

    def _draw_zones(self, env):
        for zone in env.zones:
            x, y, w, h = zone.rect
            color = ZONE_COLORS.get(zone.zone_type, (128, 128, 128, 60))
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            surf.fill(color)
            self._screen.blit(surf, (x, y))
            label = self._font.render(zone.zone_type, True, color[:3])
            self._screen.blit(label, (x + 4, y + 4))

    def _draw_walls(self, env):
        for wall in env.walls:
            pygame.draw.line(
                self._screen, (200, 160, 80),
                wall.start.astype(int), wall.end.astype(int), 3
            )

    def _draw_waypoints(self, env):
        for i, wp in enumerate(env.waypoints):
            pygame.draw.circle(self._screen, (80, 200, 255), (int(wp[0]), int(wp[1])), 7, 2)
            txt = self._font.render(str(i + 1), True, (80, 200, 255))
            self._screen.blit(txt, (int(wp[0]) + 9, int(wp[1]) - 9))

    def _draw_drag_preview(self):
        ed = self.editor
        if ed.drag_start and ed.current_tool in ("W", "Z"):
            mx, my = pygame.mouse.get_pos()
            sx, sy = ed.drag_start
            if ed.current_tool == "W":
                pygame.draw.line(self._screen, (255, 200, 80), (sx, sy), (mx, my), 2)
            else:
                x, y = min(sx, mx), min(sy, my)
                w, h = abs(mx - sx), abs(my - sy)
                color = ZONE_COLORS.get(ed.zone_type, (128, 128, 128, 60))
                surf = pygame.Surface((max(w, 1), max(h, 1)), pygame.SRCALPHA)
                surf.fill(color)
                self._screen.blit(surf, (x, y))

    def _draw_hud(self):
        env = self.editor.environment
        lines = [
            f"[EDITOR]  Tool: {self.editor.current_tool}",
            f"Walls: {len(env.walls)}  Zones: {len(env.zones)}  WPs: {len(env.waypoints)}",
            "W=Wall Z=Zone P=Waypoint D=Drag DEL=Delete",
            "G=Grid  CTRL+S=Save  CTRL+O=Load  E=Exit",
        ]
        for i, line in enumerate(lines):
            surf = self._font.render(line, True, (160, 200, 160))
            self._screen.blit(surf, (8, 8 + i * 18))

        # Zone type shown top-right
        w = config.WORLD_SIZE[0]
        zt = self._font.render(f"ZoneType: {self.editor.zone_type}", True, (255, 200, 80))
        self._screen.blit(zt, (w - zt.get_width() - 8, 8))

    def _draw_zone_menu(self):
        if not self.editor.show_zone_menu:
            return
        x, y = self.editor.zone_menu_pos
        from .environment import ZONE_TYPES
        for i, zt in enumerate(ZONE_TYPES):
            color = (60, 60, 80) if zt != self.editor.zone_type else (100, 100, 140)
            rect = pygame.Rect(x, y + i * 28, 130, 26)
            pygame.draw.rect(self._screen, color, rect, border_radius=4)
            txt = self._font.render(zt, True, (220, 220, 220))
            self._screen.blit(txt, (x + 6, y + i * 28 + 6))
