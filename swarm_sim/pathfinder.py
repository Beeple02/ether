"""
Grid-based A* pathfinder for the relay drone.

World is divided into CELL×CELL pixel tiles. Obstacles are rasterised onto
the grid with a clearance margin. find_path() returns a world-space list of
positions, string-pulled (visibility graph shortcut) to remove redundant
intermediate nodes.
"""

import numpy as np
import heapq
from . import config

CELL   = 16    # world pixels per grid tile
MARGIN = 2     # extra tile clearance around obstacles


class PathFinder:
    def __init__(self):
        self.cw      = config.WORLD_SIZE[0] // CELL
        self.ch      = config.WORLD_SIZE[1] // CELL
        self.blocked = np.zeros((self.ch, self.cw), dtype=bool)
        self.version = 0          # incremented each rebuild; renderer uses for cache
        self._surf   = None       # cached debug surface

    # ── rasterise environment ─────────────────────────────────────────────────
    def rebuild(self, environment):
        self.blocked[:] = False
        self._surf = None
        if not environment:
            self.version += 1
            return

        for bld in environment.buildings:
            x, y, w, h = bld.rect
            gx0 = max(0, int(x / CELL) - MARGIN)
            gy0 = max(0, int(y / CELL) - MARGIN)
            gx1 = min(self.cw, int((x + w) / CELL) + MARGIN + 1)
            gy1 = min(self.ch, int((y + h) / CELL) + MARGIN + 1)
            self.blocked[gy0:gy1, gx0:gx1] = True

        for tree in environment.trees:
            cx, cy = tree.position
            r   = tree.radius + MARGIN * CELL * 0.8
            gcx = int(cx / CELL)
            gcy = int(cy / CELL)
            gr  = int(r / CELL) + 2
            for dy in range(-gr, gr + 1):
                for dx in range(-gr, gr + 1):
                    gx, gy = gcx + dx, gcy + dy
                    if 0 <= gx < self.cw and 0 <= gy < self.ch:
                        ccx = (gx + 0.5) * CELL
                        ccy = (gy + 0.5) * CELL
                        if (ccx - cx) ** 2 + (ccy - cy) ** 2 < r ** 2:
                            self.blocked[gy, gx] = True

        for wall in environment.walls:
            s, e = wall.start, wall.end
            ln = np.linalg.norm(e - s)
            if ln < 1:
                continue
            steps = max(int(ln / (CELL * 0.35)), 1)
            for i in range(steps + 1):
                pt = s + (i / steps) * (e - s)
                gx = int(pt[0] / CELL)
                gy = int(pt[1] / CELL)
                for dy in range(-MARGIN, MARGIN + 1):
                    for dx in range(-MARGIN, MARGIN + 1):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.cw and 0 <= ny < self.ch:
                            self.blocked[ny, nx] = True

        for zone in environment.zones:
            if zone.zone_type == "NOFLYZONE":
                x, y, w, h = zone.rect
                gx0 = max(0, int(x / CELL) - MARGIN)
                gy0 = max(0, int(y / CELL) - MARGIN)
                gx1 = min(self.cw, int((x + w) / CELL) + MARGIN + 1)
                gy1 = min(self.ch, int((y + h) / CELL) + MARGIN + 1)
                self.blocked[gy0:gy1, gx0:gx1] = True

        self.version += 1

    # ── coordinate helpers ────────────────────────────────────────────────────
    def _to_cell(self, pos):
        return (int(np.clip(pos[0] / CELL, 0, self.cw - 1)),
                int(np.clip(pos[1] / CELL, 0, self.ch - 1)))

    def _to_world(self, cell):
        return np.array([(cell[0] + 0.5) * CELL,
                         (cell[1] + 0.5) * CELL], dtype=float)

    def _free(self, x, y):
        return 0 <= x < self.cw and 0 <= y < self.ch and not self.blocked[y, x]

    def _nearest_free(self, cell):
        if self._free(*cell):
            return cell
        for r in range(1, 25):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nc = (cell[0] + dx, cell[1] + dy)
                        if self._free(*nc):
                            return nc
        return cell

    # ── A* ────────────────────────────────────────────────────────────────────
    _DIRS = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
             (-1,-1,1.414),(1,-1,1.414),(-1,1,1.414),(1,1,1.414)]

    def find_path(self, start_pos, goal_pos):
        start_pos = np.array(start_pos, dtype=float)
        goal_pos  = np.array(goal_pos,  dtype=float)
        start = self._nearest_free(self._to_cell(start_pos))
        goal  = self._nearest_free(self._to_cell(goal_pos))

        if start == goal:
            return [goal_pos.copy()]

        gx_g, gy_g = goal
        def h(x, y):
            return ((x - gx_g) ** 2 + (y - gy_g) ** 2) ** 0.5

        open_set  = [(0.0, start)]
        came_from = {}
        g_score   = {start: 0.0}
        visited   = set()

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur in visited:
                continue
            visited.add(cur)

            if cur == goal:
                path = []
                while cur in came_from:
                    path.append(self._to_world(cur))
                    cur = came_from[cur]
                path.reverse()
                return self._smooth(path, start_pos, goal_pos)

            cx, cy = cur
            for dx, dy, cost in self._DIRS:
                nx, ny = cx + dx, cy + dy
                nb = (nx, ny)
                if not self._free(nx, ny) or nb in visited:
                    continue
                # prevent diagonal cuts through wall corners
                if dx != 0 and dy != 0:
                    if not self._free(cx + dx, cy) or not self._free(cx, cy + dy):
                        continue
                tg = g_score[cur] + cost
                if tg < g_score.get(nb, 1e18):
                    came_from[nb] = cur
                    g_score[nb]   = tg
                    heapq.heappush(open_set, (tg + h(nx, ny), nb))

        return [goal_pos.copy()]   # no path found — go direct

    # ── string-pull smoothing ─────────────────────────────────────────────────
    def _smooth(self, path, start_pos, goal_pos):
        full = [start_pos] + path + [goal_pos]
        if len(full) <= 2:
            return [goal_pos.copy()]
        out = [full[0]]
        i = 0
        while i < len(full) - 1:
            j = len(full) - 1
            while j > i + 1:
                if self._los(out[-1], full[j]):
                    break
                j -= 1
            out.append(full[j])
            i = j
        return [np.array(p, dtype=float) for p in out[1:]]

    def _los(self, a, b):
        a, b = np.array(a), np.array(b)
        diff = b - a
        ln   = np.linalg.norm(diff)
        if ln < 1:
            return True
        steps = max(int(ln / (CELL * 0.35)), 1)
        for i in range(steps + 1):
            pt = a + (i / steps) * diff
            gx = int(np.clip(pt[0] / CELL, 0, self.cw - 1))
            gy = int(np.clip(pt[1] / CELL, 0, self.ch - 1))
            if self.blocked[gy, gx]:
                return False
        return True

    # ── debug surface (cached) ────────────────────────────────────────────────
    def debug_surface(self):
        if self._surf is not None:
            return self._surf
        import pygame
        W, H = config.WORLD_SIZE
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        for gy in range(self.ch):
            for gx in range(self.cw):
                if self.blocked[gy, gx]:
                    pygame.draw.rect(surf, (200, 50, 50, 35),
                                     (gx * CELL + 1, gy * CELL + 1,
                                      CELL - 2, CELL - 2))
        self._surf = surf
        return self._surf
