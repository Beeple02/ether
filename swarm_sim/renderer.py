import pygame
import numpy as np
from . import config
from .environment import ZONE_COLORS
from .effects import EMPBlast, SmokeCloud, NetDeploy, Explosion

# ── palette ───────────────────────────────────────────────────────────────────
C_BG          = (14,  15,  24)
C_DRONE       = (205, 215, 230)
C_RELAY_RING  = (160, 120,  10)
C_RELAY       = (255, 220,  40)
C_LINE        = ( 38,  48,  72)
C_WALL        = (190, 150,  65)
C_WP          = ( 75, 185, 255)
C_HUD_BG      = ( 18,  20,  34, 210)
C_HUD_BDR     = ( 50,  62,  98)
C_HUD_LABEL   = ( 95, 125, 165)
C_HUD_VAL     = (155, 215, 175)
C_TREE_OUT    = ( 28,  85,  38)
C_TREE_MID    = ( 42, 125,  52)
C_TREE_HI     = ( 68, 155,  70)
C_TREE_TRUNK  = ( 75,  48,  18)
C_BLD_FILL    = ( 48,  50,  64)
C_BLD_BDR     = ( 78,  82, 106)
C_BLD_WIN     = ( 88, 118, 162, 160)
C_SB_BG       = ( 16,  18,  28, 230)
C_SB_BDR      = ( 44,  54,  88)
C_BTN         = ( 30,  34,  52)
C_BTN_ACT     = ( 55,  80, 140)
C_BTN_TEXT    = (185, 200, 220)
C_BTN_KEY     = (110, 160, 220)
C_SECTION     = ( 70,  82, 120)
C_COUNT       = (120, 145, 175)

# debug colors
DC_NEIGHBOR   = ( 55,  75, 160,  70)
DC_PERC       = (255, 255, 255,  20)
DC_SEP_RING   = (255, 220,  40,  35)
DC_VEL        = ( 80, 230,  80)
DC_FSEP       = (255,  55,  55)
DC_FALN       = ( 55, 100, 255)
DC_FCOH       = ( 55, 215, 175)
DC_FREL       = (255, 200,  40)
DC_FENV       = (255, 130,  40)
DC_WALL_HIT   = (255, 165,  50)
DC_TREE_HIT   = ( 50, 255, 100)
DC_BLD_HIT    = (255,  75,  50)
DC_ZONE_T     = ( 50, 255, 120)
DC_ZONE_N     = (255,  50,  50)
DC_RECON_SCAN = ( 75, 195,  75)   # RECON scan radius ring
DC_THREAT_T   = (255,  80,  80)   # TURRET_THREAT label
DC_THREAT_P   = (255, 215,  55)   # PROJECTILE_THREAT label
DC_THREAT_D   = (220,  80, 220)   # DRONE_THREAT label

FORCE_SCALE   = 55   # pixels per unit of force
SIDEBAR_W     = config.EDITOR_SIDEBAR_W


# ── shape helpers ─────────────────────────────────────────────────────────────
def _draw_diamond(surface, color, center, size):
    x, y = int(center[0]), int(center[1])
    pts = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
    pygame.draw.polygon(surface, color, pts)


def _draw_triangle(surface, color, center, size, velocity=None):
    x, y = int(center[0]), int(center[1])
    if velocity is not None and np.linalg.norm(velocity) > 0.1:
        angle = np.arctan2(velocity[1], velocity[0])
    else:
        angle = -np.pi / 2
    tip = (x + np.cos(angle) * size,              y + np.sin(angle) * size)
    lft = (x + np.cos(angle + 2.3) * size * 0.75, y + np.sin(angle + 2.3) * size * 0.75)
    rgt = (x + np.cos(angle - 2.3) * size * 0.75, y + np.sin(angle - 2.3) * size * 0.75)
    pygame.draw.polygon(surface, color,
                        [(int(tip[0]), int(tip[1])),
                         (int(lft[0]), int(lft[1])),
                         (int(rgt[0]), int(rgt[1]))])


# ── pentagon helper ───────────────────────────────────────────────────────────
def _draw_pentagon(surface, color, center, size, width=0):
    x, y = int(center[0]), int(center[1])
    pts = [
        (int(x + np.cos(-np.pi / 2 + 2 * np.pi * i / 5) * size),
         int(y + np.sin(-np.pi / 2 + 2 * np.pi * i / 5) * size))
        for i in range(5)
    ]
    pygame.draw.polygon(surface, color, pts, width)


# ── arrow helper ──────────────────────────────────────────────────────────────
def _arrow(surface, color, start, end, width=1):
    sx, sy = int(start[0]), int(start[1])
    ex, ey = int(end[0]), int(end[1])
    dx, dy = ex - sx, ey - sy
    length = (dx * dx + dy * dy) ** 0.5
    if length < 2:
        return
    pygame.draw.line(surface, color, (sx, sy), (ex, ey), width)
    if length > 6:
        ux, uy = dx / length, dy / length
        sz = min(7, length * 0.35)
        l1 = (ex - ux * sz - uy * sz * 0.5, ey - uy * sz + ux * sz * 0.5)
        l2 = (ex - ux * sz + uy * sz * 0.5, ey - uy * sz - ux * sz * 0.5)
        pygame.draw.polygon(surface, color,
                            [(ex, ey), (int(l1[0]), int(l1[1])), (int(l2[0]), int(l2[1]))])


# ── Sim renderer ──────────────────────────────────────────────────────────────
class Renderer:
    def __init__(self, swarm):
        self.swarm              = swarm
        self.show_lines         = False
        self.debug_mode         = False
        self._font              = None
        self._font_sm           = None
        self._screen            = None
        self._grid_surf         = None   # cached A* blocked-cell surface
        self._grid_surf_ver     = -1     # pathfinder version it was built from
        self._end_overlay_timer = 0.0
        self._end_overlay_stats = None

    def init(self, screen):
        self._screen  = screen
        self._font    = pygame.font.SysFont("monospace", 14)
        self._font_sm = pygame.font.SysFont("monospace", 12)

    def draw(self, fps, paused):
        s = self._screen
        s.fill(C_BG)
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

    # ── world objects ─────────────────────────────────────────────────────────
    def _draw_env(self, env):
        for z in env.zones:     self._zone(z)
        for b in env.buildings: self._building(b)
        for t in env.trees:     self._tree(t)
        for w in env.walls:     self._wall(w)
        for i, wp in enumerate(env.waypoints):
            self._waypoint(wp, i + 1)
        for t in env.turrets:   self._turret(t)
        if env.enemy_base:      self._enemy_base(env.enemy_base)

    def _zone(self, zone):
        x, y, w, h = (int(v) for v in zone.rect)
        col  = ZONE_COLORS.get(zone.zone_type, (128, 128, 128, 55))
        surf = pygame.Surface((max(w, 1), max(h, 1)), pygame.SRCALPHA)
        surf.fill(col)
        self._screen.blit(surf, (x, y))
        border = (min(col[0]+50,255), min(col[1]+50,255), min(col[2]+50,255))
        pygame.draw.rect(self._screen, border, (x, y, w, h), 1)

    def _building(self, bld):
        x, y, w, h = (int(v) for v in bld.rect)
        pygame.draw.rect(self._screen, C_BLD_FILL, (x, y, w, h))
        pygame.draw.rect(self._screen, C_BLD_BDR,  (x, y, w, h), 2)
        if w > 28 and h > 28:
            cols = min(8, max(1, w // 28))
            rows = min(6, max(1, h // 28))
            gx   = max(5, (w - cols * 8) // (cols + 1))
            gy   = max(5, (h - rows * 7) // (rows + 1))
            ww   = max(6, (w - (cols + 1) * gx) // cols)
            wh   = max(5, (h - (rows + 1) * gy) // rows)
            ws   = pygame.Surface((max(w,1), max(h,1)), pygame.SRCALPHA)
            for c in range(cols):
                for r in range(rows):
                    wx, wy = gx + c * (ww + gx), gy + r * (wh + gy)
                    if wx + ww < w and wy + wh < h:
                        pygame.draw.rect(ws, C_BLD_WIN, (wx, wy, ww, wh))
            self._screen.blit(ws, (x, y))

    def _tree(self, tree):
        pos = tree.position.astype(int)
        r   = int(tree.radius)
        pygame.draw.circle(self._screen, C_TREE_OUT,   pos, r)
        pygame.draw.circle(self._screen, C_TREE_MID,   pos, max(r - 4, 2))
        hi = (pos[0] - r // 3, pos[1] - r // 3)
        pygame.draw.circle(self._screen, C_TREE_HI,    hi,  max(r // 3, 2))
        pygame.draw.circle(self._screen, C_TREE_TRUNK, pos, 3)

    def _wall(self, wall):
        pygame.draw.line(self._screen, C_WALL,
                         wall.start.astype(int), wall.end.astype(int), 3)

    def _waypoint(self, wp, idx):
        px, py = int(wp[0]), int(wp[1])
        pygame.draw.circle(self._screen, C_WP, (px, py), 7, 2)
        txt = self._font.render(str(idx), True, C_WP)
        self._screen.blit(txt, (px + 9, py - 9))

    def _turret(self, turret):
        pos = turret.position.astype(int)
        disabled = turret._disabled_timer > 0
        body_col = (70, 72, 88) if disabled else (195, 55, 40)
        ring_col = (55, 58, 75) if disabled else (230, 90, 65)
        # faint range ring
        r = int(turret.range)
        alpha_surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(alpha_surf, (*ring_col, 18), (r + 1, r + 1), r, 1)
        self._screen.blit(alpha_surf, (pos[0] - r - 1, pos[1] - r - 1))
        # body
        pygame.draw.circle(self._screen, body_col, pos, 7)
        pygame.draw.circle(self._screen, ring_col, pos, 7, 2)
        if disabled:
            # small EMP-disabled indicator cross
            pygame.draw.line(self._screen, (120, 60, 220), (pos[0]-4, pos[1]-4), (pos[0]+4, pos[1]+4), 1)
            pygame.draw.line(self._screen, (120, 60, 220), (pos[0]+4, pos[1]-4), (pos[0]-4, pos[1]+4), 1)

    def _enemy_base(self, eb):
        pos = eb.position
        ipos = (int(pos[0]), int(pos[1]))
        _draw_pentagon(self._screen, (150, 25, 25), pos, 13)
        _draw_pentagon(self._screen, (255, 60, 60), pos, 13, width=2)
        txt = self._font_sm.render("ENEMY BASE", True, (255, 80, 80))
        self._screen.blit(txt, (ipos[0] - txt.get_width() // 2, ipos[1] + 17))

    # ── agents ────────────────────────────────────────────────────────────────
    def _draw_relay_lines(self):
        rp = self.swarm.relay.position
        for d in self.swarm.drones:
            if d.alive and np.linalg.norm(d.position - rp) < config.PERCEPTION_RADIUS:
                pygame.draw.line(self._screen, C_LINE,
                                 d.position.astype(int), rp.astype(int), 1)

    def _draw_drones(self):
        all_drones = list(self.swarm.drones) + list(self.swarm.enemy_drones)
        for d in all_drones:
            if not d.alive:
                continue
            if d._is_relay:
                continue   # promoted relay is drawn by _draw_relay()

            color = d.type_color
            size  = d.type_size
            pos   = d.position
            af    = d.airframe
            ipos  = (int(pos[0]), int(pos[1]))

            # Mimicry window: draw relay-style ring around the drone
            if d._mimicry_active:
                pygame.draw.circle(self._screen, C_RELAY_RING, ipos, size + 6, 1)

            if af == "medium":
                _draw_diamond(self._screen, color, pos, size)
            elif af == "large":
                _draw_triangle(self._screen, color, pos, size, d.velocity)
            else:
                pygame.draw.circle(self._screen, color, ipos, size)

            # EMP arming pulse — expanding purple ring grows over 3 s charge window
            if d.drone_type == "emp" and getattr(d, "_emp_stage", None) == "arming":
                t_frac  = min(1.0, d._emp_arming_t / config.EMP_ARMING_DELAY)
                ring_r  = int(8 + t_frac * 14)
                ring_a  = int(80 + t_frac * 160)
                rsurf   = pygame.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(rsurf, (175, 70, 220, ring_a),
                                   (ring_r + 2, ring_r + 2), ring_r, 2)
                self._screen.blit(rsurf, (ipos[0] - ring_r - 2, ipos[1] - ring_r - 2))

    def _draw_relay(self):
        rp = self.swarm.relay.position.astype(int)
        pygame.draw.circle(self._screen, C_RELAY_RING, rp, 12)
        pygame.draw.circle(self._screen, C_RELAY,      rp, 10)

    def _draw_projectiles(self):
        for p in self.swarm.projectiles:
            if p.alive:
                pygame.draw.circle(self._screen, (255, 215, 55),
                                   p.position.astype(int), config.PROJECTILE_RADIUS)

    def _draw_effects(self):
        for fx in self.swarm.effects:
            if not fx.alive:
                continue
            r = max(1, int(getattr(fx, 'radius', 0)))
            a = int(getattr(fx, 'alpha', 0) * 255)
            if r < 1 or a < 1:
                continue
            pos = fx.position.astype(int)
            sz  = r * 2 + 4

            if isinstance(fx, SmokeCloud):
                surf = pygame.Surface((sz, sz), pygame.SRCALPHA)
                pygame.draw.circle(surf, (155, 160, 165, a), (r + 2, r + 2), r)
                self._screen.blit(surf, (pos[0] - r - 2, pos[1] - r - 2))

            elif isinstance(fx, EMPBlast):
                surf = pygame.Surface((sz, sz), pygame.SRCALPHA)
                pygame.draw.circle(surf, (120, 60, 220, a),     (r + 2, r + 2), r, 3)
                pygame.draw.circle(surf, (200, 150, 255, a // 2),
                                   (r + 2, r + 2), max(1, r // 3))
                self._screen.blit(surf, (pos[0] - r - 2, pos[1] - r - 2))

            elif isinstance(fx, NetDeploy):
                surf = pygame.Surface((sz, sz), pygame.SRCALPHA)
                pygame.draw.circle(surf, (70, 215, 215, a), (r + 2, r + 2), r, 2)
                self._screen.blit(surf, (pos[0] - r - 2, pos[1] - r - 2))

            elif isinstance(fx, Explosion):
                col  = (*fx.color[:3], a)
                surf = pygame.Surface((sz, sz), pygame.SRCALPHA)
                pygame.draw.circle(surf, col, (r + 2, r + 2), r)
                self._screen.blit(surf, (pos[0] - r - 2, pos[1] - r - 2))

    # ── debug overlay ─────────────────────────────────────────────────────────
    def _draw_debug(self, env):
        alive = [d for d in self.swarm.drones if d.alive]
        W, H  = config.WORLD_SIZE

        # ── A* grid (blocked cells) — cached surface ──
        pf = self.swarm._pf
        if self._grid_surf_ver != pf.version:
            self._grid_surf     = pf.debug_surface()
            self._grid_surf_ver = pf.version
        if self._grid_surf:
            self._screen.blit(self._grid_surf, (0, 0))

        # ── single alpha surface for translucent rings & neighbor lines ──
        alpha = pygame.Surface((W, H), pygame.SRCALPHA)
        for drone in alive:
            pos = drone.position.astype(int)
            for nbp in drone.last_neighbor_positions:
                pygame.draw.line(alpha, DC_NEIGHBOR, pos, (int(nbp[0]), int(nbp[1])), 1)
            pygame.draw.circle(alpha, DC_PERC,     pos, config.PERCEPTION_RADIUS, 1)
            pygame.draw.circle(alpha, DC_SEP_RING, pos, config.SEPARATION_RADIUS, 1)
        self._screen.blit(alpha, (0, 0))

        # ── relay planned path ──
        path = self.swarm._relay_path
        pidx = self.swarm._path_idx
        if len(path) > 1:
            pts = [p.astype(int) for p in path]
            pygame.draw.lines(self._screen, (70, 180, 70), False, pts, 2)
            for i, p in enumerate(pts):
                col = (255, 255, 60) if i == pidx else (80, 200, 80)
                pygame.draw.circle(self._screen, col, p, 4 if i == pidx else 3)
        if self.swarm._path_target is not None:
            pt = self.swarm._path_target.astype(int)
            pygame.draw.line(self._screen, (220, 220, 50), (pt[0]-9,pt[1]-9),(pt[0]+9,pt[1]+9), 2)
            pygame.draw.line(self._screen, (220, 220, 50), (pt[0]+9,pt[1]-9),(pt[0]-9,pt[1]+9), 2)

        # ── drone follow target ──
        ft = self.swarm._drone_follow_target()
        pygame.draw.circle(self._screen, (150, 255, 150), ft.astype(int), 6, 2)

        # ── solid environment hitboxes ──
        if env:
            for wall in env.walls:
                pygame.draw.line(self._screen, DC_WALL_HIT,
                                 wall.start.astype(int), wall.end.astype(int), 3)
            for tree in env.trees:
                p = tree.position.astype(int)
                pygame.draw.circle(self._screen, DC_TREE_HIT, p, int(tree.radius), 2)
                eff = int(tree.radius + config.PERCEPTION_RADIUS * 0.3)
                pygame.draw.circle(self._screen, (50, 180, 80), p, eff, 1)
            for bld in env.buildings:
                x, y, w, h = (int(v) for v in bld.rect)
                pygame.draw.rect(self._screen, DC_BLD_HIT, (x, y, w, h), 2)
            for zone in env.zones:
                x, y, w, h = (int(v) for v in zone.rect)
                col = DC_ZONE_T if zone.zone_type == "TARGET" else DC_ZONE_N
                pygame.draw.rect(self._screen, col, (x, y, w, h), 2)

        # ── per-drone arrows ──
        for drone in alive:
            pos = drone.position.astype(int)
            # velocity
            ve = (pos[0] + int(drone.velocity[0] * 9),
                  pos[1] + int(drone.velocity[1] * 9))
            _arrow(self._screen, DC_VEL, pos, ve, 2)
            # force components
            for key, col in (("sep", DC_FSEP), ("aln", DC_FALN),
                             ("coh", DC_FCOH), ("rel", DC_FREL), ("env", DC_FENV)):
                f = drone.last_forces.get(key, np.zeros(2))
                if np.linalg.norm(f) > 0.004:
                    fe = (pos[0] + int(f[0] * FORCE_SCALE),
                          pos[1] + int(f[1] * FORCE_SCALE))
                    _arrow(self._screen, col, pos, fe, 1)

        # ── RECON scan radii ──
        recon_r = config.RECON_RANGE
        for rd in alive:
            if rd.drone_type != "recon":
                continue
            rpos = rd.position.astype(int)
            scan_sz = recon_r * 2 + 2
            scan_surf = pygame.Surface((scan_sz, scan_sz), pygame.SRCALPHA)
            pygame.draw.circle(scan_surf, (*DC_RECON_SCAN, 18),
                               (recon_r + 1, recon_r + 1), recon_r)
            pygame.draw.circle(scan_surf, (*DC_RECON_SCAN, 70),
                               (recon_r + 1, recon_r + 1), recon_r, 1)
            self._screen.blit(scan_surf, (rpos[0] - recon_r - 1, rpos[1] - recon_r - 1))

        # ── threat classification labels ──
        _KIND_COL = {
            "TURRET_THREAT":      DC_THREAT_T,
            "PROJECTILE_THREAT":  DC_THREAT_P,
            "DRONE_THREAT":       DC_THREAT_D,
        }
        _KIND_LABEL = {
            "TURRET_THREAT":      "TURRET",
            "PROJECTILE_THREAT":  "PROJ",
            "DRONE_THREAT":       "DRONE",
        }
        seen_threat_ids = set()
        for t in getattr(self.swarm, "_last_threats", []):
            oid = id(t["obj"])
            if oid in seen_threat_ids:
                continue
            seen_threat_ids.add(oid)
            col   = _KIND_COL.get(t["kind"],   (200, 200, 200))
            label = _KIND_LABEL.get(t["kind"], t["kind"])
            ipos  = (int(t["pos"][0]), int(t["pos"][1]))
            pygame.draw.circle(self._screen, col, ipos, 5, 1)
            txt = self._font_sm.render(label, True, col)
            self._screen.blit(txt, (ipos[0] - txt.get_width() // 2, ipos[1] - 22))

        # ── reflex indicator dots (per-drone, per-active rule) ──
        _REFLEX_DOT = {
            "COLLISION_AVOID": ((255, 120,  30), (-5, -8)),
            "GEOFENCE":        ((255, 220,  40), ( 5, -8)),
            "IFF_SAFE":        (( 40, 210, 210), (-5,  8)),
            "RELAY_SAFE":      (( 80, 220,  80), ( 5,  8)),
        }
        for drone in alive:
            if not drone.active_reflexes:
                continue
            pos = drone.position.astype(int)
            for rule, (col, (ox, oy)) in _REFLEX_DOT.items():
                if rule in drone.active_reflexes:
                    pygame.draw.circle(self._screen, col, (pos[0] + ox, pos[1] + oy), 3)

        # relay velocity
        rp  = self.swarm.relay.position.astype(int)
        rv  = self.swarm.relay.velocity
        rve = (rp[0] + int(rv[0] * 12), rp[1] + int(rv[1] * 12))
        _arrow(self._screen, (255, 255, 80), rp, rve, 2)

    def _draw_debug_legend(self):
        entries = [
            (DC_VEL,        "Velocity"),
            (DC_FSEP,       "Separation force"),
            (DC_FALN,       "Alignment force"),
            (DC_FCOH,       "Cohesion force"),
            (DC_FREL,       "Relay force"),
            (DC_FENV,       "Env repulsion"),
            ((200,200,200), "Perception radius"),
            ((255,220, 40), "Separation radius"),
            ((55,  75,160), "Neighbor link"),
            ((70, 180, 70), "Relay A* path"),
            ((255,255, 60), "Path next node"),
            ((150,255,150), "Drone follow target"),
            ((200, 50, 50), "Blocked grid cell"),
            (DC_RECON_SCAN,    "RECON scan radius"),
            (DC_THREAT_T,      "TURRET_THREAT"),
            (DC_THREAT_P,      "PROJECTILE_THREAT"),
            (DC_THREAT_D,      "DRONE_THREAT"),
            ((255, 120,  30),  "Reflex: COLLISION_AVOID"),
            ((255, 220,  40),  "Reflex: GEOFENCE"),
            (( 40, 210, 210),  "Reflex: IFF_SAFE"),
            (( 80, 220,  80),  "Reflex: RELAY_SAFE"),
        ]
        x0 = config.WORLD_SIZE[0] - 200
        y0 = 8
        pw, ph = 192, len(entries) * 17 + 22
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill(C_HUD_BG)
        self._screen.blit(panel, (x0 - 6, y0))
        pygame.draw.rect(self._screen, C_HUD_BDR, (x0 - 6, y0, pw, ph), 1)
        hdr = self._font_sm.render("[TAB] Debug ON", True, (130, 165, 215))
        self._screen.blit(hdr, (x0, y0 + 3))
        y = y0 + 20
        for col, label in entries:
            pygame.draw.line(self._screen, col[:3], (x0, y+6), (x0+18, y+6), 2)
            self._screen.blit(self._font_sm.render(label, True, C_HUD_LABEL), (x0+22, y))
            y += 17

    # ── type legend ───────────────────────────────────────────────────────────
    def _draw_type_legend(self):
        alive_counts = {}
        for d in self.swarm.drones:
            if d.alive and d.drone_type:
                alive_counts[d.drone_type] = alive_counts.get(d.drone_type, 0) + 1

        W, H  = config.WORLD_SIZE
        lh    = 16
        pad   = 6
        pw    = 172
        ph    = pad * 2 + (len(config.DRONE_TYPES) + 1) * lh
        x0    = W - pw - 6
        y0    = H - ph - 18

        pnl = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pnl.fill(C_HUD_BG)
        self._screen.blit(pnl, (x0, y0))
        pygame.draw.rect(self._screen, C_HUD_BDR, (x0, y0, pw, ph), 1)

        hdr = self._font_sm.render("TYPE          CNT", True, C_HUD_BDR)
        self._screen.blit(hdr, (x0 + 22, y0 + pad))

        y = y0 + pad + lh
        for dt, info in config.DRONE_TYPES.items():
            color = info["color"]
            af    = info["airframe"]
            sz    = min(info["size"], 5)
            count = alive_counts.get(dt, 0)
            cx, cy = x0 + 11, y + lh // 2

            if af == "medium":
                _draw_diamond(self._screen, color, (cx, cy), sz)
            elif af == "large":
                _draw_triangle(self._screen, color, (cx, cy), sz)
            else:
                pygame.draw.circle(self._screen, color, (cx, cy), sz)

            label_col = color if count else C_HUD_BDR
            row = f"{dt:<14} {count:3d}"
            self._screen.blit(self._font_sm.render(row, True, label_col), (x0 + 20, y + 2))
            y += lh

    # ── HUD ──────────────────────────────────────────────────────────────────
    def _draw_hud(self, fps, paused):
        w       = config.WEIGHTS
        mission = self.swarm._mission
        t0 = self.swarm._t_zero
        te = mission.total_elapsed
        conv_str = ("OFF" if not config.CONVERGENCE_ENABLED
                    else f"T-{max(0.0, t0 - te):.1f}s" if t0 is not None
                    else "ON")
        rows = [
            ("FPS",    f"{fps:.0f}" + ("  ■ PAUSED" if paused else "")),
            ("PHASE",  mission.phase),
            ("FORM",   self.swarm._formation.mode),
            ("CONV",   conv_str),
            ("DRONES", f"{self.swarm.alive_count} / {config.NUM_DRONES}"),
            ("SPEED",  f"{config.MAX_SPEED:.1f}"),
            ("SEP",    f"{w['separation']:.1f}"),
            ("ALN",    f"{w['alignment']:.1f}"),
            ("COH",    f"{w['cohesion']:.1f}"),
            ("REL",    f"{w['relay']:.1f}"),
        ]
        lh, pad, pw = 18, 9, 168
        ph  = pad * 2 + len(rows) * lh
        pnl = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pnl.fill(C_HUD_BG)
        self._screen.blit(pnl, (6, 6))
        pygame.draw.rect(self._screen, C_HUD_BDR, (6, 6, pw, ph), 1)
        for i, (label, val) in enumerate(rows):
            y = 6 + pad + i * lh
            self._screen.blit(self._font.render(label, True, C_HUD_LABEL), (14, y))
            self._screen.blit(self._font.render(val,   True, C_HUD_VAL),   (82, y))
        hints = ("[SPC]Pause [R]Reset [M]Phase [V]Conv [E]Editor [L]Lines [K]Kill "
                 "[↑↓]Drones [+−]Speed [1-4/S+1-4]Weights [TAB]Debug [F11]Fullscreen")
        hs = pygame.font.SysFont("monospace", 11).render(hints, True, (52, 65, 95))
        self._screen.blit(hs, (6, config.WORLD_SIZE[1] - 15))

    # ── post-run end overlay ──────────────────────────────────────────────────
    def _tick_end_overlay(self, dt):
        stats = self.swarm.stats
        if stats.ended:
            if self._end_overlay_stats is not stats:
                self._end_overlay_stats = stats
                self._end_overlay_timer = config.END_OVERLAY_SECONDS
            if self._end_overlay_timer > 0:
                self._end_overlay_timer -= dt
                self._draw_end_overlay(stats)

    def _draw_end_overlay(self, stats):
        W, H = config.WORLD_SIZE
        rows = [
            ("RESULT",              stats.end_reason.upper().replace("_", " ")),
            ("Time",                f"{stats.time_elapsed:.1f} s"),
            ("Drones survived",     f"{stats.drones_survived} / {stats.drones_total}"),
            ("Relay survived",      "YES" if stats.relay_survived else "NO"),
            ("Target reached",      "YES" if stats.target_reached else "NO"),
            ("Signal coverage pk",  f"{stats.peak_signal_coverage:.1%}"),
            ("Phase reached",       stats.phase_reached),
            ("Relay successions",   str(stats.relay_successions)),
            ("Intercepted projs",   str(stats.intercepted_projectiles)),
            ("Intercepted enemies", str(stats.intercepted_enemy_drones)),
            ("EMP turrets hit",     str(stats.emp_turrets_disabled)),
            ("Smoke deployments",   str(stats.smokescreen_deployments)),
            ("Convergence used",    "YES" if stats.convergence_used else "NO"),
            ("Enemy swarm elim.",   "YES" if stats.enemy_swarm_eliminated else "NO"),
        ]
        lh, pad = 18, 12
        pw = 310
        ph = pad * 2 + len(rows) * lh + 4
        x0 = (W - pw) // 2
        y0 = (H - ph) // 2

        pnl = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pnl.fill((10, 12, 24, 230))
        self._screen.blit(pnl, (x0, y0))
        pygame.draw.rect(self._screen, (80, 100, 160), (x0, y0, pw, ph), 2)

        result_col = (80, 240, 120) if stats.target_reached else (240, 90, 80)
        hdr = self._font.render(rows[0][1], True, result_col)
        self._screen.blit(hdr, (x0 + (pw - hdr.get_width()) // 2, y0 + pad))
        for i, (label, val) in enumerate(rows[1:], start=1):
            y = y0 + pad + i * lh + 2
            self._screen.blit(self._font_sm.render(label, True, C_HUD_LABEL), (x0 + 10, y))
            vtxt = self._font_sm.render(val, True, C_HUD_VAL)
            self._screen.blit(vtxt, (x0 + pw - vtxt.get_width() - 10, y))


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
        self._screen.fill(C_BG)
        if self.show_grid:
            self._draw_grid()
        env = self.editor.environment
        for z in env.zones:     self._zone(z)
        for b in env.buildings: self._building(b)
        for t in env.trees:     self._tree(t)
        for w in env.walls:     self._wall(w)
        for i, wp in enumerate(env.waypoints):
            self._waypoint(wp, i + 1)
        if env.enemy_base:      self._enemy_base(env.enemy_base)
        self._drag_preview()
        self._sidebar()

    def _enemy_base(self, eb):
        pos  = eb.position
        ipos = (int(pos[0]), int(pos[1]))
        _draw_pentagon(self._screen, (150, 25, 25), pos, 13)
        _draw_pentagon(self._screen, (255, 60, 60), pos, 13, width=2)
        txt = self._font_sm.render("ENEMY BASE", True, (255, 80, 80))
        self._screen.blit(txt, (ipos[0] - txt.get_width() // 2, ipos[1] + 17))

    def _draw_grid(self):
        w, h = config.WORLD_SIZE
        for x in range(0, w, self.GRID):
            pygame.draw.line(self._screen, (20, 22, 36), (x, 0), (x, h))
        for y in range(0, h, self.GRID):
            pygame.draw.line(self._screen, (20, 22, 36), (0, y), (w, y))

    def _zone(self, zone):
        x, y, w, h = (int(v) for v in zone.rect)
        col  = ZONE_COLORS.get(zone.zone_type, (128,128,128,55))
        surf = pygame.Surface((max(w,1), max(h,1)), pygame.SRCALPHA)
        surf.fill(col)
        self._screen.blit(surf, (x, y))
        border = (min(col[0]+60,255), min(col[1]+60,255), min(col[2]+60,255))
        pygame.draw.rect(self._screen, border, (x, y, w, h), 1)
        self._screen.blit(self._font_sm.render(zone.zone_type, True, border), (x+4, y+4))

    def _building(self, bld):
        x, y, w, h = (int(v) for v in bld.rect)
        pygame.draw.rect(self._screen, C_BLD_FILL, (x, y, w, h))
        pygame.draw.rect(self._screen, C_BLD_BDR,  (x, y, w, h), 2)
        if w > 28 and h > 28:
            cols = min(8, max(1, w // 28))
            rows = min(6, max(1, h // 28))
            gx   = max(5, (w - cols * 8) // (cols + 1))
            gy   = max(5, (h - rows * 7) // (rows + 1))
            ww   = max(6, (w - (cols + 1) * gx) // cols)
            wh   = max(5, (h - (rows + 1) * gy) // rows)
            ws   = pygame.Surface((max(w,1), max(h,1)), pygame.SRCALPHA)
            for c in range(cols):
                for r in range(rows):
                    wx, wy = gx + c*(ww+gx), gy + r*(wh+gy)
                    if wx+ww < w and wy+wh < h:
                        pygame.draw.rect(ws, C_BLD_WIN, (wx, wy, ww, wh))
            self._screen.blit(ws, (x, y))
        self._screen.blit(self._font_sm.render("BLD", True, C_BLD_BDR), (x+4, y+4))

    def _tree(self, tree):
        pos = tree.position.astype(int)
        r   = int(tree.radius)
        pygame.draw.circle(self._screen, C_TREE_OUT,   pos, r)
        pygame.draw.circle(self._screen, C_TREE_MID,   pos, max(r-4, 2))
        hi = (pos[0]-r//3, pos[1]-r//3)
        pygame.draw.circle(self._screen, C_TREE_HI,    hi,  max(r//3, 2))
        pygame.draw.circle(self._screen, C_TREE_TRUNK, pos, 3)

    def _wall(self, wall):
        pygame.draw.line(self._screen, C_WALL,
                         wall.start.astype(int), wall.end.astype(int), 3)
        for ep in (wall.start, wall.end):
            pygame.draw.circle(self._screen, C_WALL, ep.astype(int), 5, 2)

    def _waypoint(self, wp, idx):
        px, py = int(wp[0]), int(wp[1])
        pygame.draw.circle(self._screen, C_WP, (px, py), 8, 2)
        self._screen.blit(self._font.render(str(idx), True, C_WP), (px+10, py-10))

    def _drag_preview(self):
        ed = self.editor
        if not ed.drag_start or ed.current_tool not in ("W", "B", "Z"):
            return
        mx, my = pygame.mouse.get_pos()
        sx, sy = ed.drag_start
        if ed.current_tool == "W":
            pygame.draw.line(self._screen, (255, 200, 80), (sx, sy), (mx, my), 2)
        elif ed.current_tool == "B":
            x, y = min(sx,mx), min(sy,my)
            w, h = abs(mx-sx), abs(my-sy)
            if w > 1 and h > 1:
                pygame.draw.rect(self._screen, C_BLD_FILL, (x,y,w,h))
                pygame.draw.rect(self._screen, C_BLD_BDR,  (x,y,w,h), 2)
        elif ed.current_tool == "Z":
            x, y = min(sx,mx), min(sy,my)
            w, h = abs(mx-sx), abs(my-sy)
            if w > 1 and h > 1:
                col  = ZONE_COLORS.get(ed.zone_type, (128,128,128,55))
                surf = pygame.Surface((w,h), pygame.SRCALPHA)
                surf.fill(col)
                self._screen.blit(surf, (x,y))

    def _sidebar(self):
        sw, H = SIDEBAR_W, config.WORLD_SIZE[1]
        pnl   = pygame.Surface((sw, H), pygame.SRCALPHA)
        pnl.fill(C_SB_BG)
        self._screen.blit(pnl, (0, 0))
        pygame.draw.line(self._screen, C_SB_BDR, (sw, 0), (sw, H), 1)

        ed  = self.editor
        env = ed.environment
        y   = 10

        title = self._font.render("EDITOR", True, (140, 170, 220))
        self._screen.blit(title, (sw//2 - title.get_width()//2, y)); y += 22
        pygame.draw.line(self._screen, C_SECTION, (6, y), (sw-6, y)); y += 8

        self._slabel("TOOLS", y); y += 16
        for key, label in ed.TOOLS:
            active = ed.current_tool == key
            rect   = pygame.Rect(6, y, sw-12, 26)
            pygame.draw.rect(self._screen, C_BTN_ACT if active else C_BTN, rect, border_radius=4)
            if active:
                pygame.draw.rect(self._screen, C_BTN_KEY, rect, 1, border_radius=4)
            self._screen.blit(self._font_sm.render(f"[{key}]", True, C_BTN_KEY),  (10, y+6))
            self._screen.blit(self._font_sm.render(label,      True, C_BTN_TEXT), (36, y+6))
            y += 30

        y += 4
        pygame.draw.line(self._screen, C_SECTION, (6, y), (sw-6, y)); y += 8

        if ed.current_tool == "Z":
            from .environment import ZONE_TYPES
            self._slabel("ZONE TYPE", y); y += 16
            for zt in ZONE_TYPES:
                rect = pygame.Rect(6, y, sw-12, 22)
                pygame.draw.rect(self._screen,
                                 C_BTN_ACT if ed.zone_type == zt else C_BTN,
                                 rect, border_radius=3)
                col = ZONE_COLORS.get(zt, (128,128,128,55))
                pygame.draw.circle(self._screen, col[:3], (16, y+11), 6)
                self._screen.blit(self._font_sm.render(zt, True, C_BTN_TEXT), (26, y+5))
                y += 26
            y += 4
            pygame.draw.line(self._screen, C_SECTION, (6, y), (sw-6, y)); y += 8

        self._slabel("OBJECTS", y); y += 16
        for name, cnt in [("Walls", len(env.walls)), ("Trees", len(env.trees)),
                          ("Buildings", len(env.buildings)), ("Zones", len(env.zones)),
                          ("Waypoints", len(env.waypoints))]:
            ns = self._font_sm.render(name,    True, C_COUNT)
            cs = self._font_sm.render(str(cnt), True, C_HUD_VAL)
            self._screen.blit(ns, (10, y))
            self._screen.blit(cs, (sw - 10 - cs.get_width(), y))
            y += 17

        y += 6
        pygame.draw.line(self._screen, C_SECTION, (6, y), (sw-6, y)); y += 8
        self._slabel("ACTIONS", y); y += 16
        for key, lbl in [("[G]","Grid"), ("^S","Save"), ("^O","Load"), ("[E]","Exit")]:
            self._screen.blit(self._font_sm.render(key, True, C_BTN_KEY),  (10, y))
            self._screen.blit(self._font_sm.render(lbl, True, C_COUNT),    (42, y))
            y += 16

    def _slabel(self, text, y):
        s = pygame.font.SysFont("monospace", 11).render(text, True, C_SECTION)
        self._screen.blit(s, (SIDEBAR_W//2 - s.get_width()//2, y))
