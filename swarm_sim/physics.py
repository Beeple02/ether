import numpy as np


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def limit(v, max_len):
    n = np.linalg.norm(v)
    if n > max_len:
        return v * (max_len / n)
    return v


def seek(position, target, velocity, max_speed, max_force):
    desired = normalize(target - position) * max_speed
    steer = desired - velocity
    return limit(steer, max_force)


def euler_integrate(position, velocity, dt=1.0):
    return position + velocity * dt


# ── geometry ─────────────────────────────────────────────────────────────────
def segments_intersect(a1, a2, b1, b2):
    """True if segment a1-a2 intersects segment b1-b2."""
    def ccw(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    d1 = ccw(b1, b2, a1)
    d2 = ccw(b1, b2, a2)
    d3 = ccw(a1, a2, b1)
    d4 = ccw(a1, a2, b2)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def segment_rect_intersect(a, b, rect):
    """True if segment a-b intersects rect (x, y, w, h)."""
    x, y, w, h = rect
    if x <= a[0] <= x + w and y <= a[1] <= y + h:
        return True
    if x <= b[0] <= x + w and y <= b[1] <= y + h:
        return True
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    for i in range(4):
        if segments_intersect(a, b, corners[i], corners[(i + 1) % 4]):
            return True
    return False


def rect_t_range(a2, b2, rect):
    """Parametric (t_enter, t_exit) where segment a2→b2 is inside rect.

    Uses Liang-Barsky clipping.  Returns (t_enter, t_exit) with
    0 ≤ t_enter < t_exit ≤ 1 when the segment crosses the rectangle's
    interior, or None when there is no crossing.

    Callers use this to find the z-height of a 3D segment at the point(s)
    where its XY projection crosses a building footprint.
    """
    x, y, w, h = rect
    dx = b2[0] - a2[0]
    dy = b2[1] - a2[1]
    t_min = 0.0
    t_max = 1.0

    for p, q in ((-dx, a2[0] - x), (dx, x + w - a2[0]),
                 (-dy, a2[1] - y), (dy, y + h - a2[1])):
        if p == 0.0:
            if q < 0.0:
                return None          # parallel and outside this edge
        else:
            t = q / p
            if p < 0.0:
                t_min = max(t_min, t)
            else:
                t_max = min(t_max, t)

    if t_min > t_max:
        return None
    return (t_min, t_max)
