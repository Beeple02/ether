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
