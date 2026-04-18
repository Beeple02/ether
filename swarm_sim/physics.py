import numpy as np


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros(2)
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
