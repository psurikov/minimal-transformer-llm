import math

def learning_rate_schedule(t: int, a_max: int, a_min: int, t_w: int, t_c: int):
    if t < t_w:
        a_t = (t / t_w) * a_max
    if t_w <= t <= t_c:
        a_t = a_min + 0.5 * (1 + math.cos((t - t_w)/(t_c - t_w) * math.pi)) * (a_max - a_min)
    if t > t_c:
        a_t = a_min
    return a_t