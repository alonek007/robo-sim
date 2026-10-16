"""Path generators shared by both stages: trapezoidal speed profile along each leg."""
import numpy as np
import params as P


def leg_angles(name):
    return P.PATH_LEG_ANGLES_DEG[name]


def ideal_end(name):
    """Ideal commanded end-point (x, y) in the start frame."""
    a = np.radians(leg_angles(name))
    return P.LEG * np.array([np.cos(a).sum(), np.sin(a).sum()])


def _leg_speed(dt):
    """Speed samples (mid-point) for one leg: ramp up, cruise, ramp down; integrates to P.LEG."""
    t_cruise = (P.LEG - P.SPEED * P.RAMP_T) / P.SPEED
    T = t_cruise + 2 * P.RAMP_T
    tau = (np.arange(int(round(T / dt))) + 0.5) * dt
    return np.minimum.reduce([P.SPEED * tau / P.RAMP_T, np.full_like(tau, P.SPEED),
                              P.SPEED * (T - tau) / P.RAMP_T])


def build(name, dt=P.DT):
    """Return dict(t, v_cmd (N+1,3) [vx,vy,wz], leg (N+1,) leg index, angle_deg (N+1,) leg direction).

    The last row is a rest row (v_cmd = 0) so that the log has N intervals and N+1 poses.
    """
    sp = _leg_speed(dt)
    pause = np.zeros(int(round(P.CORNER_PAUSE / dt)))
    v, leg, ang = [], [], []
    for i, a_deg in enumerate(leg_angles(name)):
        a = np.radians(a_deg)
        seg = np.stack([sp * np.cos(a), sp * np.sin(a), np.zeros_like(sp)], axis=1)
        v.append(seg); leg += [i] * len(sp); ang += [a_deg] * len(sp)
        if i < len(leg_angles(name)) - 1:
            v.append(np.zeros((len(pause), 3))); leg += [i] * len(pause); ang += [a_deg] * len(pause)
    v = np.vstack(v + [np.zeros((1, 3))])
    leg.append(leg[-1]); ang.append(ang[-1])
    return {"t": np.arange(len(v)) * dt, "v_cmd": v, "leg": np.array(leg), "angle_deg": np.array(ang)}
