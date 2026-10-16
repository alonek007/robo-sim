"""Odometry estimators that read only the common log schema (works on real robot logs)."""
import numpy as np
import params as P
from model.com import static_state
from model.mecanum import J_PINV, integrate_pose
from model.load_aware import body_velocity_batch, TRUE

OMEGA = ["omega1", "omega2", "omega3", "omega4"]
Q = ["q1", "q2", "q3"]


def body_velocities(log, payload, kind, slip=TRUE):
    """Estimated body velocity per interval, (N+1,3); kind in {'standard','corrected'}."""
    om = log[OMEGA].values
    if kind == "standard":
        return P.WHEEL_R * om @ J_PINV.T
    if kind != "corrected":
        raise ValueError(kind)
    q = np.round(log[Q].values, 1)                       # arm encoder resolution 0.1 deg
    out = np.zeros((len(om), 3))
    for key in {tuple(r) for r in q}:                       # one batch solve per distinct arm configuration
        idx = np.all(q == key, axis=1)
        st = static_state(key, payload)
        out[idx] = body_velocity_batch(om[idx], st["N"], st["M"], slip)
    return out


def estimate(log, payload, kind, slip=TRUE):
    """Dead-reckoned pose (N+1,3) in the start frame."""
    v = body_velocities(log, payload, kind, slip)
    dt = np.diff(log.t.values)
    # integrate_pose assumes uniform dt; real logs are checked to be (nearly) uniform
    if not np.allclose(dt, dt[0], rtol=1e-3):
        raise ValueError("non-uniform sample times; resample the log first")
    return integrate_pose(v[:-1], dt[0])
