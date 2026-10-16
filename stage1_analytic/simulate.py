"""Stage 1: pure-python plant (eq.5 with TRUE kappa, c, f_r) commanded by standard inverse kinematics."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import params as P
import paths
from model.com import static_state
from model.mecanum import wheel_speeds, integrate_pose
from model.load_aware import body_velocity, TRUE
from model.posture import apply_rule
from analysis.logio import COLUMNS


def run_one(pose, payload, path, posture_rule, seed):
    """Simulate one run, return a log DataFrame in the common schema (see analysis/logio.py)."""
    prof = paths.build(path)
    rng = np.random.default_rng(seed)
    v_cmd, n = prof["v_cmd"], len(prof["t"]) - 1
    q_req = P.POSES[pose]

    q_leg, state_leg = {}, {}
    for i, a in zip(prof["leg"], prof["angle_deg"]):
        if i not in q_leg:
            q_leg[i] = apply_rule(q_req, payload, a) if posture_rule else q_req
            state_leg[i] = static_state(q_leg[i], payload)

    q = np.array([q_leg[i] for i in prof["leg"]])
    omega_cmd = np.array([wheel_speeds(v) for v in v_cmd])
    v_true = np.zeros((n, 3))
    sig = np.array([P.VEL_NOISE, P.VEL_NOISE, P.VEL_NOISE / P.L])   # yaw noise: same rim-speed noise at radius L
    for k in range(n):
        st = state_leg[prof["leg"][k]]
        v_true[k] = body_velocity(omega_cmd[k], st["N"], st["M"], TRUE) + rng.normal(0, sig)
    gt = integrate_pose(v_true, P.DT)

    omega_meas = omega_cmd + rng.normal(0, P.OMEGA_NOISE, omega_cmd.shape)
    omega_meas[-1] = 0.0
    imu = np.append(v_true[:, 2], 0.0) + rng.normal(0, P.GYRO_NOISE, n + 1)
    df = pd.DataFrame(np.column_stack([prof["t"], omega_meas, q, imu, gt]), columns=COLUMNS)
    return df
