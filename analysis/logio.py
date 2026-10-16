"""Common log schema (simulation and real robot).

CSV columns: t, omega1..omega4 (FL,FR,RL,RR wheel speed [rad/s]), q1..q3 [deg], imu (gyro z [rad/s]),
gt_x, gt_y, gt_theta (ground truth pose [m, m, rad]).

Row k holds the wheel speeds that act during the interval [t_k, t_k+1] and the ground-truth pose AT t_k.
The last row is a rest row (omega = 0); its omega is never integrated. So N+1 rows -> N intervals.
"""
import numpy as np, pandas as pd

COLUMNS = ["t", "omega1", "omega2", "omega3", "omega4", "q1", "q2", "q3", "imu", "gt_x", "gt_y", "gt_theta"]


def write_log(df, path):
    df.to_csv(path, index=False)


def read_log(path):
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"log {path} is missing columns {missing}")
    return df[COLUMNS]


def relative_gt(df):
    """Ground-truth pose (N+1,3) expressed in the frame of the first sample."""
    x, y, th = df.gt_x.values, df.gt_y.values, df.gt_theta.values
    c, s = np.cos(th[0]), np.sin(th[0])
    dx, dy = x - x[0], y - y[0]
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy, th - th[0]])
