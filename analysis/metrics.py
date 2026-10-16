"""Per-run metrics, all in the start frame of the run."""
import numpy as np
import paths
from analysis.logio import relative_gt


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def score_run(log, est_pose, path):
    gt = relative_gt(log)
    ideal = paths.ideal_end(path)
    return {
        "odom_err_m": float(np.hypot(*(est_pose[-1, :2] - gt[-1, :2]))),          # estimator vs truth
        "odom_heading_err_deg": float(np.degrees(wrap(est_pose[-1, 2] - gt[-1, 2]))),
        "heading_drift_deg": float(np.degrees(wrap(gt[-1, 2]))),                   # truth vs commanded (0)
        "path_dev_m": float(np.hypot(*(gt[-1, :2] - ideal))),                      # truth vs ideal end-point
    }
