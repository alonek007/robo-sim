"""Step 5 output: MuJoCo logs (common CSV schema) -> both odometry estimators -> table + plot (payload 0.4 kg)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
from stage2_mujoco.simulate import run_one
from analysis.logio import write_log, read_log, relative_gt
from analysis.odometry import estimate
from analysis.metrics import score_run

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "results" / "logs" / "stage2_step5"
PAYLOAD = 0.4


def main():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    rows, logs = [], {}
    for pose in P.POSES:
        for path in P.PATH_LEG_ANGLES_DEG:
            f = LOGDIR / f"{pose}_{PAYLOAD:g}kg_{path}.csv"
            write_log(run_one(pose, PAYLOAD, path, False, seed=0), f)
            log = read_log(f)                       # read back through the same reader a real-robot log would use
            logs[(pose, path)] = log
            std = score_run(log, estimate(log, PAYLOAD, "standard"), path)
            cor = score_run(log, estimate(log, PAYLOAD, "corrected"), path)   # stage-1 nominal params, NOT fitted yet
            rows.append(dict(pose=pose, path=path, drift_deg=std["heading_drift_deg"], path_dev_mm=1e3 * std["path_dev_m"],
                             odom_err_std_mm=1e3 * std["odom_err_m"], head_err_std_deg=std["odom_heading_err_deg"],
                             odom_err_cor_nominal_mm=1e3 * cor["odom_err_m"], head_err_cor_nominal_deg=cor["odom_heading_err_deg"]))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "step5_stage2_table.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"MuJoCo, payload {PAYLOAD} kg, one run per cell (seed 0). 'cor_nominal' = Stage-1 true parameters, not fitted.\n")
    print(df.round(2).to_string(index=False))

    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for j, path in enumerate(P.PATH_LEG_ANGLES_DEG):
        for pose, col in (("tucked", "C0"), ("forward", "C3"), ("side", "C2")):
            log = logs[(pose, path)]
            gt = relative_gt(log)
            ax[0, j].plot(gt[:, 0], gt[:, 1], col, label=f"truth, arm {pose}")
            ax[1, j].plot(log.t, np.degrees(gt[:, 2]), col, label=f"truth, arm {pose}")
            e = estimate(log, PAYLOAD, "standard")
            ax[1, j].plot(log.t, np.degrees(e[:, 2]), col, ls=":", lw=1)
        ax[0, j].set_title(f"{path}  (MuJoCo, {PAYLOAD:g} kg)"); ax[0, j].set_aspect("equal")
        ax[0, j].set_xlabel("x [m]"); ax[0, j].set_ylabel("y [m]")
        ax[1, j].set_xlabel("t [s]"); ax[1, j].set_ylabel("heading [deg] (dotted: standard odometry)")
        for a in ax[:, j]: a.grid(alpha=.3)
        if j == 0: ax[0, j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "results" / "plots" / "step5_stage2_trajectories.png", dpi=105)


if __name__ == "__main__":
    main()
