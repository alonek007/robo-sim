"""Step 3 output: Stage-1 trajectories plot + end-point error / heading-drift table (payload 0.4 kg)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
from stage1_analytic.simulate import run_one
from analysis.odometry import estimate
from analysis.metrics import score_run
from analysis.logio import relative_gt

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
PAYLOAD = 0.4


def table():
    rows = []
    for pose in P.POSES:
        for path in P.PATH_LEG_ANGLES_DEG:
            for rep in range(P.N_REPEATS):
                log = run_one(pose, PAYLOAD, path, False, seed=rep)
                std = score_run(log, estimate(log, PAYLOAD, "standard"), path)
                cor = score_run(log, estimate(log, PAYLOAD, "corrected"), path)
                logp = run_one(pose, PAYLOAD, path, True, seed=rep)
                rule = score_run(logp, estimate(logp, PAYLOAD, "corrected"), path)
                rows.append(dict(pose=pose, path=path, rep=rep,
                                 drift_deg=std["heading_drift_deg"],
                                 odom_err_std_mm=1e3 * std["odom_err_m"], odom_err_cor_mm=1e3 * cor["odom_err_m"],
                                 head_err_std_deg=std["odom_heading_err_deg"], head_err_cor_deg=cor["odom_heading_err_deg"],
                                 drift_rule_deg=rule["heading_drift_deg"]))
    df = pd.DataFrame(rows)
    g = df.groupby(["pose", "path"], sort=False)
    out = g.agg(drift_deg=("drift_deg", "mean"), drift_sd=("drift_deg", "std"),
                err_std_mm=("odom_err_std_mm", "mean"), err_cor_mm=("odom_err_cor_mm", "mean"),
                hd_err_std=("head_err_std_deg", "mean"), hd_err_cor=("head_err_cor_deg", "mean"),
                drift_rule_deg=("drift_rule_deg", "mean")).reset_index()
    return df, out


def plot(path):
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for j, p in enumerate(P.PATH_LEG_ANGLES_DEG):
        log = run_one("forward", PAYLOAD, p, False, seed=0)
        gt = relative_gt(log)
        a = ax[0, j]
        a.plot(gt[:, 0], gt[:, 1], "C3", lw=2, label="truth (plant)")
        for kind, st in (("standard", "C0--"), ("corrected", "C2:")):
            e = estimate(log, PAYLOAD, kind)
            a.plot(e[:, 0], e[:, 1], st, lw=2, label=f"{kind} odometry")
            ax[1, j].plot(log.t, np.degrees(e[:, 2]), st, lw=2, label=f"{kind} odometry")
        ax[1, j].plot(log.t, np.degrees(gt[:, 2]), "C3", lw=2, label="truth (plant)")
        if p == "forward":
            a.set_ylim(-0.04, 0.04); a.set_xlim(-0.05, 1.05); note = "\n(y axis stretched)"
        elif p == "strafe":
            a.set_xlim(-0.04, 0.04); a.set_ylim(-0.05, 1.05); note = "\n(x axis stretched)"
        else:
            a.set_aspect("equal"); note = ""
        a.grid(alpha=.3); a.set_title(f"{p}: arm forward, {PAYLOAD:g} kg{note}", fontsize=10)
        a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
        ax[1, j].grid(alpha=.3); ax[1, j].set_xlabel("t [s]"); ax[1, j].set_ylabel("heading [deg]")
        if j == 0: a.legend(fontsize=8); ax[1, j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=105); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True); (OUT / "plots").mkdir(exist_ok=True)
    raw, tab = table()
    raw.to_csv(OUT / "step3_stage1_runs.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"Stage 1, payload {PAYLOAD} kg, mean over {P.N_REPEATS} repeats (corrected = oracle parameters)\n")
    print(tab.round(3).to_string(index=False))
    plot(OUT / "plots" / "step3_stage1_trajectories.png")
