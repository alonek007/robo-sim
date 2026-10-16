"""Result figures from a matrix CSV: python analysis/plots.py results/matrix_stage1.csv  (writes results/plots/*_<tag>.png)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P

POSE_COL = {"tucked": "#4c78a8", "forward": "#e45756", "side": "#54a24b", "raised": "#b279a2"}
PATHS = list(P.PATH_LEG_ANGLES_DEG)


def drift_bars(df, out, title):
    d = df[df["mode"] == "baseline"]
    fig, ax = plt.subplots(1, len(P.PAYLOADS), figsize=(5.2 * len(P.PAYLOADS), 4.2), sharey=True)
    for a, pay in zip(np.atleast_1d(ax), P.PAYLOADS):
        for i, pose in enumerate(P.POSES):
            g = d[(d.pose == pose) & (d.payload == pay)].groupby("path").heading_drift_deg.agg(["mean", "std"]).reindex(PATHS)
            a.bar(np.arange(len(PATHS)) + (i - 1.5) * 0.2, g["mean"], 0.2, yerr=g["std"], color=POSE_COL[pose], label=pose, capsize=2)
        a.axhline(0, color="k", lw=0.8); a.set_xticks(range(len(PATHS))); a.set_xticklabels(PATHS)
        a.set_title(f"payload {pay:g} kg"); a.grid(axis="y", alpha=.3)
    np.atleast_1d(ax)[0].set_ylabel("true heading drift at end [deg]"); np.atleast_1d(ax)[0].legend(title="arm pose", fontsize=8)
    fig.suptitle(f"{title}: heading drift without posture rule (mean +- sd over repeats)"); fig.tight_layout(); fig.savefig(out, dpi=105); plt.close(fig)


def h3_scatter(df, out, title):
    c = df.groupby(["pose", "payload", "path", "mode"]).odom_err_m.mean().unstack("mode") * 1e3
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for pose in P.POSES:
        x = c.xs(pose, level="pose")
        ax[0].scatter(x["baseline"], x["corrected"], s=24, color=POSE_COL[pose], label=pose,
                      marker="o" if pose in (P.FIT_TEST_POSES if df.stage.iloc[0] == 2 else P.POSES) else "x")
    m = c[["baseline", "corrected"]].values.max() * 1.05
    ax[0].plot([0, m], [0, m], "k--", lw=1); ax[0].set_xlabel("standard odometry end-point error [mm]")
    ax[0].set_ylabel("corrected odometry end-point error [mm]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("per condition (below dashed line = corrected better)" + ("\n(x = training poses)" if df.stage.iloc[0] == 2 else ""), fontsize=9)
    h = df.groupby(["pose", "payload", "path", "mode"]).odom_heading_err_deg.apply(lambda v: v.abs().mean()).unstack("mode")
    for pose in P.POSES:
        x = h.xs(pose, level="pose")
        ax[1].scatter(x["baseline"], x["corrected"], s=24, color=POSE_COL[pose], label=pose)
    m = h[["baseline", "corrected"]].values.max() * 1.05
    ax[1].plot([0, m], [0, m], "k--", lw=1); ax[1].set_xlabel("standard |heading error| [deg]"); ax[1].set_ylabel("corrected |heading error| [deg]")
    ax[1].grid(alpha=.3); ax[1].set_title("odometry heading error", fontsize=9)
    fig.suptitle(f"{title}: H3"); fig.tight_layout(); fig.savefig(out, dpi=105); plt.close(fig)


def h4_bars(df, out, title):
    x = df.assign(v=df.heading_drift_deg.abs()).groupby(["pose", "payload", "path", "mode"]).v.mean().unstack("mode")
    x = x.xs(0.4, level="payload")
    labels = [f"{p}\n{q}" for p, q in x.index]
    fig, ax = plt.subplots(figsize=(14, 4.2))
    i = np.arange(len(x))
    ax.bar(i - 0.2, x["corrected"], 0.4, color="#e45756", label="no posture rule")
    ax.bar(i + 0.2, x["corrected+posture"], 0.4, color="#4c78a8", label="posture rule")
    ax.set_xticks(i); ax.set_xticklabels(labels, fontsize=7); ax.set_ylabel("|true heading drift| [deg]"); ax.legend(); ax.grid(axis="y", alpha=.3)
    ax.set_title(f"{title}: H4, payload 0.4 kg (pose / path)"); fig.tight_layout(); fig.savefig(out, dpi=105); plt.close(fig)


if __name__ == "__main__":
    f = pathlib.Path(sys.argv[1]); tag = f.stem.replace("matrix_", "")
    df = pd.read_csv(f); out = f.parent / "plots"
    drift_bars(df, out / f"drift_{tag}.png", tag); h3_scatter(df, out / f"h3_{tag}.png", tag); h4_bars(df, out / f"h4_{tag}.png", tag)
