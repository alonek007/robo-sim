"""Step 1 output: CoM / wheel-load table for every pose x payload, plus pose figure."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
from model.arm import arm_points
from model.com import static_state, joint_gravity_torques

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"


def table():
    rows = []
    for name, q in P.POSES.items():
        for pay in P.PAYLOADS:
            s = static_state(q, pay)
            t2, t3 = joint_gravity_torques(q, pay)
            rows.append(dict(pose=name, q1=q[0], q2=q[1], q3=q[2], payload=pay, M=s["M"],
                             tau_shoulder_Nm=t2, tau_elbow_Nm=t3, x_c_mm=1e3 * s["x_c"], y_c_mm=1e3 * s["y_c"], s=s["s"],
                             **{f"N_{w}": n for w, n in zip(P.WHEEL_NAMES, s["N"])}))
    return pd.DataFrame(rows)


def figure(path):
    fig, ax = plt.subplots(2, len(P.POSES), figsize=(4 * len(P.POSES), 7.5))
    for j, (name, q) in enumerate(P.POSES.items()):
        p = arm_points(q)
        chain = [p["mount"], p["shoulder"], p["elbow"], p["tip"]]
        for i, (u, v, xl, yl) in enumerate([(0, 2, "x [m]", "z [m]"), (0, 1, "x [m]", "y [m]")]):
            a = ax[i, j]
            a.plot([c[u] for c in chain], [c[v] for c in chain], "o-", color="C0", lw=3, label="arm")
            if i == 1:   # top view: wheel footprint and support polygon
                a.plot(*P.WHEEL_XY[[0, 1, 3, 2, 0]].T, "k--", lw=1)
                a.scatter(*P.WHEEL_XY.T, c="k", marker="s", s=40, label="wheels")
            for pay, col in zip(P.PAYLOADS, ["C2", "C1", "C3"]):
                s = static_state(q, pay)
                a.scatter(s["x_c"] if u == 0 else 0, s["z_c"] if v == 2 else s["y_c"],
                          c=col, marker="*", s=140, zorder=5, label=f"CoM, {pay:g} kg (s={s['s']:.2f})")
            a.set_aspect("equal"); a.grid(alpha=.3); a.set_xlabel(xl); a.set_ylabel(yl)
            a.set_title(f"{name} {tuple(q)}\n{'side view' if i == 0 else 'top view'}", fontsize=9)
            a.set_xlim(-0.2, 0.32)
            a.set_ylim((-0.02, 0.4) if i == 0 else (-0.2, 0.32))
            if j == 0: a.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True); (OUT / "plots").mkdir(exist_ok=True)
    df = table()
    df.to_csv(OUT / "static_loads.csv", index=False)
    figure(OUT / "plots" / "poses.png")
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print("\nmin stability margin:", round(df.s.min(), 3), "(posture-rule threshold", P.S_MIN, ")")
