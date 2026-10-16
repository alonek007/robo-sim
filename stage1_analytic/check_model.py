"""Step 2 output: noise-free load-aware plant vs standard odometry for constant-velocity drives."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
from model.com import static_state
from model.mecanum import wheel_speeds, standard_odometry, integrate_pose
from model.load_aware import body_velocity, TRUE

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
DRIVES = {"forward": (P.SPEED, 0, 0), "strafe": (0, P.SPEED, 0), "diagonal": (P.SPEED / np.sqrt(2), P.SPEED / np.sqrt(2), 0)}
T = P.LEG / P.SPEED


def table(payload=0.4):
    rows = []
    for pose, q in P.POSES.items():
        s = static_state(q, payload)
        for d, v in DRIVES.items():
            vp = body_velocity(wheel_speeds(v), s["N"], s["M"], TRUE)
            rows.append(dict(pose=pose, drive=d, vx=vp[0], vy=vp[1], wz_mrad_s=1e3 * vp[2],
                             heading_after_5s_deg=np.degrees(vp[2] * T),
                             speed_loss_pct=100 * (1 - np.hypot(*vp[:2]) / P.SPEED)))
    return pd.DataFrame(rows)


def plot(path, payload=0.4):
    s = static_state(P.POSES["forward"], payload)
    v_cmd = np.array(DRIVES["strafe"])
    om = wheel_speeds(v_cmd)
    n = int(round(T / P.DT))
    plant = body_velocity(om, s["N"], s["M"], TRUE)
    std = standard_odometry(om)
    gt = integrate_pose(np.tile(plant, (n, 1)), P.DT)
    od = integrate_pose(np.tile(std, (n, 1)), P.DT)
    t = np.arange(n + 1) * P.DT
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(gt[:, 0], gt[:, 1], "C3", label="plant (true)")
    ax[0].plot(od[:, 0], od[:, 1], "C0--", label="standard odometry")
    ax[0].set_xlim(-0.03, 0.03); ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[0].set_title(f"strafe 1 m, arm forward, {payload:g} kg payload\n(x axis stretched ~30x)", fontsize=10)
    ax[1].plot(t, np.degrees(gt[:, 2]), "C3", label="plant heading")
    ax[1].plot(t, np.degrees(od[:, 2]), "C0--", label="odometry heading")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("heading [deg]"); ax[1].grid(alpha=.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return gt[-1], od[-1]


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(table().round(4).to_string(index=False))
    g, o = plot(OUT / "plots" / "step2_strafe_forward_arm.png")
    print(f"\nend pose  plant: x={g[0]:.4f} y={g[1]:.4f} th={np.degrees(g[2]):.2f} deg"
          f"\n          odom : x={o[0]:.4f} y={o[1]:.4f} th={np.degrees(o[2]):.2f} deg")
