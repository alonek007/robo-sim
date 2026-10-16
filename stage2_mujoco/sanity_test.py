"""Step 4 sanity test: arm tucked, no payload -> forward / strafe / spin behave as commanded."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
import paths
from stage2_mujoco.simulate import simulate
from analysis.logio import relative_gt
from model.mecanum import wheel_speeds

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
POSE, PAYLOAD = "tucked", 0.0


def spin_profile(angle_rad=np.pi / 2):
    """Pure rotation: peak rim speed = P.SPEED at radius L."""
    wmax = P.SPEED / P.L
    w = paths.trapezoid(angle_rad, wmax, P.RAMP_T, P.DT)
    v = np.zeros((len(w) + 1, 3)); v[:-1, 2] = w
    return v, angle_rad


def profile(name):
    if name == "spin":
        return spin_profile()
    p = paths.build(name)
    return p["v_cmd"], None


def main():
    q = np.tile(P.POSES[POSE], (1, 1))
    results, runs, ripple = [], {}, {}
    for name in ("forward", "strafe", "diagonal", "spin"):
        v, ang = profile(name)
        ex = {}
        log = simulate(PAYLOAD, np.tile(P.POSES[POSE], (len(v), 1)), v, extra=ex)
        zrp = ex["z_roll_pitch"]; cruise = slice(int(P.RAMP_T / P.DT) + 20, len(v) - int(P.RAMP_T / P.DT) - 20)
        ripple[name] = (np.ptp(zrp[cruise, 0]) * 1e3, np.degrees(np.ptp(zrp[cruise, 1])), np.degrees(np.ptp(zrp[cruise, 2])))
        gt = relative_gt(log)
        runs[name] = (log, gt)
        cmd = (v[:, :3].sum(0) * P.DT)
        results.append(dict(test=name, cmd_x=cmd[0], cmd_y=cmd[1], cmd_theta_deg=np.degrees(cmd[2]),
                            meas_x=gt[-1, 0], meas_y=gt[-1, 1], meas_theta_deg=np.degrees(gt[-1, 2])))
    df = pd.DataFrame(results)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))

    print("\nbody motion during cruise (peak-to-peak): " + "; ".join(
        f"{k}: z {z:.2f} mm, roll {r:.2f} deg, pitch {p:.2f} deg" for k, (z, r, p) in ripple.items()))
    ratio = {r.test: (r.meas_x, r.meas_y, r.meas_theta_deg) for r in df.itertuples()}
    f, a = plt.subplots(2, 4, figsize=(16, 7))
    for j, (name, (log, gt)) in enumerate(runs.items()):
        v, _ = profile(name)
        om_cmd = np.array([wheel_speeds(x) for x in v])
        for i, w in enumerate(P.WHEEL_NAMES):
            a[0, j].plot(log.t, log[f"omega{i+1}"], color=f"C{i}", label=f"{w} measured")
            a[0, j].plot(log.t, om_cmd[:, i], "k--", lw=0.8)
        a[0, j].set_title(f"{name}: wheel speeds (dashed = command)", fontsize=10)
        a[0, j].set_xlabel("t [s]"); a[0, j].set_ylabel("omega [rad/s]"); a[0, j].grid(alpha=.3)
        a[1, j].plot(log.t, np.degrees(gt[:, 2]), label="heading [deg]")
        a[1, j].plot(log.t, gt[:, 0] * 100, label="x [cm]"); a[1, j].plot(log.t, gt[:, 1] * 100, label="y [cm]")
        a[1, j].grid(alpha=.3); a[1, j].set_xlabel("t [s]")
        if j == 0: a[0, j].legend(fontsize=7); a[1, j].legend(fontsize=8)
    f.tight_layout(); f.savefig(OUT / "plots" / "step4_sanity.png", dpi=105)
    df.to_csv(OUT / "step4_sanity.csv", index=False)


if __name__ == "__main__":
    main()
