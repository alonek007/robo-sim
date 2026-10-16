"""Step 6: fit (kappa, f_r/c) of the load-aware model to MuJoCo logs by least squares.

Train on P.FIT_TRAIN_POSES, test on the held-out P.FIT_TEST_POSES (split fixed in params.py before any fit).
c and f_r are not separately identifiable (c cancels except in f_r/c), so c is held at P.FIT_C_FIXED and
f_r = (f_r/c) * c is reported.   usage: python stage2_mujoco/fit_params.py [--motor soft|stiff]
"""
import sys, pathlib, json, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
from stage2_mujoco.simulate import prefetch, cached_clean_log, apply_noise
from model.load_aware import SlipParams
from analysis.odometry import estimate
from analysis.metrics import score_run, wrap
from analysis.logio import relative_gt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCALE = 1e-3                                    # parameters are optimised as theta / SCALE (both O(1))


def slip_from_theta(theta):
    kappa, rho = np.asarray(theta) * SCALE
    return SlipParams(kappa=kappa, c=P.FIT_C_FIXED, f_r=rho * P.FIT_C_FIXED)


def conditions(poses):
    return [(pose, pay, path, False) for pose in poses for pay in P.PAYLOADS for path in P.PATH_LEG_ANGLES_DEG]


def load_runs(poses, motor, seed0):
    runs = []
    for i, c in enumerate(conditions(poses)):
        clean = cached_clean_log(*c, motor)
        runs.append(dict(pose=c[0], payload=c[1], path=c[2], log=apply_noise(clean, seed0 + i)))
    return runs


def residuals(theta, runs):
    slip = slip_from_theta(theta)
    out = []
    for r in runs:
        est, gt = estimate(r["log"], r["payload"], "corrected", slip), relative_gt(r["log"])
        idx = np.arange(0, len(gt), int(round(P.FIT_SAMPLE_DT / P.DT)))
        d = est[idx] - gt[idx]
        out.append(np.concatenate([d[:, 0], d[:, 1], P.L * wrap(d[:, 2])]))     # heading error scaled to metres at radius L
    return np.concatenate(out)


def fit(runs):
    best = None
    for x0 in ([0.5, 0.5], [10.0, 5.0], [-10.0, 0.0]):
        sol = least_squares(residuals, x0, args=(runs,), bounds=([-100, 0], [100, 50]), x_scale=1.0)
        if best is None or sol.cost < best.cost:
            best = sol
    return best


def evaluate(runs, slip, group):
    rows = []
    nominal = SlipParams()
    for r in runs:
        row = dict(group=group, pose=r["pose"], payload=r["payload"], path=r["path"])
        for name, kind, sp in (("std", "standard", None), ("fit", "corrected", slip), ("nom", "corrected", nominal)):
            m = score_run(r["log"], estimate(r["log"], r["payload"], kind, sp) if sp else estimate(r["log"], r["payload"], kind), r["path"])
            row[f"err_{name}_mm"] = 1e3 * m["odom_err_m"]
            row[f"hderr_{name}_deg"] = m["odom_heading_err_deg"]
        row["drift_deg"] = m["heading_drift_deg"]
        rows.append(row)
    return pd.DataFrame(rows)


def main(motor):
    prefetch(conditions(P.FIT_TRAIN_POSES + P.FIT_TEST_POSES), motor)
    train = load_runs(P.FIT_TRAIN_POSES, motor, 1000)
    test = load_runs(P.FIT_TEST_POSES, motor, 2000)
    sol = fit(train)
    theta = sol.x
    slip = slip_from_theta(theta)
    # standard errors from the Jacobian (residual variance scaled)
    dof = max(1, len(sol.fun) - 2)
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * (2 * sol.cost / dof)
    se = np.sqrt(np.diag(cov)) * SCALE
    rms = lambda runs, th: float(np.sqrt(np.mean(residuals(th, runs) ** 2)))
    res = dict(motor=motor, kappa=slip.kappa, fr_over_c=slip.f_r / slip.c, c_fixed=slip.c, f_r_at_fixed_c=slip.f_r,
               kappa_se=se[0], fr_over_c_se=se[1], hit_bounds=bool(np.any(np.isclose(theta, [-100, 0])) or np.any(np.isclose(theta, [100, 50]))),
               success=bool(sol.success), rms_train_fit_mm=1e3 * rms(train, theta), rms_train_standard_mm=1e3 * rms(train, [0.0, 0.0]),
               rms_test_fit_mm=1e3 * rms(test, theta), rms_test_standard_mm=1e3 * rms(test, [0.0, 0.0]),
               truth_stage1=dict(kappa=P.KAPPA_TRUE, fr_over_c=P.FR_TRUE / P.C_TRUE))
    json.dump(res, open(ROOT / "results" / f"fit_params_{motor}.json", "w"), indent=2)
    ev = pd.concat([evaluate(train, slip, "train"), evaluate(test, slip, "test")])
    ev.to_csv(ROOT / "results" / f"fit_eval_{motor}.csv", index=False)

    pd.set_option("display.width", 200)
    print(json.dumps(res, indent=1))
    agg = ev.groupby("group").agg(n=("pose", "size"),
                                  err_std_mm=("err_std_mm", "median"), err_fit_mm=("err_fit_mm", "median"), err_nom_mm=("err_nom_mm", "median"),
                                  hderr_std=("hderr_std_deg", lambda x: np.median(np.abs(x))),
                                  hderr_fit=("hderr_fit_deg", lambda x: np.median(np.abs(x))),
                                  frac_improved=("err_fit_mm", lambda x: np.nan))
    ev["improved"] = ev.err_fit_mm < ev.err_std_mm
    agg["frac_improved"] = ev.groupby("group").improved.mean()
    print("\nmedians per group (end-point error [mm], |heading error| [deg]):\n", agg.round(3).to_string())
    print("\nheld-out test poses, per condition:\n", ev[ev.group == "test"].round(2).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for g, col in (("train", "C0"), ("test", "C3")):
        d = ev[ev.group == g]
        ax[0].scatter(d.err_std_mm, d.err_fit_mm, c=col, label=g, s=22)
        ax[1].scatter(d.hderr_std_deg.abs(), d.hderr_fit_deg.abs(), c=col, label=g, s=22)
        ax[2].scatter(d.drift_deg.abs(), d.hderr_fit_deg.abs(), c=col, label=g, s=22)
    for a, t, xl, yl in ((ax[0], "end-point error [mm]", "standard", "corrected (fitted)"),
                         (ax[1], "|heading error| [deg]", "standard", "corrected (fitted)"),
                         (ax[2], "residual |heading error| vs true |heading drift| [deg]", "true |drift|", "corrected (fitted) |error|")):
        m = max(a.get_xlim()[1], a.get_ylim()[1]); a.plot([0, m], [0, m], "k--", lw=1)
        a.set_title(t, fontsize=10); a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=.3); a.legend()
    fig.suptitle(f"MuJoCo ({motor} motors): fitted corrected odometry vs standard; below the dashed line = corrected is better")
    fig.tight_layout(); fig.savefig(ROOT / "results" / "plots" / f"step6_fit_{motor}.png", dpi=105)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--motor", default=P.DEFAULT_MOTOR, choices=list(P.WHEEL_MOTORS))
    main(ap.parse_args().motor)
