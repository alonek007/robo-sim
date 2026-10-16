"""H1-H4 evaluation on a results DataFrame (one row per run). Criteria: params.H_CRIT."""
import numpy as np, pandas as pd
from scipy import stats
import params as P

C = P.H_CRIT
KEY = ["pose", "payload", "path"]


def _paired(df, pose_a, path_a, pose_b, path_b, payload):
    """Per-repeat |heading drift| of two conditions (physical, no posture rule), aligned by repeat."""
    d = df[(df["mode"] == "baseline") & (df.payload == payload)]
    a = d[(d.pose == pose_a) & (d.path == path_a)].set_index("repeat").heading_drift_deg.abs().sort_index()
    b = d[(d.pose == pose_b) & (d.path == path_b)].set_index("repeat").heading_drift_deg.abs().sort_index()
    return a, b


def drift_test(df, ext_pose, path):
    """H1 / H2: extended arm vs tucked arm, same path."""
    out = {}
    for payload in (0.0,) + tuple(C["h12_payloads"]):
        ext, tuck = _paired(df, ext_pose, path, "tucked", path, payload)
        diff = (ext - tuck).values
        n = len(diff)
        sd = diff.std(ddof=1) if n > 1 else 0.0
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
        lo, hi = diff.mean() - half, diff.mean() + half
        ratio = ext.median() / max(tuck.median(), 1e-9)
        ok = bool(ratio >= C["h12_ratio"] and np.median(diff) >= C["h12_min_deg"] and lo > 0)
        out[payload] = dict(ext_median_deg=ext.median(), tucked_median_deg=tuck.median(), ratio=ratio,
                            median_diff_deg=float(np.median(diff)), ci95=(lo, hi), criteria_met=ok)
    scored = [out[p]["criteria_met"] for p in C["h12_payloads"]]
    return dict(per_payload=out, passed=bool(all(scored)))


def h3(df, poses=None):
    """Corrected vs standard end-point error, per (pose, payload, path) condition (mean over repeats)."""
    d = df if poses is None else df[df.pose.isin(poses)]
    cell = d.groupby(KEY + ["mode"]).odom_err_m.mean().unstack("mode")
    hd = d.groupby(KEY + ["mode"]).odom_heading_err_deg.apply(lambda x: x.abs().mean()).unstack("mode")
    frac = float((cell["corrected"] < cell["baseline"]).mean())
    red = float((cell["baseline"].median() - cell["corrected"].median()) / cell["baseline"].median())
    rel = float(((cell["baseline"] - cell["corrected"]) / cell["baseline"]).median())
    return dict(n_conditions=len(cell), frac_improved=frac, median_err_baseline_mm=1e3 * cell["baseline"].median(),
                median_err_corrected_mm=1e3 * cell["corrected"].median(), reduction_of_median=red,
                median_of_relative_reduction=rel,
                median_abs_heading_err_baseline_deg=float(hd["baseline"].median()),
                median_abs_heading_err_corrected_deg=float(hd["corrected"].median()),
                passed=bool(frac >= C["h3_frac"] and red >= C["h3_reduction"]))


def h4(df, subset=None):
    """Posture rule vs corrected-without-rule on actual drift and path deviation (per-condition means)."""
    d = df if subset is None else df[subset(df)]
    out = {}
    for col, name in (("heading_drift_deg", "heading_drift"), ("path_dev_m", "path_dev")):
        x = d.assign(v=d[col].abs()).groupby(KEY + ["mode"]).v.mean().unstack("mode")
        no, ru = x["corrected"], x["corrected+posture"]
        out[name] = dict(median_no_rule=float(no.median()), median_rule=float(ru.median()),
                         reduction=float((no.median() - ru.median()) / no.median()),
                         frac_conditions_improved=float((ru < no).mean()))
    out["n_conditions"] = int(len(x))
    out["passed"] = bool(all(out[k]["reduction"] >= C["h4_reduction"] for k in ("heading_drift", "path_dev")))
    return out


def evaluate(df):
    res = {
        "H1 (arm forward + strafe vs tucked)": drift_test(df, "forward", "strafe"),
        "H2 (arm sideways + forward vs tucked)": drift_test(df, "side", "forward"),
        "H3 all poses": h3(df),
        "H4 all conditions": h4(df),
        "H4 extended-arm conditions only (forward/side pose, payload >= 0.2)":
            h4(df, lambda d: d.pose.isin(["forward", "side"]) & (d.payload >= 0.2)),
    }
    if "fit_group" in df and (df.fit_group == "test").any():
        res["H3 held-out poses (verdict for stage 2)"] = h3(df, P.FIT_TEST_POSES)
        res["H3 training poses (in-sample)"] = h3(df, P.FIT_TRAIN_POSES)
    return res
