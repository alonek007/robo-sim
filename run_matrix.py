"""Experiment matrix: poses x payloads x paths x modes {baseline, corrected, corrected+posture} x repeats.

usage: python run_matrix.py --stage 1
       python run_matrix.py --stage 2 [--motor soft|stiff]
Writes results/matrix_stage<N>[_<motor>].csv (one row per run) and results/hypotheses_stage<N>[_<motor>].json.
Baseline and corrected share the same physical run (they differ only in the estimator); the posture-rule mode
is its own physical run. Stage 2 physics is deterministic, so repeats differ only by log noise.
"""
import sys, pathlib, json, argparse, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import params as P
from analysis.odometry import estimate
from analysis.metrics import score_run
from analysis.hypotheses import evaluate
from model.load_aware import SlipParams, TRUE
from model.posture import apply_rule
import paths as paths_mod

ROOT = pathlib.Path(__file__).resolve().parent
CELLS = list(itertools.product(P.POSES, P.PAYLOADS, P.PATH_LEG_ANGLES_DEG))


def fold_triggered(pose, payload, path):
    q = P.POSES[pose]
    return any(apply_rule(q, payload, a)[1:] != tuple(q[1:]) for a in paths_mod.leg_angles(path))


def rows_for(pose, payload, path, rep, log_plain, log_rule, slip, extra):
    out = []
    for mode, (kind, rule) in P.MODES.items():
        log = log_rule if rule else log_plain
        m = score_run(log, estimate(log, payload, kind, slip) if kind == "corrected" else estimate(log, payload, "standard"), path)
        out.append(dict(pose=pose, payload=payload, path=path, mode=mode, repeat=rep, **m,
                        fold_triggered=bool(rule and fold_triggered(pose, payload, path)), **extra))
    return out


def stage1_cell(cell):
    from stage1_analytic.simulate import run_one
    pose, payload, path = cell
    rows = []
    for rep in range(P.N_REPEATS):
        rows += rows_for(pose, payload, path, rep, run_one(pose, payload, path, False, rep),
                         run_one(pose, payload, path, True, rep), TRUE, dict(stage=1, motor="n/a", fit_group="n/a"))
    return rows


def run_stage1():
    from multiprocessing import Pool
    with Pool(4) as p:
        return [r for rows in p.map(stage1_cell, CELLS) for r in rows]


def run_stage2(motor):
    from stage2_mujoco.simulate import prefetch, cached_clean_log, apply_noise
    fit = json.load(open(ROOT / "results" / f"fit_params_{motor}.json"))
    slip = SlipParams(kappa=fit["kappa"], c=fit["c_fixed"], f_r=fit["f_r_at_fixed_c"])
    prefetch([(*c, rule) for c in CELLS for rule in (False, True)], motor)
    rows = []
    for pose, payload, path in CELLS:
        plain = cached_clean_log(pose, payload, path, False, motor)
        rule = cached_clean_log(pose, payload, path, True, motor)
        group = "train" if pose in P.FIT_TRAIN_POSES else "test"
        for rep in range(P.N_REPEATS):
            rows += rows_for(pose, payload, path, rep, apply_noise(plain, 3000 + rep), apply_noise(rule, 3000 + rep), slip,
                             dict(stage=2, motor=motor, fit_group=group))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=(1, 2))
    ap.add_argument("--motor", default=P.DEFAULT_MOTOR, choices=list(P.WHEEL_MOTORS))
    a = ap.parse_args()
    tag = "stage1" if a.stage == 1 else f"stage2_{a.motor}"
    df = pd.DataFrame(run_stage1() if a.stage == 1 else run_stage2(a.motor))
    df.to_csv(ROOT / "results" / f"matrix_{tag}.csv", index=False)
    res = evaluate(df)
    json.dump(res, open(ROOT / "results" / f"hypotheses_{tag}.json", "w"), indent=1, default=float)
    print(f"{tag}: {len(df)} runs ({df.groupby(['pose','payload','path']).ngroups} conditions x {P.N_REPEATS} repeats x {len(P.MODES)} modes)")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
