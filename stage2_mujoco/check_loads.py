"""Step 5 check: per-wheel floor normal force in MuJoCo at rest vs eq.3, for every pose x payload."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import params as P
from model.com import static_state
from stage2_mujoco.simulate import simulate

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"


def main():
    rows = []
    n = 100                                    # 1 s of rest after the settling phase
    for pose, q in P.POSES.items():
        for pay in P.PAYLOADS:
            ex = {}
            log = simulate(pay, np.tile(q, (n, 1)), np.zeros((n, 3)), seed=0, extra=ex)
            nf = ex["wheel_normal_force"].mean(0)
            st = static_state(q, pay)
            rows.append(dict(pose=pose, payload=pay, s_eq3=st["s"],
                             **{f"N_{w}_mj": v for w, v in zip(P.WHEEL_NAMES, nf)},
                             **{f"N_{w}_eq3": v for w, v in zip(P.WHEEL_NAMES, st["N"])},
                             total_mj=nf.sum(), total_eq3=st["M"] * P.G,
                             max_abs_err_N=np.abs(nf - st["N"]).max(),
                             q_sag_deg=np.abs(log[["q1", "q2", "q3"]].values.mean(0) - q).max()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "step5_load_check.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["pose", "payload", "s_eq3"] + [f"N_{w}_mj" for w in P.WHEEL_NAMES] + [f"N_{w}_eq3" for w in P.WHEEL_NAMES] + ["max_abs_err_N", "q_sag_deg"]
    print(df[cols].round(2).to_string(index=False))
    print(f"\nlargest |MuJoCo - eq.3| over all wheels: {df.max_abs_err_N.max():.2f} N")


if __name__ == "__main__":
    main()
