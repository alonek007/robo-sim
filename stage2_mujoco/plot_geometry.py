"""Matplotlib schematic of the generated MJCF (no GL needed): top and side view of rollers, wheels, arm."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
from stage2_mujoco.simulate import get_model

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "plots"


def main(pose="forward", payload=0.4):
    m = get_model(payload); d = mujoco.MjData(m)
    for j, a in zip(("q1", "q2", "q3"), np.radians(P.POSES[pose])):
        d.qpos[m.joint(j).qposadr[0]] = a
    d.qpos[2] = P.WHEEL_R
    mujoco.mj_forward(m, d)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    for gi in range(m.ngeom):
        t = m.geom_type[gi]
        if t == mujoco.mjtGeom.mjGEOM_PLANE: continue
        c, R, sz = d.geom_xpos[gi], d.geom_xmat[gi].reshape(3, 3), m.geom_size[gi]
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ends = np.array([c - R[:, 2] * sz[1], c + R[:, 2] * sz[1]])
            col, lw = ("k", 1.8) if m.geom_contype[gi] == 2 else ("C0", 3)
            for a, (u, v) in zip(ax[:2], ((0, 1), (0, 2))): a.plot(ends[:, u], ends[:, v], col, lw=lw)
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            for a, (u, v) in zip(ax[:2], ((0, 1), (0, 2))):
                a.plot(c[u], c[v], "s", color="C2" if sz[0] > 0.05 else "C1", ms=6)
    for a in ax[:2]: a.set_aspect("equal"); a.grid(alpha=.3)
    ax[0].set_title(f"top view ({pose}, {payload:g} kg): roller axes black"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
    ax[1].set_title("side view"); ax[1].set_xlabel("x"); ax[1].set_ylabel("z"); ax[1].axhline(0, color="gray")
    # zoom on front-left wheel: roller layout seen along the axle (x-z) with the roller axes
    w = m.body("wheel_FL").id
    for gi in range(m.ngeom):
        if m.geom_contype[gi] == 2 and m.body(m.geom_bodyid[gi]).name.startswith("FL_r"):
            c, R, sz = d.geom_xpos[gi], d.geom_xmat[gi].reshape(3, 3), m.geom_size[gi]
            e = np.array([c - R[:, 2] * sz[1], c + R[:, 2] * sz[1]])
            ax[2].plot(e[:, 0], e[:, 2], "k", lw=3); ax[2].plot(e[:, 0], e[:, 2], "C1", lw=0, marker=".")
    ax[2].axhline(0, color="gray"); ax[2].set_aspect("equal"); ax[2].grid(alpha=.3)
    ax[2].set_title("FL wheel along its axle (x-z): 10 rollers, floor at z=0"); ax[2].set_xlabel("x"); ax[2].set_ylabel("z")
    fig.tight_layout(); fig.savefig(OUT / "step4_model_schematic.png", dpi=100)


if __name__ == "__main__":
    main()
