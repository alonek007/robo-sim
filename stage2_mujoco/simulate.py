"""Stage 2: MuJoCo simulation. Writes logs in the common schema (analysis/logio.py)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import mujoco
import params as P
import paths
from model.mecanum import wheel_speeds
from model.posture import apply_rule
from analysis.logio import COLUMNS
from stage2_mujoco.build_mjcf import build_xml

SUBSTEPS = int(round(P.DT / P.MJ_TIMESTEP))
assert abs(SUBSTEPS * P.MJ_TIMESTEP - P.DT) < 1e-12, "P.DT must be a multiple of P.MJ_TIMESTEP"
_MODEL_CACHE = {}


def get_model(payload):
    if payload not in _MODEL_CACHE:
        _MODEL_CACHE[payload] = mujoco.MjModel.from_xml_string(build_xml(payload))
    return _MODEL_CACHE[payload]


def _yaw(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def arm_schedule(prof, q_req, payload, posture_rule):
    """Arm joint targets (N+1,3) deg. Leg-wise posture rule; re-pointing is interpolated over the corner pause."""
    n = len(prof["t"])
    q_leg = {}
    for i, a in zip(prof["leg"], prof["angle_deg"]):
        if i not in q_leg:
            q_leg[i] = np.array(apply_rule(q_req, payload, a) if posture_rule else q_req, float)
    q = np.array([q_leg[i] for i in prof["leg"]])
    # during a pause (v_cmd == 0 between legs) blend from the previous leg's pose to the next leg's pose
    moving = np.linalg.norm(prof["v_cmd"][:, :2], axis=1) > 0
    legs = sorted(q_leg)
    for a, b in zip(legs[:-1], legs[1:]):
        idx = np.where((prof["leg"] == a) & ~moving)[0]
        idx = idx[idx > np.where((prof["leg"] == a) & moving)[0].max()]
        if len(idx):
            w = (np.arange(len(idx)) + 1) / len(idx)
            q[idx] = (1 - w)[:, None] * q_leg[a] + w[:, None] * q_leg[b]
    return q


def simulate(payload, q_target_deg, v_cmd, seed, extra=None):
    """Run the physics. q_target_deg, v_cmd: (N+1,3) per log row. Returns log DataFrame (+ extras dict if asked)."""
    model = get_model(payload)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    jid = lambda n: model.joint(n).qposadr[0]
    vid = lambda n: model.joint(n).dofadr[0]
    arm = ["q1", "q2", "q3"]
    wheels = [f"j_{n}" for n in P.WHEEL_NAMES]
    q0 = np.radians(q_target_deg[0])
    for j, a in zip(arm, q0):
        data.qpos[jid(j)] = a
    data.ctrl[4:7] = q0
    mujoco.mj_forward(model, data)

    for _ in range(int(round(P.SETTLE_T / P.MJ_TIMESTEP))):
        mujoco.mj_step(model, data)

    n = len(v_cmd)
    om_cmd = np.array([wheel_speeds(v) for v in v_cmd])
    rec = np.zeros((n, 12))
    wheel_ang = np.zeros((n, 4))
    nf = np.zeros((n, 4)) if extra is not None else None
    zr = np.zeros((n, 3)) if extra is not None else None   # chassis z, roll, pitch
    x0 = data.qpos[0:2].copy()
    for k in range(n):
        wheel_ang[k] = [data.qpos[jid(w)] for w in wheels]
        rec[k, 5:8] = np.degrees([data.qpos[jid(j)] for j in arm])
        rec[k, 9] = data.qpos[0]; rec[k, 10] = data.qpos[1]
        rec[k, 11] = _yaw(data.qpos[3:7])
        rec[k, 8] = data.qvel[5]
        if nf is not None:
            nf[k] = _wheel_normal_forces(model, data)
            qw, qx, qy, qz = data.qpos[3:7]
            zr[k] = [data.qpos[2], np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)),
                     np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))]
        data.ctrl[0:4] = om_cmd[k]
        data.ctrl[4:7] = np.radians(q_target_deg[k])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
    rec[:, 11] = np.unwrap(rec[:, 11])
    omega = np.zeros((n, 4))
    omega[:-1] = np.diff(wheel_ang, axis=0) / P.DT            # encoder-style interval average
    omega += rng.normal(0, P.OMEGA_NOISE, omega.shape) * (np.arange(n) < n - 1)[:, None]
    rec[:, 1:5] = omega
    rec[:, 8] += rng.normal(0, P.GYRO_NOISE, n)
    rec[:, 0] = np.arange(n) * P.DT
    df = pd.DataFrame(rec, columns=COLUMNS)
    if extra is not None:
        extra["wheel_normal_force"] = nf
        extra["z_roll_pitch"] = zr
    return df


def _wheel_normal_forces(model, data):
    """Sum of floor-contact normal forces per wheel (roller bodies grouped by wheel name)."""
    out = np.zeros(4)
    f6 = np.zeros(6)
    floor = model.geom("floor").id
    for i in range(data.ncon):
        c = data.contact[i]
        if floor not in (c.geom1, c.geom2):
            continue
        g = c.geom2 if c.geom1 == floor else c.geom1
        bname = model.body(model.geom_bodyid[g]).name
        mujoco.mj_contactForce(model, data, i, f6)
        for w, wn in enumerate(P.WHEEL_NAMES):
            if bname.startswith(wn + "_r"):
                out[w] += f6[0]
    return out


def run_one(pose, payload, path, posture_rule, seed, extra=None):
    prof = paths.build(path)
    q = arm_schedule(prof, P.POSES[pose], payload, posture_rule)
    return simulate(payload, q, prof["v_cmd"], seed, extra)
