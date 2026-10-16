"""Stage 2: MuJoCo simulation. Writes logs in the common schema (analysis/logio.py)."""
import sys, pathlib, hashlib
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


def get_model(payload, motor=None):
    key = (payload, motor or P.DEFAULT_MOTOR)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = mujoco.MjModel.from_xml_string(build_xml(payload, motor))
    return _MODEL_CACHE[key]


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


def simulate(payload, q_target_deg, v_cmd, motor=None, extra=None):
    """Run the physics (deterministic). q_target_deg, v_cmd: (N+1,3) per log row.
    Returns the noise-free log DataFrame (see apply_noise); extras dict is filled if given."""
    model = get_model(payload, motor)
    data = mujoco.MjData(model)
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
    rec[:, 1:5] = omega
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


def apply_noise(log, seed):
    """Encoder noise on omega and gyro noise on imu (the physics is deterministic, so repeats differ only by this)."""
    rng = np.random.default_rng(seed)
    out = log.copy()
    n = len(out)
    cols = ["omega1", "omega2", "omega3", "omega4"]
    out[cols] = out[cols].values + rng.normal(0, P.OMEGA_NOISE, (n, 4)) * (np.arange(n) < n - 1)[:, None]
    out["imu"] = out["imu"].values + rng.normal(0, P.GYRO_NOISE, n)
    return out


def clean_log(pose, payload, path, posture_rule, motor=None, extra=None):
    prof = paths.build(path)
    q = arm_schedule(prof, P.POSES[pose], payload, posture_rule)
    return simulate(payload, q, prof["v_cmd"], motor, extra)


def run_one(pose, payload, path, posture_rule, seed, motor=None, extra=None):
    return apply_noise(clean_log(pose, payload, path, posture_rule, motor, extra), seed)


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_FILES = [_ROOT / "params.py", _ROOT / "paths.py", _ROOT / "model" / "posture.py", _ROOT / "model" / "com.py",
              pathlib.Path(__file__), pathlib.Path(__file__).with_name("build_mjcf.py")]


def cached_clean_log(pose, payload, path, posture_rule, motor=None, cache_root=_ROOT / "results" / "logs" / "stage2_clean"):
    """clean_log with an on-disk cache. The file name carries a hash of the model XML and all source files that
    influence the physics, so any change to them re-runs the simulation instead of returning stale data."""
    motor = motor or P.DEFAULT_MOTOR
    h = hashlib.sha1(build_xml(payload, motor).encode())
    for f in _SRC_FILES:
        h.update(f.read_bytes())
    f = pathlib.Path(cache_root) / motor / f"{pose}_{payload:g}kg_{path}_{'rule' if posture_rule else 'norule'}_{h.hexdigest()[:10]}.csv"
    if f.exists():
        return pd.read_csv(f)[COLUMNS]
    log = clean_log(pose, payload, path, posture_rule, motor)
    f.parent.mkdir(parents=True, exist_ok=True)
    for old in f.parent.glob(f"{pose}_{payload:g}kg_{path}_{'rule' if posture_rule else 'norule'}_*.csv"):
        old.unlink()                                     # drop stale versions of this condition
    log.to_csv(f, index=False)
    return log


def _job(args):
    return cached_clean_log(*args)


def prefetch(conditions, motor=None, processes=4):
    """Fill the cache for many (pose, payload, path, rule) conditions in parallel."""
    from multiprocessing import Pool
    todo = [(*c, motor) for c in conditions]
    with Pool(processes) as pool:
        pool.map(_job, todo, chunksize=1)
