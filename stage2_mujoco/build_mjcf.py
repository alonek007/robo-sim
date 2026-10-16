"""Generate the MJCF model from params.py.

Chassis body frame: origin at the geometric chassis centre, at axle height (z = WHEEL_R above the floor).
Wheels: hub body on a hinge about +y (positive speed drives +x for all four wheels, matching J), with
N_ROLLERS capsule rollers on free hinges; each roller axis lies in the wheel's tangent/axial plane at 45 deg
to the wheel axis. Handedness (derived from the rows of J): a roller rolls freely perpendicular to its axis and is
constrained along it, so FL and RR have their ground-contact roller axis along (1,-1), FR and RL along (1,1).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import brentq
import params as P

WHEEL_SIGMA = {"FL": -1, "FR": +1, "RL": +1, "RR": -1}
ROLLER_ANG = np.radians(P.ROLLER_ANGLE_DEG)


def _roller_axis(phi, sigma):
    """Unit roller axis in the wheel frame for a roller at rim angle phi (wheel axis = +y)."""
    return np.array([-np.cos(ROLLER_ANG) * np.sin(phi),
                     sigma * np.sin(ROLLER_ANG),
                     np.cos(ROLLER_ANG) * np.cos(phi)])


def hub_height_profile(r_c, half_len, n_theta=400, n_s=201):
    """Hub height above the lowest roller point, for wheel rotation theta over one roller pitch."""
    pitch = 2 * np.pi / P.N_ROLLERS
    s = np.linspace(-half_len, half_len, n_s)
    h = []
    for th in np.linspace(0, pitch, n_theta, endpoint=False):
        zmin = np.inf
        for k in range(P.N_ROLLERS):
            phi = k * pitch + th
            c = r_c * np.array([np.cos(phi), 0.0, np.sin(phi)])
            u = _roller_axis(phi, +1)
            z = (c[2] + s * u[2]).min() - P.ROLLER_RADIUS   # lowest point of the capsule
            zmin = min(zmin, z)
        h.append(-zmin)
    return np.array(h)


def roller_geometry():
    """(R_c, half_len): roller-centre circle radius such that the mean hub height (= effective rolling
    radius) is WHEEL_R, and cylinder half-length so neighbouring roller ends just meet.
    The two depend on each other, so iterate."""
    half_len = 0.0
    for _ in range(6):
        f = lambda rc: hub_height_profile(rc, half_len, n_theta=90, n_s=61).mean() - P.WHEEL_R
        r_c = brentq(f, 0.5 * P.WHEEL_R, P.WHEEL_R, xtol=1e-10)
        half_len = P.ROLLER_END_FACTOR * r_c * np.sin(np.pi / P.N_ROLLERS) / np.cos(ROLLER_ANG)
    return r_c, half_len


def _v(a):
    return " ".join(f"{x:.9g}" for x in a)


def build_xml(payload):
    r_c, half_len = roller_geometry()
    z_axle_floor = P.WHEEL_R
    box_z = P.CHASSIS_CLEARANCE + P.CHASSIS_HALF[2] - z_axle_floor          # chassis box centre, body frame
    mount_z = P.CHASSIS_CLEARANCE + 2 * P.CHASSIS_HALF[2] - z_axle_floor    # arm mount (chassis top)
    m_wheel = P.WHEEL_HUB_MASS + P.N_ROLLERS * P.ROLLER_MASS
    m_ch = P.BASE_MASS - 4 * m_wheel
    z_ch_floor = (P.BASE_MASS * P.BASE_COM_Z - 4 * m_wheel * z_axle_floor) / m_ch   # base CoM height after wheels
    bx, by, bz = P.BASE_BOX_DIMS
    inertia = (m_ch / 12) * np.array([by**2 + bz**2, bx**2 + bz**2, bx**2 + by**2])
    mu = f"{P.MU} 0.005 0.0001"

    wheels = []
    for name, (wx, wy), in zip(P.WHEEL_NAMES, P.WHEEL_XY):
        sg = WHEEL_SIGMA[name]
        rollers = []
        for k in range(P.N_ROLLERS):
            phi = 2 * np.pi * k / P.N_ROLLERS
            c = r_c * np.array([np.cos(phi), 0.0, np.sin(phi)])
            u = _roller_axis(phi, sg)
            rollers.append(
                f'<body name="{name}_r{k}" pos="{_v(c)}">'
                f'<joint name="{name}_rj{k}" type="hinge" axis="{_v(u)}" damping="{P.ROLLER_DAMPING}"/>'
                f'<geom type="capsule" fromto="{_v(-half_len * u)} {_v(half_len * u)}" '
                f'size="{P.ROLLER_RADIUS}" mass="{P.ROLLER_MASS}" friction="{mu}" contype="2" conaffinity="1" '
                f'solref="{_v(P.MJ_SOLREF)}" solimp="{_v(P.MJ_SOLIMP)}" rgba="0.2 0.2 0.2 1"/></body>')
        wheels.append(
            f'<body name="wheel_{name}" pos="{wx} {wy} 0">'
            f'<joint name="j_{name}" type="hinge" axis="0 1 0" armature="{P.WHEEL_ARMATURE}"/>'
            f'<geom type="cylinder" fromto="0 {-P.WHEEL_HUB_HALF_WIDTH} 0 0 {P.WHEEL_HUB_HALF_WIDTH} 0" '
            f'size="{P.WHEEL_HUB_RADIUS}" mass="{P.WHEEL_HUB_MASS}" contype="0" conaffinity="0" rgba="0.8 0.3 0.3 1"/>'
            + "".join(rollers) + "</body>")

    sx, sy, sz = P.SERVO_BOX_HALF
    servo = lambda m=P.SERVO_MASS, pos="0 0 0": (
        f'<geom type="box" size="{sx} {sy} {sz}" pos="{pos}" mass="{m}" contype="0" conaffinity="0" rgba="0.1 0.1 0.6 1"/>')
    payload_xml = (f'<body name="payload" pos="{P.L3} 0 0"><geom type="sphere" size="{P.PAYLOAD_RADIUS}" '
                   f'mass="{payload}" contype="0" conaffinity="0" rgba="0.9 0.7 0.1 1"/></body>') if payload > 0 else ""
    arm = (
        f'<body name="turret" pos="0 0 {mount_z}">'
        f'<joint name="q1" type="hinge" axis="0 0 1" damping="{P.ARM_DAMPING}" armature="{P.ARM_ARMATURE}"/>'
        f'{servo()}'
        f'<body name="upper" pos="0 0 {P.D1}">'
        f'<joint name="q2" type="hinge" axis="0 -1 0" damping="{P.ARM_DAMPING}" armature="{P.ARM_ARMATURE}"/>'
        f'{servo()}'
        f'<geom type="capsule" fromto="0 0 0 {P.L2} 0 0" size="0.008" mass="{P.BRACKET_MASS}" contype="0" conaffinity="0" rgba="0.5 0.5 0.5 1"/>'
        f'<body name="fore" pos="{P.L2} 0 0">'
        f'<joint name="q3" type="hinge" axis="0 -1 0" damping="{P.ARM_DAMPING}" armature="{P.ARM_ARMATURE}"/>'
        f'{servo()}'
        f'<geom type="capsule" fromto="0 0 0 {P.L3} 0 0" size="0.008" mass="{P.BRACKET_MASS}" contype="0" conaffinity="0" rgba="0.5 0.5 0.5 1"/>'
        f'{servo(pos=f"{P.L3} 0 0")}'
        f'{payload_xml}'
        '</body></body></body>')

    act = "".join(
        f'<velocity name="w_{n}" joint="j_{n}" kv="{P.WHEEL_KV}" forcerange="{-P.WHEEL_TORQUE_LIMIT} {P.WHEEL_TORQUE_LIMIT}" '
        f'ctrlrange="-40 40"/>' for n in P.WHEEL_NAMES)
    act += "".join(
        f'<position name="a_{j}" joint="{j}" kp="{P.ARM_KP}" kv="{P.ARM_KV}" '
        f'forcerange="{-P.ARM_TORQUE_LIMIT} {P.ARM_TORQUE_LIMIT}"/>' for j in ("q1", "q2", "q3"))

    return f"""<mujoco model="mecanum_arm">
  <compiler angle="radian" inertiafromgeom="auto"/>
  <option timestep="{P.MJ_TIMESTEP}" integrator="implicitfast" cone="elliptic" impratio="{P.MJ_IMPRATIO}" gravity="0 0 -{P.G}"/>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" friction="{mu}" contype="1" conaffinity="2"
          solref="{_v(P.MJ_SOLREF)}" solimp="{_v(P.MJ_SOLIMP)}" rgba="0.9 0.9 0.9 1"/>
    <body name="chassis" pos="0 0 {P.WHEEL_R + 0.004}">
      <freejoint name="root"/>
      <inertial pos="0 0 {z_ch_floor - z_axle_floor}" mass="{m_ch}" diaginertia="{_v(inertia)}"/>
      <geom type="box" size="{_v(P.CHASSIS_HALF)}" pos="0 0 {box_z}" contype="0" conaffinity="0" rgba="0.3 0.6 0.3 1"/>
      {''.join(wheels)}
      {arm}
    </body>
  </worldbody>
  <actuator>{act}</actuator>
</mujoco>"""


if __name__ == "__main__":
    import mujoco
    rc, hl = roller_geometry()
    h = hub_height_profile(rc, hl)
    print(f"roller-centre circle R_c = {rc*1e3:.3f} mm (wheel R = {P.WHEEL_R*1e3:.2f} mm, roller r = {P.ROLLER_RADIUS*1e3:.1f} mm), "
          f"cylinder half-length {hl*1e3:.2f} mm")
    print(f"hub height over one roller pitch: mean {h.mean()*1e3:.3f} mm, pk-pk {np.ptp(h)*1e3:.3f} mm")
    for pay in P.PAYLOADS:
        m = mujoco.MjModel.from_xml_string(build_xml(pay))
        print(f"payload {pay}: nbody={m.nbody} njnt={m.njnt} ngeom={m.ngeom} total mass={m.body_subtreemass[1]:.4f} kg "
              f"(expected {P.BASE_MASS + 3*P.SERVO_MASS + P.GRIPPER_SERVO_MASS + 2*P.BRACKET_MASS + pay:.4f})")
