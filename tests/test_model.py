"""Plain-assert self-checks: python tests/test_model.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import params as P
from model.arm import arm_points
from model.com import static_state, total_mass_and_com, wheel_loads, mass_points


def test_fk_reach():
    p = arm_points((0, 0, 0))
    assert np.allclose(p["tip"], [P.L2 + P.L3, 0, P.D1])
    p = arm_points((90, 0, 0))
    assert np.allclose(p["tip"], [0, P.L2 + P.L3, P.D1], atol=1e-12)
    p = arm_points((0, 90, 0))
    assert np.allclose(p["tip"], [0, 0, P.D1 + P.L2 + P.L3], atol=1e-12)
    p = arm_points((30, 40, -20))   # |tip - shoulder| segments keep their lengths
    assert np.isclose(np.linalg.norm(p["elbow"] - p["shoulder"]), P.L2)
    assert np.isclose(np.linalg.norm(p["tip"] - p["elbow"]), P.L3)


def test_loads_balance():
    for q in P.POSES.values():
        for pay in P.PAYLOADS:
            s = static_state(q, pay)
            assert np.isclose(s["N"].sum(), s["M"] * P.G)                      # force
            xs, ys = P.WHEEL_XY[:, 0], P.WHEEL_XY[:, 1]
            assert np.isclose((s["N"] * xs).sum(), s["M"] * P.G * s["x_c"])    # moment about y
            assert np.isclose((s["N"] * ys).sum(), s["M"] * P.G * s["y_c"])    # moment about x


def test_tucked_is_centred():
    s = static_state(P.POSES["tucked"], 0.0)
    assert abs(s["x_c"]) < 1e-3 and abs(s["y_c"]) < 1e-6


def test_clip_when_unstable():
    n = wheel_loads(2.0, 0.2, 0.0)          # |x_c| > a -> rear wheels lift
    assert n.min() == 0.0 and n.sum() > 0


def test_ik_pinv_roundtrip():
    from model.mecanum import wheel_speeds, standard_odometry
    v = np.array([0.2, -0.1, 0.3])
    assert np.allclose(standard_odometry(wheel_speeds(v)), v)


def test_reduces_to_standard_odometry():
    from model.mecanum import wheel_speeds, standard_odometry
    from model.load_aware import body_velocity, SlipParams
    om = wheel_speeds([0.2, 0.05, 0.1])
    N = np.full(4, 4.4)
    v = body_velocity(om, N, 1.78, SlipParams(kappa=0.0, c=15.0, f_r=0.0))
    assert np.allclose(v, standard_odometry(om))


def test_yaw_coupling_mechanism():
    """Yaw drift needs CoM offset *perpendicular* to travel; none when CoM is on the travel line."""
    from model.mecanum import wheel_speeds
    from model.load_aware import body_velocity
    def wz(v, q, pay=0.4):
        s = static_state(q, pay)
        return body_velocity(wheel_speeds(v), s["N"], s["M"])[2]
    fwd, side = P.POSES["forward"], P.POSES["side"]
    assert abs(wz([0.2, 0, 0], fwd)) < 1e-9          # arm forward, drive forward
    assert abs(wz([0, 0.2, 0], side)) < 1e-9         # arm sideways, drive sideways
    assert abs(wz([0, 0.2, 0], fwd)) > 1e-4          # arm forward, strafe  (H1)
    assert abs(wz([0.2, 0, 0], side)) > 1e-4         # arm sideways, drive forward (H2)
    assert wz([0, 0.2, 0], fwd) * wz([0, -0.2, 0], fwd) < 0   # sign flips with direction


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
