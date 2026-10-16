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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
