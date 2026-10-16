"""Eq.2-3: mass-weighted CoM, wheel loads and stability margin."""
import numpy as np
import params as P
from model.arm import arm_points


def mass_points(q_deg, payload):
    """List of (mass, position) point masses in the body frame."""
    p = arm_points(q_deg)
    return [
        (P.BASE_MASS, np.array([0.0, 0.0, P.BASE_COM_Z])),
        (P.SERVO_MASS, p["mount"]),        # q1 servo
        (P.SERVO_MASS, p["shoulder"]),     # q2 servo
        (P.SERVO_MASS, p["elbow"]),        # q3 servo
        (P.GRIPPER_SERVO_MASS, p["tip"]),
        (P.BRACKET_MASS, p["link2_mid"]),
        (P.BRACKET_MASS, p["link3_mid"]),
        (payload, p["tip"]),
    ]


def total_mass_and_com(q_deg, payload):
    pts = mass_points(q_deg, payload)
    m = sum(mi for mi, _ in pts)
    com = sum(mi * pi for mi, pi in pts) / m
    return m, com


def wheel_loads(m, x_c, y_c):
    """Eq.3. Order FL, FR, RL, RR. Clipped at 0 (wheel lifts when s < 0)."""
    n = (m * P.G / 4.0) * (1.0 + np.array([+1, +1, -1, -1]) * x_c / P.A
                           + np.array([+1, -1, +1, -1]) * y_c / P.B)
    return np.clip(n, 0.0, None)


def stability_margin(x_c, y_c):
    return 1.0 - abs(x_c) / P.A - abs(y_c) / P.B


def static_state(q_deg, payload):
    """Convenience: mass, CoM, loads, margin for a configuration."""
    m, com = total_mass_and_com(q_deg, payload)
    return {"M": m, "x_c": com[0], "y_c": com[1], "z_c": com[2],
            "N": wheel_loads(m, com[0], com[1]), "s": stability_margin(com[0], com[1])}
