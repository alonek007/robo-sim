"""Eq.1: arm forward kinematics in the body frame (origin = chassis centre, arm mounted there)."""
import numpy as np
import params as P


def _rot_z(q1):
    c, s = np.cos(q1), np.sin(q1)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def arm_points(q_deg):
    """Return dict of body-frame positions [m] for joints, link mid-points and tip.

    Arm plane coordinates are (rho, z); q1 rotates the plane about body z.
    """
    q1, q2, q3 = np.radians(q_deg)
    plane = lambda rho, z: _rot_z(q1) @ np.array([rho, 0.0, z])
    rho_e = P.L2 * np.cos(q2)
    z_e = P.D1 + P.L2 * np.sin(q2)
    rho_g = rho_e + P.L3 * np.cos(q2 + q3)
    z_g = z_e + P.L3 * np.sin(q2 + q3)
    return {
        "mount": np.zeros(3),
        "shoulder": plane(0.0, P.D1),
        "elbow": plane(rho_e, z_e),
        "tip": plane(rho_g, z_g),
        "link2_mid": plane(rho_e / 2, (P.D1 + z_e) / 2),
        "link3_mid": plane((rho_e + rho_g) / 2, (z_e + z_g) / 2),
    }
