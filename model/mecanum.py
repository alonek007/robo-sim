"""Eq.4: mecanum kinematics, standard odometry and dead-reckoning."""
import numpy as np
import params as P

# rows FL, FR, RL, RR ; columns vx, vy, wz   (rim speed rho = J @ v)
J = np.array([[1, -1, -P.L],
              [1,  1,  P.L],
              [1,  1, -P.L],
              [1, -1,  P.L]], dtype=float)
J_PINV = np.linalg.pinv(J)


def wheel_speeds(v):
    """Inverse kinematics: body velocity [vx, vy, wz] -> wheel angular speeds [rad/s]."""
    return (J @ np.asarray(v, dtype=float)) / P.WHEEL_R


def standard_odometry(omega):
    """Standard odometry: wheel angular speeds -> body velocity [vx, vy, wz]."""
    return P.WHEEL_R * (J_PINV @ np.asarray(omega, dtype=float))


def integrate_pose(v_body, dt, pose0=(0.0, 0.0, 0.0)):
    """Dead-reckon a pose history from body-frame velocities (N,3) sampled every dt.

    Returns (N+1, 3) array of [x, y, theta]; row k is the pose before sample k.
    Uses the mid-point heading for the translation step.
    """
    v = np.atleast_2d(v_body)
    pose = np.zeros((len(v) + 1, 3))
    pose[0] = pose0
    for k, (vx, vy, wz) in enumerate(v):
        th = pose[k, 2] + 0.5 * wz * dt
        c, s = np.cos(th), np.sin(th)
        pose[k + 1] = pose[k] + dt * np.array([c * vx - s * vy, s * vx + c * vy, wz])
    return pose
