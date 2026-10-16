"""Eq.5: load-aware wheel/ground model.

    rho_i = r0 (1 - kappa N_i) omega_i          effective rim speed
    C     = diag(c N_i)                         traction stiffness (force per unit slip speed)
    v     = (J^T C J)^-1 (J^T C rho - W)        force balance J^T F = W, F = C (rho - J v)
    W     = f_r M g * vhat  (zero yaw moment)   resistance opposing the travel direction

The same function is the stage-1 *plant* (true parameters) and the *corrected odometry*
(estimated parameters); only the parameters passed in differ.
"""
from dataclasses import dataclass
import numpy as np
import params as P
from model.mecanum import J


@dataclass(frozen=True)
class SlipParams:
    kappa: float = P.KAPPA_TRUE
    c: float = P.C_TRUE
    f_r: float = P.FR_TRUE


TRUE = SlipParams()
NONE = SlipParams(kappa=0.0, c=P.C_TRUE, f_r=0.0)   # no slip effects at all


def body_velocity(omega, N, mass, p=TRUE, v_init=None, n_iter=4):
    """Body velocity [vx, vy, wz] for wheel speeds omega [rad/s], loads N [N], total mass [kg].

    W depends on the travel direction of the answer, so iterate (starting from v_init or
    the resistance-free solution); the smoothed direction avoids a singularity at v = 0.
    """
    omega, N = np.asarray(omega, float), np.asarray(N, float)
    rho = P.WHEEL_R * (1.0 - p.kappa * N) * omega
    C = np.diag(p.c * N)
    A = J.T @ C @ J
    b = J.T @ C @ rho
    v = np.linalg.solve(A, b) if v_init is None else np.asarray(v_init, float)
    for _ in range(n_iter):
        vt = v[:2]
        vhat = vt / np.sqrt(vt @ vt + P.V_EPS ** 2)
        W = np.array([p.f_r * mass * P.G * vhat[0], p.f_r * mass * P.G * vhat[1], 0.0])
        v = np.linalg.solve(A, b - W)
    return v


def body_velocity_batch(omega, N, mass, p=TRUE, n_iter=4):
    """Vectorised body_velocity for many samples sharing the same loads N and mass. omega: (n,4) -> (n,3)."""
    omega, N = np.atleast_2d(omega), np.asarray(N, float)
    rho = P.WHEEL_R * (1.0 - p.kappa * N) * omega                       # (n,4)
    C = np.diag(p.c * N)
    A = J.T @ C @ J
    b = rho @ (J.T @ C).T                                               # (n,3)
    Ainv = np.linalg.inv(A)
    v = b @ Ainv.T
    for _ in range(n_iter):
        vt = v[:, :2]
        vhat = vt / np.sqrt((vt ** 2).sum(1, keepdims=True) + P.V_EPS ** 2)
        W = np.zeros_like(v)
        W[:, :2] = p.f_r * mass * P.G * vhat
        v = (b - W) @ Ainv.T
    return v
