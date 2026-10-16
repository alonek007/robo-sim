"""Single source of truth for every number in the project (SI units, angles in degrees unless named *_rad)."""
import numpy as np

# ---- gravity / floor ------------------------------------------------------
G = 9.81
MU = 0.8                       # floor friction (sweep 0.5-1.0 later)

# ---- chassis / wheels -----------------------------------------------------
A = 0.10                       # wheel x offset  (FL,FR: +A ; RL,RR: -A)
B = 0.10                       # wheel y offset  (FL,RL: +B ; FR,RR: -B)
L = A + B
WHEEL_R = 0.0325               # r0
ROLLER_ANGLE_DEG = 45.0
N_ROLLERS = 10                 # per wheel
WHEEL_NAMES = ("FL", "FR", "RL", "RR")
WHEEL_XY = np.array([[+A, +B], [+A, -B], [-A, +B], [-A, -B]])   # body frame, order FL FR RL RR
BASE_MASS = 1.5
BASE_COM_Z = 0.10

# ---- arm ------------------------------------------------------------------
D1 = 0.08                      # shoulder height above mount
L2 = 0.12
L3 = 0.15
SERVO_MASS = 0.055             # MG996R
BRACKET_MASS = 0.03            # per link, at mid-length
GRIPPER_SERVO_MASS = 0.055     # extra servo at the wrist/gripper tip (plan Q6)
PAYLOADS = (0.0, 0.2, 0.4)     # kg at gripper tip

# poses: (q1, q2, q3) deg.  q2 measured from horizontal, q3 relative to link 2.
POSES = {
    "tucked":   (0.0, 110.0, -165.0),
    "forward":  (0.0, 0.0, 0.0),
    "side":     (90.0, 0.0, 0.0),
    "raised":   (0.0, 60.0, 0.0),
}
FOLD_POSE = "tucked"           # posture rule: pose used to "fold the arm in"
S_MIN = 0.3                    # posture rule stability threshold

# ---- load-aware plant truth (plan Q1; fixed a priori, NOT tuned) ----------
KAPPA_TRUE = 0.002             # 1/N   rho_i = r0 (1 - kappa N_i) omega_i
C_TRUE = 15.0                  # s/m   traction stiffness per unit load
FR_TRUE = 0.03                 # rolling-resistance coefficient
V_EPS = 1e-3                   # m/s   smoothing of the travel direction in W

# ---- noise (stage 1) ------------------------------------------------------
OMEGA_NOISE = 0.02             # rad/s  encoder noise on measured omega
VEL_NOISE = 0.002              # m/s    process noise on plant velocity
GYRO_NOISE = 0.005             # rad/s

# ---- paths ----------------------------------------------------------------
SPEED = 0.2                    # m/s
LEG = 1.0                      # m
RAMP_T = 0.5                   # s   acceleration ramp
CORNER_PAUSE = 0.5             # s   stop at square corners
DT = 0.01                      # s   logging / stage-1 integration step
N_REPEATS = 5
