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
# leg directions (deg, in the body/start frame; heading is held at 0 - holonomic square, plan Q3)
PATH_LEG_ANGLES_DEG = {
    "forward":  (0.0,),
    "strafe":   (90.0,),
    "diagonal": (45.0,),
    "square":   (0.0, 90.0, 180.0, 270.0),
}
DT = 0.01                      # s   logging / stage-1 integration step
N_REPEATS = 5

# ---- experiment modes: (odometry estimator, posture rule active in the physical run) ----
MODES = {
    "baseline":            ("standard",  False),
    "corrected":           ("corrected", False),
    "corrected+posture":   ("corrected", True),
}

# ---- stage 2: MuJoCo geometry / actuators / numerics (documented choices, not tuned to hypotheses) ----
CHASSIS_HALF = (0.12, 0.07, 0.02)   # chassis box half-extents (x, y, z)
CHASSIS_CLEARANCE = 0.03            # bottom of the chassis box above the floor
BASE_BOX_DIMS = (0.24, 0.16, 0.10)  # box used only for the base inertia tensor
WHEEL_HUB_MASS = 0.045              # per wheel, hub only
ROLLER_MASS = 0.0015                # per roller  (wheel total = 0.045 + 10 * 0.0015 = 0.06 kg, part of BASE_MASS)
WHEEL_HUB_RADIUS = 0.020
WHEEL_HUB_HALF_WIDTH = 0.012
ROLLER_RADIUS = 0.006               # capsule cap radius
ROLLER_END_FACTOR = 1.0             # capsule cylinder half-length = factor * Rc sin(pi/N)/cos(roller angle): neighbouring
                                    # rollers' ends just meet. Longer rollers (>~1.05) touch the floor two at a time and
                                    # drop strafe/spin gain to ~0.89/0.93 (sanity-test sweep, step 4)
ROLLER_DAMPING = 1e-5               # N m s/rad, small so rollers "free-spin" but stay numerically calm
WHEEL_ARMATURE = 5e-4               # kg m^2, reflected motor inertia
# wheel motor models (velocity actuator tau = kv (w_cmd - w), clipped at the torque limit):
#   soft  = small geared open-loop DC motor (stall 0.6 N m, no-load 12 rad/s)   <- default, used for the main results
#   stiff = same wheels with a tight encoder speed loop (10x kv, 5x torque)      <- sensitivity run (added at user request)
WHEEL_MOTORS = {"soft": (0.05, 0.6), "stiff": (0.5, 3.0)}    # (kv [N m s/rad], torque limit [N m])
DEFAULT_MOTOR = "soft"
SERVO_BOX_HALF = (0.02, 0.01, 0.02) # servo body, mass SERVO_MASS
MG996R_STALL_TORQUE = 1.0           # N m (~10-11 kg cm): reference only, used in the pose-feasibility report
ARM_KP = 100.0                      # N m/rad  position servo gain
ARM_KV = 3.0                        # N m s/rad
ARM_TORQUE_LIMIT = 3.0              # N m  IDEALISED: an MG996R (~1 N m) cannot hold the forward/side-extended 0.4 kg poses
                                    # (needs ~1.3 N m, see results/step5_load_check.csv); the simulation assumes the arm holds
                                    # its pose so that the study isolates the base-slip effect
ARM_DAMPING = 0.2
ARM_ARMATURE = 1e-3
PAYLOAD_RADIUS = 0.02
MJ_TIMESTEP = 0.002                 # s ; log step P.DT must be a multiple of this
MJ_SOLREF = (0.005, 1.0)            # contact softness
MJ_SOLIMP = (0.95, 0.99, 0.001)
MJ_IMPRATIO = 10.0
SETTLE_T = 1.0                      # s of settling before every logged run

# ---- slip-parameter fit (stage 2): split fixed BEFORE looking at any fit result ----
FIT_TRAIN_POSES = ("tucked", "forward")
FIT_TEST_POSES = ("side", "raised")        # held out: never used in the fit
FIT_C_FIXED = C_TRUE                       # c and f_r are not separately identifiable: fit (kappa, f_r/c), c held at this value
FIT_SAMPLE_DT = 0.5                        # s, spacing of trajectory samples in the residual

# ---- hypothesis pass criteria (fixed in PLAN.md section 6 before any result was seen) ----
H_CRIT = dict(
    h12_ratio=2.0,        # median |heading drift| of the extended-arm case >= this x the tucked case ...
    h12_min_deg=1.0,      # ... and the median difference >= this many degrees, and the 95% CI of the paired difference excludes 0
    h12_payloads=(0.2, 0.4),
    h3_frac=0.75,         # corrected end-point error below standard in >= this fraction of conditions ...
    h3_reduction=0.25,    # ... and the median per-condition end-point error reduced by >= this fraction
    h4_reduction=0.25,    # posture rule reduces median |heading drift| and median path deviation by >= this fraction
)
