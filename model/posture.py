"""Eq.6: posture rule. Arm faces the direction of travel; fold in if the stability margin is too low."""
import params as P
from model.com import static_state


def apply_rule(q_req, payload, travel_angle_deg):
    """q_req: requested (q1,q2,q3) deg; travel_angle_deg = atan2(vy, vx) of the current leg."""
    q = (float(travel_angle_deg), q_req[1], q_req[2])
    if static_state(q, payload)["s"] < P.S_MIN:
        fold = P.POSES[P.FOLD_POSE]
        q = (float(travel_angle_deg), fold[1], fold[2])
    return q
