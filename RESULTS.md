# RESULTS: slip-aware mecanum mobile manipulator

Pass/fail uses the criteria fixed in `PLAN.md` section 6 / `params.H_CRIT` **before any result was seen**; none were changed.
Matrix: 4 poses x 3 payloads x 4 paths x 3 modes x 5 repeats = 720 runs per stage (`results/matrix_*.csv`, one row per run).
Verdicts and numbers come from `results/hypotheses_*.json`; figures are in `results/plots/` (`drift_*`, `h3_*`, `h4_*`).

## Verdicts

| | Stage 1 (analytic, oracle parameters) | Stage 2 (MuJoCo, soft motors, **main**) | Stage 2 (MuJoCo, stiff motors, sensitivity) |
|---|---|---|---|
| **H1** arm forward + strafe -> heading drift | **FAIL** (0.4 kg only) | **PASS** | **FAIL** (0.4 kg only) |
| **H2** arm sideways + forward -> heading drift | **FAIL** (0.4 kg only) | **PASS** | **FAIL** (0.4 kg only) |
| **H3** corrected odometry beats standard | **PASS** (oracle params) | **FAIL** (held-out poses) | **FAIL** (held-out poses) |
| **H4** posture rule reduces drift | **FAIL** | **FAIL** (primary criterion) | **FAIL** |

Required for H1/H2: extended-arm |heading drift| >= 2x tucked, median difference >= 1 deg, 95% CI of the paired difference excludes 0, at **both** 0.2 and 0.4 kg.
Required for H3: corrected end-point error below standard in >= 75% of conditions and median error reduced >= 25%.
Required for H4: median |heading drift| **and** median path deviation both reduced >= 25% (all 48 conditions).

## Numbers

**H1 / H2: median |heading drift| after 1 m (extended arm vs tucked arm, same path)**

| | payload | Stage 1 | Stage 2 soft | Stage 2 stiff |
|---|---|---|---|---|
| H1 forward arm, strafe | 0 kg | 0.16 vs 0.13 deg | 0.46 vs 0.00 deg | 0.04 vs 0.05 deg |
| | 0.2 kg | 0.38 vs 0.14 deg (diff 0.35, CI [-0.24, 0.56]) **criteria not met** | 3.75 vs 0.01 deg (met) | 0.44 vs 0.01 deg (**not met**) |
| | 0.4 kg | 1.28 vs 0.13 deg (diff 1.25, CI [0.65, 1.47]) met | 10.60 vs 0.03 deg (met) | 1.54 vs 0.02 deg (met) |
| H2 side arm, forward | 0 kg | 0.16 vs 0.13 deg | 0.19 vs 0.00 deg | 0.03 vs 0.00 deg |
| | 0.2 kg | 0.48 vs 0.13 deg (diff 0.43, CI [0.24, 0.52]) **not met** (< 1 deg) | 2.09 vs 0.00 deg (met) | 0.16 vs 0.00 deg (**not met**) |
| | 0.4 kg | 1.38 vs 0.13 deg (diff 1.33, CI [1.14, 1.42]) met | 5.74 vs 0.00 deg (met) | 1.39 vs 0.00 deg (met) |

- The effect exists and grows with payload and CoM offset in every setting. Stage 1 and stiff-motor Stage 2 are just too small at 0.2 kg to clear the 1 deg bar.
- In Stage 2 the physics is deterministic, so repeats differ only by encoder noise. Ground-truth drift has **zero variance across repeats** and the CIs above are zero-width. They show the difference is not noise, not that it is statistically estimated.

**H3: corrected vs standard odometry, end-point error (median over conditions)**

| | conditions | improved | standard | corrected | reduction | \|heading err\| std -> corr |
|---|---|---|---|---|---|---|
| Stage 1, oracle params | 48 | 87.5% | 20.9 mm | 2.5 mm | 88% | 0.18 -> 0.17 deg |
| Stage 2 soft, held-out poses (side, raised) | 24 | 58% | 37.6 mm | 18.7 mm | 50% | 0.49 -> 0.52 deg |
| Stage 2 soft, training poses (tucked, forward) | 24 | 67% | 23.4 mm | 17.0 mm | 28% | 0.29 -> 0.27 deg |
| Stage 2 stiff, held-out poses | 24 | 54% | 25.2 mm | 37.9 mm | **-50% (worse)** | 0.61 -> 2.16 deg |

Fitted parameters (soft motors): kappa = -0.0022 (+-0.0001), f_r/c = 0.0049 (+-0.0001); Stage 1 truth was +0.002 and 0.002. Stiff: kappa = +0.0008, f_r/c = 0.0042.

**H4: posture rule (corrected odometry, rule vs no rule; per-condition mean, median over conditions)**

| | |drift| no rule -> rule | path deviation no rule -> rule | passes? |
|---|---|---|---|
| Stage 1, all 48 | 0.164 -> 0.163 deg (-0.6%) | 20.9 -> 20.4 mm (-2%) | no |
| Stage 2 soft, all 48 | 0.72 -> 0.05 deg (**-93%**) | 47.4 -> 42.5 mm (**-10%**) | no (path deviation) |
| Stage 2 stiff, all 48 | 0.32 -> 0.15 deg (-55%) | 30.9 -> 27.4 mm (-11%) | no |

Secondary view, **chosen after seeing the Stage 1 results, so exploratory not a verdict**: only the 16 conditions with the arm extended (forward/side pose, payload >= 0.2 kg), where there is something to reduce:
Stage 1 drift -40%, path deviation -6%. Stage 2 soft drift **-97%**, path deviation -35% (would pass). Stage 2 stiff drift -91%, path deviation -12%.

## What the numbers mean

1. **Drift from the arm's CoM shift is real in the physics simulation.** With soft motors, strafing with the arm forward and 0.4 kg turns the robot by 10.6 deg in 1 m; tucked, the same run turns 0.03 deg. Standard odometry misses most of it (it reported about -3 deg for the +10.6 deg turn).
2. **The size depends strongly on wheel motor stiffness.** Making the wheel motors 10x stiffer cuts the 10.6 deg case to 1.5 deg. If your real wheels are speed-controlled with encoder feedback, expect an effect closer to the stiff column. This is the biggest open question for the hardware.
3. **Stage 1 shows the model's own behaviour, not independent evidence.** The plant and the corrector use the same equation, so H3 passing there mostly reflects removing a 2-3% speed loss (kappa, f_r), not the heading effect, and H4 follows from the model (no yaw coupling when the CoM is on the travel line). The `0.2 kg` failures for H1/H2 in Stage 1 are pure effect-size: the default plant parameters give about 0.4 deg.
4. **The load-aware correction, as specified, does not transfer to MuJoCo.** The two-parameter, quasi-static model halves the median end-point error on held-out poses but fails the 75%-of-conditions test, does not reduce heading error, and fits a negative kappa. The MuJoCo drift develops during acceleration and is not captured by a quasi-static slip model. With stiff motors the fit is worse than standard odometry.
5. **The posture rule removes the drift on single-leg paths but not on the square.** In Stage 2 soft it drops the extended-arm single-leg drift to about 0. On the square, 0.4 kg forward/side poses still drift 5.3 deg with the rule (4.1 / 7.0 deg without), and tucked/square gets slightly worse (0.2 -> 0.6 deg). The final heading depends on corner pause length (forward/0.4 kg/square with rule: -5.2 deg at 0.5 s, -2.1 deg at 2.0 s, one diagnostic run each, not investigated further). The rule's path deviation barely improves because path deviation is dominated by the roller-geometry speed shortfall (3-5 cm over 1 m), not by CoM-induced drift. The fold-in branch triggered in only 2 of 48 conditions (forward or side pose, 0.4 kg, diagonal path: pointing the arm along 45 deg gives s = 0.11 < 0.3 because the margin is a diamond norm, larger on the diagonals); there the arm folds to tucked and drift falls to -0.05 deg. In every other condition only the q1 alignment acted. (My earlier note that it would never trigger used the unrotated poses and was wrong for diagonal travel.)

## Caveats and things not tuned

- **MuJoCo is not the real robot.** Key sim choices (all documented in `params.py`): soft velocity actuator `kv` = 0.05 N m s/rad (stiff run: 0.5); 10 straight capsule rollers per wheel (roller length chosen so neighbouring rollers just meet; longer rollers touch two at a time and cut strafe gain to 0.89); idealised 3 N m arm servos.
- **Roller-geometry bias:** with the arm tucked the MuJoCo robot goes 0.983 of the commanded distance forward, 0.959 strafing, and 87.6 deg of a commanded 90 deg spin. This direction-dependent gain is independent of load and shows up as baseline end-point error.
- **Hardware finding:** forward- or side-extended with 0.4 kg needs 1.35 N m at the shoulder (0.82 N m at 0.2 kg). An MG996R (~1 N m) cannot hold the 0.4 kg poses. See `results/static_loads.csv`.
- c and f_r are not separately identifiable (only f_r/c is); c was held at 15 s/m.
- Stage 2 repeats are noise-only (see above); Stage 2 friction was fixed at mu = 0.8 (the 0.5-1.0 sweep was left for later; a quick check at the start showed drift is not monotonic in mu: 3.3 deg at 0.5, 10.6 at 0.8, 8.0 at 1.5 for the forward-arm strafe).
- The MuJoCo sign of the effect on the diagonal path differs from the quasi-static model (forward arm: strafe +10.6 deg, diagonal -8.2 deg); not explained.
- A fit uses 2 parameters and one noise seed. H3 for Stage 2 is evaluated on held-out poses only, as pre-registered.

## Reproduce

```
python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python tests/test_model.py
python run_matrix.py --stage 1
python stage2_mujoco/fit_params.py --motor soft      # and --motor stiff
python run_matrix.py --stage 2 --motor soft          # and --motor stiff
python analysis/plots.py results/matrix_stage2_soft.csv
```

The `analysis/` code (`logio.py`, `odometry.py`, `metrics.py`, `hypotheses.py`) reads only the log schema
`t, omega1..4, q1..3, imu, gt_x, gt_y, gt_theta` (`analysis/logio.py` documents the row convention),
so it can be pointed at real robot CSVs.
