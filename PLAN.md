# PLAN: slip-aware mecanum mobile manipulator simulation

Status: **awaiting approval. No project code has been written.**
(Only a throwaway back-of-envelope CoM calculation was run to sanity-check the spec; it is not in the repo.)

---

## 1. Environment

- Repo is empty. The machine has Python 3.13 by default, but `python3.11` exists, so I'll make a **3.11 venv** and pin `numpy scipy matplotlib mujoco` in `requirements.txt`. `pip` reaches PyPI (dry-run OK; mujoco 3.16, scipy 1.18 available).
- Everything headless: matplotlib `Agg`, MuJoCo without a viewer. A `--view` flag can come later if you want it.

## 2. File layout

```
robo-sim/
  params.py                 # ONLY place with numbers (robot, arm, plant truth, noise, sim, thresholds)
  requirements.txt
  model/                    # pure functions, numpy only, shared by both stages and by analysis
    arm.py                  # eq.1  FK of joints / link mid-points / servos / payload (body frame)
    com.py                  # eq.2-3 total mass, CoM (x_c,y_c), wheel loads N_i, stability margin s
    mecanum.py              # eq.4  J, wheel speeds from v, standard odometry (pinv)
    load_aware.py           # eq.5  forward "plant" (omega,N -> v) and corrected odometry (same math, estimated params)
    posture.py              # eq.6  q1 = atan2(vy,vx); fold if s < 0.3
  paths.py                  # fwd 1 m / strafe 1 m / diagonal 1 m / square 1 m -> v_cmd(t) @ 0.2 m/s, with ramps
  stage1_analytic/
    simulate.py             # run_one(pose,payload,path,mode,seed) -> log DataFrame
    run_stage1.py           # demo: trajectories plot + end-point/heading table
  stage2_mujoco/
    build_mjcf.py           # params.py -> MJCF string (chassis, 4 wheels x 10 capsule rollers @45 deg, arm, payload)
    simulate.py             # run_one(...) -> log DataFrame (same schema as stage 1 / real robot)
    sanity_test.py          # tucked arm: forward / strafe / spin
    fit_params.py           # least_squares for kappa, c, f_r on train poses, test on held-out poses
  analysis/                 # reads the CSV log schema only, so it will work on real robot logs
    logio.py                # CSV schema + read/write
    odometry.py             # standard + corrected estimators applied to a log
    metrics.py              # end-point error, heading drift, path deviation
    hypotheses.py           # H1-H4 evaluation with the pass criteria in section 6
    plots.py
  run_matrix.py             # --stage {1,2}; poses x payloads x paths x modes x repeats -> results/
  tests/test_model.py       # plain-assert self-checks (no framework), see section 5
  results/                  # *.csv, plots/*.png
  RESULTS.md
```

**Log schema** (stage 1, stage 2, real robot): `t, omega1..omega4, q1..q3, imu, gt_x, gt_y, gt_theta`.
Extra metadata (pose, payload, path, mode, repeat) goes in the results CSV row, not in the log.
Wheel order is FL, FR, RL, RR. `imu` = gyro z rate [rad/s] (see Q9).

## 3. Modelling decisions and how I read the spec

- **Frames:** body frame x forward, y left, z up, origin at chassis centre. `q2` is measured from horizontal (0 = arm straight forward, +90° = straight up). `q3` is relative to link 2. This matches the FK in eq.1.
- **Servo placement:** q1 servo at the mount, q2 servo at the shoulder (0, 0, d1), q3 servo at the elbow, payload at the gripper tip. Link brackets (0.03 kg) at link mid-length. Wheel and motor masses are part of the 1.5 kg base.
- **Wheel loads** are eq.3. When s < 0 the formula goes negative, which means a wheel lifts. I'll clip N_i at 0 and flag the run. It shouldn't occur with the given masses (see Section 7).
- **Eq.5 interpretation.** The wheel traction force is `c·N_i·(rho_i − (J v)_i)`. Force balance `Jᵀ F = W` then gives `v = (JᵀCJ)⁻¹ (JᵀC rho − W)`. `W` has the form `f_r·M·g·v̂` and zero yaw moment. Because `v̂` depends on the answer, I'll iterate 2–3 times from the commanded direction, with a smoothed `v̂ = v/√(|v|²+ε²)`.
- **Dynamics:** Stage 1 is quasi-static, with no acceleration-induced load transfer. Speed ramps are only there to avoid unrealistic steps.
- **Corrected odometry needs** q1..q3 (encoders) and the payload mass. I assume the payload mass is known per run (Q5).
- **Noise (stage 1):** Gaussian on measured omega (encoder) and a small process noise on plant velocity.
- **Square path:** holonomic, heading held at 0. Legs are forward, left, back, right, with a short stop at each corner. The posture rule re-points the arm during the stop, not while moving.
- **Metrics (defined once in `metrics.py`):**
  - *odometry error* = |est_end − gt_end| and est_heading − gt_heading
  - *path deviation* = gt_end vs the ideal commanded end-point, and *heading drift* = gt_theta_end − 0
  - H1/H2 use heading drift. H3 uses odometry error. H4 uses path deviation and heading drift (what the robot really did).

## 4. Build order (each step run and shown to you before I continue)

| # | Step | You'll see |
|---|------|-----------|
| 0 | venv + deps | version printout |
| 1 | `params.py`, `model/arm.py`, `model/com.py`, self-checks | table of x_c, y_c, s, N_i for 4 poses × 3 payloads; side/top-view figure of the 4 poses |
| 2 | `mecanum.py`, `load_aware.py` | plot: plant vs standard odometry for one case; check that the model reduces to standard odometry when loads are equal and κ = 0 |
| 3 | Stage 1 paths + simulate + estimators | `trajectories.png`, end-point / heading-drift table |
| 4 | Stage 2a: MJCF generator + `sanity_test.py` | tucked arm: forward / strafe / spin plots and numbers (commanded vs measured displacement); snapshot render if EGL/osmesa works, otherwise a schematic |
| 5 | Stage 2b: logging + estimators on logs | a few logged runs; per-wheel contact normal force vs eq.3 (validity check of the load model) |
| 6 | `fit_params.py` | fitted values, train and held-out error |
| 7 | `run_matrix.py` for both stages, `hypotheses.py`, `RESULTS.md` | CSVs, plots, H1–H4 table |

## 5. Self-checks (tests/test_model.py)

- ΣN_i = M·g and moment balance reproduce (x_c, y_c).
- Equal loads and κ = 0 and f_r = 0 → corrected odometry equals standard odometry; `pinv(J)·J = I`.
- Tucked pose → CoM near centre.
- The plant's no-noise round trip: odometry with true parameters recovers the plant velocity.

## 6. Hypothesis pass criteria (proposed now, before seeing any results)

All are paired comparisons over the 5 repeats and over the payloads 0.2 and 0.4 kg. I'll also report the 0 kg numbers.

- **H1** (arm forward + strafe): median |heading drift| ≥ 2× the tucked value **and** the difference ≥ 1° **and** the 95% CI of the difference excludes 0.
- **H2** (arm sideways + forward drive): same test.
- **H3**: corrected end-point error below standard in ≥ 75% of conditions **and** median reduction ≥ 25%. Reported separately for oracle parameters (stage 1 only) and fitted parameters on held-out poses.
- **H4**: posture-rule mode reduces median |heading drift| and path deviation by ≥ 25% vs corrected-without-rule.

These thresholds are placeholders for your sign-off. I'll change them only before running, never after.

## 7. Things in the spec I want you to know about before you approve

1. **f_r and c are not separately identifiable.** `C` scales with `c`, so the first term of eq.5 is independent of `c`. The second term depends only on `f_r/c`. The fit will therefore estimate `(κ, f_r/c)`. To get `c` and `f_r` apart I'd need to fix one from a literature value. I'll fit `(κ, f_r/c)` unless you object.
2. **Stage 1 is largely self-consistent by construction.** The plant and corrector use the same equation. The model implies that yaw coupling is proportional to the component of the CoM offset perpendicular to the travel direction (the cross product of CoM offset and v). So H1 and H2 show the model's own behaviour, and H4 (arm along the travel direction) works almost by design. H3 is only non-trivial when the corrector uses *fitted* parameters, or the plant has a mismatch the corrector doesn't have. Only Stage 2 gives independent evidence. I'll label Stage 1 results that way in RESULTS.md.
3. **Stage 2 may well show a weak or zero effect.** With rigid rollers and Coulomb contact, a consistent set of wheel speeds produces no-slip kinematics that don't depend on load. Load-dependent slip appears only through contact softness or finite motor stiffness (the velocity actuator's `kv`). So the result depends on actuator parameters we choose. I'll pick a documented `kv` and torque limit from first principles (not tuned to a hypothesis) and report whatever happens. A null result would be a genuine finding.
4. **The posture rule's fold branch probably never triggers.** My back-of-envelope with your masses (gripper servo included) gives the worst-case s ≈ 0.37 (forward- or side-extended, 0.4 kg payload) and s ≈ 0.84 / 0.58 at 0 / 0.2 kg. All are above 0.3. So in H4 only the `q1 = atan2(vy, vx)` alignment does anything. The CoM shift is still large: x_c ≈ 63 mm means front-wheel load is ≈ 1.6× nominal and rear ≈ 0.4×. Step 1 will confirm these with proper code.
5. **MuJoCo rollers:** capsules at 45° with 10 per wheel may make the wheel bob as rollers hand over. I'll check this in the sanity step. If it's bad I'll tell you before changing anything (options: two staggered rows of 10, or fewer-but-larger rollers).
6. **Stage 2 repeats:** MuJoCo is deterministic, so repeats would be identical. I propose repeats differ by seeded noise on the logged omega (and optionally a tiny initial pose jitter). Physics runs are shared across the baseline and corrected modes, since those differ only in the estimator.

## 8. Open questions (my default in bold; reply "defaults OK" or override)

1. **Plant truth + noise:** κ, c, f_r are not in the spec. Defaults chosen from physical reasoning and fixed *before* any results: **f_r = 0.03; c = 15 s/m** (traction saturates at μ·N around 0.05 m/s of slip); **κ = 0.002 1/N**; wheel-speed noise **σ = 0.02 rad/s**; process noise on velocity **σ = 0.002 m/s**. Fixed in `params.py` and not tuned.
2. **Poses (q1, q2, q3):** **tucked (0°, 110°, −165°); forward-extended (0°, 0°, 0°); side-extended (90°, 0°, 0°); raised (0°, 60°, 0°).** A fully vertical arm has zero CoM shift, so I made "raised" a partly raised reach. Step 1 prints the values for your approval.
3. **Square path:** **holonomic (no body rotation)**, as in Section 3. Alternative: rotate 90° at each corner.
4. **"Fold the arm in":** **go to the tucked pose.** Alternative: shorten reach only until s ≥ 0.3.
5. **Payload known to the corrected estimator?** **Yes**, it's a run parameter.
6. **Gripper servo:** **include one extra 0.055 kg servo at the wrist / gripper tip** (the "servo at each joint" list covers q1..q3 only). Say no if the gripper is lighter or already counted.
7. **Wheel actuators in MuJoCo:** no motor spec given. **Velocity actuator with `kv` and torque limit chosen to match a small geared DC motor, documented in params.py; 0.5 s acceleration ramp.**
8. **Sim timestep:** **0.002 s** (about 3–10k steps per run; 480 physics runs is the estimate for a full matrix).
9. **`imu` column:** **gyro z rate [rad/s]**, with noise in sim. It is logged but not used by either estimator.
10. **Normal-force check against eq.3** in step 5 is a small validity check on the load model, not a new feature. **Include it**, drop it if you'd rather not.
11. **Thresholds in Section 6:** OK, or give your own?
