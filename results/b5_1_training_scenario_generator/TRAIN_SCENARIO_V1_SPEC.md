# TRAIN SCENARIO V1 Specification

Status: FROZEN BEFORE PRODUCTION IMPLEMENTATION  
Stage: PHASE B5.1 LOCAL TRAINING SCENARIO GENERATOR COMPLETION  
Entry baseline: B5_CLOSED  
Entry source fingerprint: `74a3567fdfc400d9960bfa995d1e085ea48b8c40b7e9c5f3fb98ac2e7ddfd9fb`  
Source protocol: `handoff/protocol/LOCAL_v2_0.docx`  
Source protocol SHA256: `ad368c0952d0d265a98ba0a6af80e3a9f88a906654ba256c5308d4443c67482a`

## 1 Scope and source hierarchy

This specification freezes only a local training-scenario generator, its immutable data model,
configuration, factory, and reproducibility metadata. It does not alter `LocalNavigationEnv`, the
fixed B5 Safe-SAC implementation, reward, cost, sensing, KF, risk, validator, or termination.

The direct DOCX extraction is preserved in `local_protocol_targeted_extract.json` and
`local_protocol_full_text.txt`. The controlling source locations are Appendix B Parameters,
Generalization G1-G10, Complete Algorithm step 1, and Stage 1-3. Where LOCAL specifies support but
does not specify an internal probability law, the rule is explicitly labelled
IMPLEMENTATION-FREEZE-V1 below.

## 2 LOCAL-FROZEN rules

### 2.1 Environment and task

- N/E/D operation bounds are `[0,100]`, `[0,100]`, and `[2,40]` metres.
- Main training current is zero. The generator contains no current model.
- Horizon is 1000 control steps; success radius is 2 m.
- Every environment reset performs the existing one-second legal detection history.
- New scenarios are represented by new immutable scenario objects and new environment instances;
  fixed-scenario `LocalNavigationEnv.reset()` semantics remain unchanged.

### 2.2 Start and goal support

- Start N/E/D: `[15,25]`, `[20,80]`, `[8,32]` m.
- Goal N/E/D: `[75,85]`, `[20,80]`, `[8,32]` m.
- At least half of indexed scenarios have `abs(D_goal-D_start) >= 4 m`.

### 2.3 Dynamic-obstacle train support

- Count: 1-4; speed magnitude: 0.2-0.8 m/s; physical radius: 0.5-1.0 m.
- Initial Body-relative distance: 8-23 m; horizontal bearing: -50 to +50 degrees;
  elevation: -25 to +25 degrees.
- Truth motion is strict CV. CA, CT, random manoeuvre, 5-6 obstacles, speeds 1.0-1.4 m/s,
  and unknown current are OOD and excluded from TRAIN_SCENARIO_V1.
- Obstacle positions are not clipped or rejected because their centres lie outside the AUV
  operation box. Obstacles do not bounce at that box.
- No pairwise obstacle non-overlap rejection is applied. Initial overlap count is diagnostic only.

### 2.4 Coordinate and execution contracts

- NED is North/East/Down; Body is forward/right/down. Positive pitch points the Body forward axis
  toward negative Down.
- Body-relative geometry is converted with the existing Body-to-NED rotation. Round-trip tests use
  the existing NED-to-Body implementation and sensor bearing/elevation implementation.
- Complete Algorithm step 1 requires the training namespace, a scenario identifier, initialized
  state, and the shared legal history. No policy outcome enters generation.

## 3 IMPLEMENTATION-FREEZE-V1 probability laws

LOCAL fixes the supports above but not these within-support laws.

### 3.1 Start and goal

Each start coordinate and each ordinary goal coordinate is independently uniform on its stated
support. For even `scenario_index`, goal D is repeatedly drawn from `Uniform(8,32)` until
`abs(D_goal-D_start) >= 4`; this is the conditional uniform distribution required for the forced
vertical-separation stratum. Odd indices use ordinary `Uniform(8,32)`. The rule guarantees at
least 50% of every consecutive index set beginning at zero is forced, and exactly every even index
is forced; no training result may change this rule.

### 3.2 Initial AUV state

The initial position is the sampled start. The goal-line-of-sight orientation is

`yaw = atan2(E_goal-E_start, N_goal-N_start)`

`pitch = -atan2(D_goal-D_start, hypot(N_goal-N_start, E_goal-E_start))`.

This is independently checked against the first column of the existing Body-to-NED rotation.
Initial surge is the project configuration minimum (0.3 m/s); yaw and pitch rates are zero.

### 3.3 Obstacle count and properties

- Count is one draw from `DiscreteUniform{1,2,3,4}` with probabilities `[0.25]*4`. Property
  generation never redraws count.
- Per obstacle, distance, bearing, elevation, radius, and speed are mutually generated from
  Uniform(8,23), Uniform(-50,50 deg), Uniform(-25,25 deg), Uniform(0.5,1.0 m), and
  Uniform(0.2,0.8 m/s).
- A Body direction is `[cos(e)cos(b), cos(e)sin(b), -sin(e)]`, exactly matching the existing
  sensor/ray convention. The current AUV rotation maps the Body displacement to NED.
- Velocity direction is isotropic on S2: draw `z ~ N(0,I3)`, redraw only the probability-zero
  zero-norm case, normalize, and multiply by the sampled speed. It is not aimed toward the AUV,
  its route, risk, TTC, or policy.
- Motion metadata is `CV`; no field is added to `GroundTruthObstacleState`.

### 3.4 Rejection and retry policy

Only rejected goal-D draws in the forced stratum and a numerical zero vector in S2 direction
sampling are allowed retries. No scenario-level rejection, TTC/risk/validator/policy screening,
success screening, collision-course requirement, boundary clipping, or overlap rejection exists.
Metadata records the two allowed retry counts and total generation retry count.

## 4 Data, identity, and RNG contracts

`TrainingScenario`, `TrainingScenarioConfig`, obstacle metadata, and scenario metadata are frozen
value objects. Arrays stored in a scenario are copied and made read-only. The scenario contains:
stable ID, distribution version, root seed, scenario index, initial AUV state, obstacle truth tuple,
goal NED position, and generation metadata.

Scenario identity is `train-v1-seed-<root_seed>-idx-<scenario_index>`. It uses no clock, UUID, or
Python randomized hash. A local deterministic RNG is derived from `(root_seed, scenario_index,
"scenario")`; generation cannot consume sensor-noise, dropout, environment, Actor, Replay, or cost
streams. Generation order does not affect a scenario.

## 5 Configuration contract

The single file `configs/train_scenario_v1.yaml` holds all distribution settings. Loading is strict:
ranges must be ordered; probabilities must be finite, nonnegative, and sum to one; count values are
exactly 1-4; speed cannot exceed 0.8; angle supports cannot exceed +/-50 and +/-25; motion must be
CV; current must be zero; velocity direction must be isotropic S2; AUV orientation/surge/rates must
be the frozen values. Count 5, speed 1.2, CT, or unknown current fails immediately.

## 6 API and orchestration boundary

`LocalTrainingScenarioGenerator.generate(root_seed, scenario_index)` returns one immutable
scenario. `make_local_navigation_env(project_config, scenario, task_config=None)` constructs one
environment and performs no randomization. This stage does not change a runner, sequence episodes,
store a future training scenario index in checkpoints, run RL training, or resume the blocked
throughput benchmark.

## 7 Pre-registered verification

- Determinism in-process and independent-process; stable identity; seed/index differentiation.
- Support, forced vertical stratum, LOS orientation, initial rates and surge.
- Count single-draw contract; relative geometry round trip; radius, speed, finite values, strict CV.
- Existing visibility and ray convention cross-checks; 1000 random geometry round trips.
- Isolation of sensor-noise, dropout, and environment namespaces.
- 10,000-scenario no-OOD and count diagnostic; 10,000 S2 direction diagnostic.
- Static no-policy/no-risk/no-validator/no-reward/no-cost import and call audit.
- Factory exact preservation; fixed-environment regression; 100 legal warm-ups.
- Ten-scenario B5 Actor readiness smoke, API and finiteness only, with no timing claim.
- Original 34, B1 35, Phase A 13, B2 75, B3 81, B4 73, B5 65, new scenario tests, full
  repository, Ruff 0.6.0, and compileall.

No Monte Carlo diagnostic is treated as a proof of a probability law or a navigation result.
