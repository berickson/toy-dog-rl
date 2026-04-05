# Training Plan

## Phase 1: Stand
**Status: complete**

Future enhancement: make height, roll, pitch commandable (same pattern as walk commands).

Goal: hold torso at a target height, level and still.

- Observation: 14 sensor values (4 joint pos, 4 joint vel, 3 gyro, 3 accel)
- Action: 4 motor torques in [-1, 1]
- Commands: none
- Reward: height tracking + uprightness - energy - velocity
- Termination: body falls below min height, or max steps

What you learn: reward shaping, training loop, debugging sim behavior.

## Phase 2: Walk (command-conditioned, incremental)
**Status: in progress — Phase 2a**

Goal: follow commanded velocity, yaw rate, and posture/gait parameters.

### Observation: 20 dims (fixed network shape throughout)

| Index  | Signal             | When enabled         |
|--------|--------------------|----------------------|
| 0–3    | joint positions    | always               |
| 4–7    | joint velocities   | always               |
| 8–10   | gyro               | always               |
| 11–13  | accelerometer      | always               |
| 14     | `vx_cmd`           | Phase 2a             |
| 15     | `yaw_rate_cmd`     | Phase 2b             |
| 16     | `height_cmd`       | Phase 2c             |
| 17     | `pitch_cmd`        | Phase 2c             |
| 18     | `stride_freq_cmd`  | Phase 2d             |
| 19     | `stride_height_cmd`| Phase 2d             |

- Action: 4 motor torques in [-1, 1]
- Unused command inputs are fixed at 0 until their phase begins
- Network shape never changes — no retraining from scratch between phases

### Phase 2a: Fixed forward speed ← current
- `vx_cmd` = 0.03 m/s (fixed, not randomized yet)
- All other commands = 0
- Reward: vx Gaussian tracking + one-sided height + uprightness - tilt - energy - smoothness - leg limit
- Once gait is stable, start randomizing `vx_cmd` in [-0.05, 0.05] range

### Phase 2b: Add yaw rate
- Resume from 2a weights
- Start rewarding `yaw_rate_cmd` tracking (Gaussian)
- `yaw_rate_cmd`: randomized [-0.5, 0.5] rad/s per episode

### Phase 2c: Add posture commands
- Resume from 2b weights
- `height_cmd`: target body ride height (crouch vs tall)
- `pitch_cmd`: forward/back lean

### Phase 2d: Add gait parameters
- Resume from 2c weights
- `stride_freq_cmd`: step frequency (slow deliberate vs fast trot)
- `stride_height_cmd`: foot lift height (high step vs shuffle)
- Note: limited by 1-DOF legs — may not be fully controllable

Each sub-phase resumes from the previous weights and only enables
the new command input + reward term. The fixed 20-dim observation
means the network shape is stable across all phases.

What you learn: command conditioning, incremental training, gait emergence.

## Phase 3: Tricks (separate policies)

Goal: one policy per trick behavior.

| Trick     | Key reward signal                        |
|-----------|------------------------------------------|
| Sit       | target body height near ground + upright |
| Roll over | body completes 360° roll                 |
| Dance     | rhythmic leg movement + staying upright  |

Each trick is a standalone env/reward with the same 14-dim observation.
No commands needed — each is a single fixed behavior.

What you learn: diverse reward design, episode structure for non-locomotion tasks.

## Phase 4: Remote control (firmware, not RL)

Goal: real-time control from a remote.

The remote sends `[vx, yaw_rate, trick_id]` over BLE/radio to the ESP32.
- `trick_id == 0`: run the walk policy with `[vx, yaw_rate]` as commands
- `trick_id > 0`: switch to the corresponding trick policy

No new training needed — this is firmware integration.

## Sim-to-real transfer

Strategies to bridge simulation → real hardware:
- Domain randomization: vary mass, friction, motor strength during training
- Observation noise: add noise to sensor readings matching real sensor quality
- Action delay: simulate the ~20ms latency of ESP32 inference
- Bang-bang discretization: quantize continuous actions to match real motor driver
- System ID: measure real robot parameters and update the MJCF model

## Training infrastructure

| Stage               | Machine        | Framework          |
|---------------------|---------------|--------------------|
| Env design & debug  | MacBook Air    | SB3 + CPU MuJoCo   |
| Short runs (<1M)    | MacBook Air    | SB3 + CPU MuJoCo   |
| Long runs (>1M)     | Linux / 4090   | MJX + JAX (GPU)    |
| Policy export       | either         | ONNX               |
| Firmware inference   | ESP32-S3       | ONNX Runtime / TFLite Micro |
