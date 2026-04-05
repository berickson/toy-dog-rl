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

## Phase 2: Walk (command-conditioned)
**Status: in progress**

Goal: follow a commanded velocity and yaw rate.

- Observation: 14 sensors + 2 commands = 16 dims
- Commands: `[vx_cmd, yaw_rate_cmd]` (randomized each episode)
  - `vx_cmd`: -0.05 to 0.05 m/s (forward/backward)
  - `yaw_rate_cmd`: -0.5 to 0.5 rad/s (turning)
  - Note: `vy_cmd` omitted — robot has Y-axis hinges only, can't strafe
- Reward: vx tracking (gaussian) + yaw tracking (gaussian) + height maintenance - tilt - energy
- Termination: body falls, or max steps

This single policy handles forward, backward, turning, and stopping
(command = zero). Turning falls out of yaw_rate tracking for free.

What you learn: command conditioning, gait emergence, domain randomization.

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
