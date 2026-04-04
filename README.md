# toy-dog-rl

Sim-to-real reinforcement learning for a RapidPower Mini robot dog (JXD-8002-TV1).

Train a locomotion policy in MuJoCo, deploy on ESP32-S3 + MPU6050.

## Structure

- `models/` — MJCF robot model for MuJoCo
- `training/` — RL training config and scripts (MuJoCo Playground / MJX)
- `policies/` — Exported trained policies (ONNX)
- `firmware/` — ESP32 deployment code

## Hardware

- 4 legs, 1 DOF each (fore-aft swing, ~20° arc)
- Bang-bang DC motors (upgrade to PWM via DRV8833 planned)
- Potentiometers for joint position feedback
- MPU6050 IMU for body orientation
- ESP32-S3 running policy inference at ~50Hz

## Machines

- **MacBook Air**: MJCF authoring, visualization, policy testing
- **Linux desktop (RTX 4090)**: RL training via MJX/JAX
