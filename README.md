# toy-dog-rl

Sim-to-real reinforcement learning for a RapidPower Mini robot dog (JXD-8002-TV1).

Train a locomotion policy in MuJoCo, deploy on ESP32-S3 + MPU6050.

This is mainly a learning exercise for me. I am collaborating with AI to build RL into the robot and understand RL concepts.

## Goals
Train the dog to
locomotion:
- walk with standard walking gaits
- walk with "rolling arm" walking gates
- turn
- respond to obstacles and failures

tricks:
- roll over
- sit
- dance

Remote control
- be controlled with a remote control to both roam around and do tricks

## Getting Started
### Install and set environment
```
conda env create -f environment.yml
conda activate toy-dog-rl

###  Run the simulation
```bash
python -m mujoco.viewer --mjcf models/robot_dog.xml
```
### Run the controller
```bash
python test_controller.py
```

## Train
cpu
```bash
python -m training.train --task stand --timesteps 200000 --n-envs 16
```
<br/>

gpu
```bash
python -m training_mjx.train --task stand --timesteps 200000 --n-envs 16
```


### Run Tensorboard
```bash
tensorboard --logdir logs/
```

### Run model in simulator
cpu
```
python -m training.play --episodes 10 --max-steps 1000 --task walk
```
<br/>
gpu
```
python -m training_mjx.play --episodes 10 --max-steps 1000 --task walk
```

## Structure

- `models/` — MJCF robot model for MuJoCo
- `training/` — RL training config and scripts (MuJoCo Playground / MJX)
- `policies/` — Exported trained policies (ONNX)
- `firmware/` — ESP32 deployment code

## Hardware

- RapidPower Mini Remote Control Pet
- 4 legs, 1 DOF each (360° controllable)
- Bang-bang DC motors (upgrade to PWM via DRV8833 planned)
- Potentiometers for joint position feedback (with dead zone)
- MPU6050 IMU for body orientation
- ESP32-S3 running policy inference at ~50Hz

## Machines

- **MacBook Air**: MJCF authoring, visualization, policy testing
- **Linux desktop (RTX 4090)**: RL training via MJX/JAX
