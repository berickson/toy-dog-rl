"""
Manual controller for the toy dog MuJoCo model.

This script demonstrates the MuJoCo Python API:
- Loading a model and creating a simulation
- Stepping physics forward
- Reading sensor data
- Applying motor commands
- Rendering with the viewer

Run: python test_controller.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time


# --- Load model and create simulation data ---
# 
# MuJoCo separates the MODEL (static: geometry, masses, joint definitions)
# from the DATA (dynamic: positions, velocities, forces — the simulation state).
#
# model = the robot definition (from XML)
# data  = the current state of the simulation (changes every step)

model = mujoco.MjModel.from_xml_path("models/robot_dog.xml")
data = mujoco.MjData(model)


# --- Understand the data layout ---
#
# MuJoCo packs everything into flat arrays. Here's what we care about:
#
# data.qpos  = joint positions (for our robot: 7 for freejoint + 4 leg angles)
#              freejoint = [x, y, z, qw, qx, qy, qz] (position + quaternion)
#              then 4 hinge angles in radians
#
# data.qvel  = joint velocities (6 for freejoint + 4 leg angular velocities)
#              freejoint = [vx, vy, vz, wx, wy, wz] (linear + angular vel)
#
# data.ctrl  = motor control inputs — this is what we write to
#              4 values in [-1, 1], one per motor
#
# data.sensordata = all sensor readings in order:
#              [4 joint pos, 4 joint vel, 3 gyro, 3 accel, 4 quaternion]

print("=== Model Info ===")
print(f"  Joints:    {model.njnt} (1 freejoint + 4 hinges)")
print(f"  Actuators: {model.nu} (4 motors)")
print(f"  Sensors:   {model.nsensor}")
print(f"  qpos size: {model.nq} (7 freejoint + 4 hinge angles)")
print(f"  qvel size: {model.nv} (6 freejoint + 4 hinge velocities)")
print(f"  Timestep:  {model.opt.timestep}s ({1/model.opt.timestep:.0f} Hz physics)")
print()


# --- Sensor helper ---
#
# Sensors are packed sequentially into data.sensordata.
# We can look up each sensor's offset and size by name.

def read_sensors(model, data):
    """Read all sensors into a friendly dict."""
    sensors = {}
    for i in range(model.nsensor):
        name = model.sensor(i).name
        adr = model.sensor(i).adr[0]   # start index in sensordata
        dim = model.sensor(i).dim[0]   # how many values
        sensors[name] = data.sensordata[adr:adr+dim].copy()
    return sensors


# --- Simple gait pattern ---
#
# Before RL learns a policy, let's manually define a simple alternating
# gait to verify the actuators work. This is the dumbest possible walk:
# drive diagonal leg pairs in opposite directions, then switch.
#
# FL+RR move together (diagonal pair 1)
# FR+RL move together (diagonal pair 2)
# This is a trot gait pattern.

STEP_DURATION = 3.3  # seconds per gait phase
TARGET_ANGLE = 0.25  # radians (~15°) — how far each leg swings from center


def trot_targets(sim_time):
    """
    Returns [FL, FR, RL, RR] target angles (radians) for a trot gait.
    Diagonal pairs swing in opposite directions, then switch.
    """
    phase = (sim_time % (2 * STEP_DURATION)) < STEP_DURATION
    if phase:
        return [TARGET_ANGLE, -TARGET_ANGLE, -TARGET_ANGLE, TARGET_ANGLE]
    else:
        return [-TARGET_ANGLE, TARGET_ANGLE, TARGET_ANGLE, -TARGET_ANGLE]


# --- Run the simulation with the viewer ---
#
# mujoco.viewer.launch_passive gives us a viewer window that we control.
# We step the physics ourselves in a loop, which lets us insert our
# controller logic between steps.

PRINT_INTERVAL = 1.0  # print sensor data every N seconds
last_print_time = 0

print("=== Starting simulation ===")
print("Watch the viewer — the dog should attempt a trot gait.")
print("Close the viewer window to exit.")
print()

with mujoco.viewer.launch_passive(model, data) as viewer:
    wall_start = time.time()
    sim_start = data.time

    while viewer.is_running():
        sim_time = data.time

        # --- Controller: set target angles ---
        # With position actuators, ctrl = target angle in radians.
        # MuJoCo's built-in PD controller drives the joint there.
        targets = trot_targets(sim_time)
        data.ctrl[:] = targets

        # --- Step physics ---
        # Each call advances by model.opt.timestep (0.002s)
        mujoco.mj_step(model, data)

        # --- Read and print sensors periodically ---
        if sim_time - last_print_time >= PRINT_INTERVAL:
            sensors = read_sensors(model, data)
            
            print(f"t={sim_time:6.2f}s | "
                  f"legs=[{sensors['pos_fl'][0]:+5.1f} {sensors['pos_fr'][0]:+5.1f} "
                  f"{sensors['pos_rl'][0]:+5.1f} {sensors['pos_rr'][0]:+5.1f}]° | "
                  f"gyro=[{sensors['body_gyro'][0]:+5.2f} {sensors['body_gyro'][1]:+5.2f} "
                  f"{sensors['body_gyro'][2]:+5.2f}] | "
                  f"ctrl={[f'{c:+.1f}' for c in targets]}")
            
            last_print_time = sim_time

        # --- Sync viewer at realtime ---
        # Sleep until wall-clock time catches up to simulation time
        elapsed_wall = time.time() - wall_start
        elapsed_sim = data.time - sim_start
        dt = elapsed_sim - elapsed_wall
        if dt > 0:
            time.sleep(dt)

        viewer.sync()
