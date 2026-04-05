"""
Diagnostic: verify model symmetry, actuator mapping, sensor layout,
and reward signals. Run this to debug asymmetric leg behavior.

Usage: python training/diagnose.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "robot_dog.xml")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("=" * 60)
print("MODEL DIAGNOSTICS")
print("=" * 60)

# --- Joint layout ---
print("\n--- Joints ---")
print(f"{'idx':<4} {'name':<12} {'type':<8} {'qposadr':<8} {'dofadr':<8} {'axis':<20} {'range'}")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jnt_type = ["free","ball","slide","hinge"][model.jnt_type[i]]
    qpos_adr = model.jnt_qposadr[i]
    dof_adr = model.jnt_dofadr[i]
    axis = model.jnt_axis[i]
    limited = model.jnt_limited[i]
    jnt_range = model.jnt_range[i] if limited else "unlimited"
    print(f"{i:<4} {name:<12} {jnt_type:<8} {qpos_adr:<8} {dof_adr:<8} {str(axis):<20} {jnt_range}")

# --- Actuators ---
print("\n--- Actuators ---")
print(f"{'idx':<4} {'name':<12} {'joint':<12} {'gear':<10} {'ctrlrange':<20} {'type'}")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    # trnid[i][0] is the joint index for joint-type actuators
    jnt_id = model.actuator_trnid[i][0]
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
    gear = model.actuator_gear[i]
    ctrlrange = model.actuator_ctrlrange[i]
    dyntype = model.actuator_dyntype[i]
    gaintype = model.actuator_gaintype[i]
    biastype = model.actuator_biastype[i]
    print(f"{i:<4} {name:<12} {jnt_name:<12} {gear[0]:<10.4f} {str(ctrlrange):<20} dyn={dyntype} gain={gaintype} bias={biastype}")

# --- Sensors ---
print("\n--- Sensors ---")
print(f"{'idx':<4} {'name':<12} {'adr':<5} {'dim':<4} {'type'}")
for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    adr = model.sensor_adr[i]
    dim = model.sensor_dim[i]
    stype = model.sensor_type[i]
    print(f"{i:<4} {name:<12} {adr:<5} {dim:<4} type={stype}")

# --- Body positions (leg attachment points) ---
print("\n--- Body positions (in parent frame) ---")
for bname in ["torso", "leg_fl", "leg_fr", "leg_rl", "leg_rr"]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    pos = model.body_pos[bid]
    print(f"  {bname:<10} pos={pos}")

# --- Per-leg symmetry test ---
print("\n--- Per-leg response test ---")
print("Applying ctrl=+1 to each leg individually, checking torque direction:\n")

legs = ["joint_fl", "joint_fr", "joint_rl", "joint_rr"]
motors = ["motor_fl", "motor_fr", "motor_rl", "motor_rr"]

for mi, mname in enumerate(motors):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Get initial joint position
    jid = model.actuator_trnid[mi][0]
    jname = legs[mi]
    qpos_adr = model.jnt_qposadr[jid]

    init_qpos = data.qpos[qpos_adr]

    # Apply +1 control to this motor only
    data.ctrl[:] = 0
    data.ctrl[mi] = 1.0

    # Step a few times
    for _ in range(50):
        mujoco.mj_step(model, data)

    final_qpos = data.qpos[qpos_adr]
    delta = final_qpos - init_qpos

    print(f"  {mname:<12} → {jname:<12}: "
          f"init={init_qpos:+.4f}  final={final_qpos:+.4f}  "
          f"delta={delta:+.4f} rad ({np.degrees(delta):+.1f}°)  "
          f"{'FORWARD' if delta > 0 else 'BACKWARD'}")

# --- Check: does ctrl=+1 on all legs push feet downward? ---
print("\n--- All legs ctrl=+1 simultaneously ---")
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

init_z = data.qpos[2]
data.ctrl[:] = 1.0

for _ in range(100):
    mujoco.mj_step(model, data)

final_z = data.qpos[2]
print(f"  Torso Z: {init_z:.4f} → {final_z:.4f}  (delta={final_z-init_z:+.4f})")

# All legs ctrl=-1
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
data.ctrl[:] = -1.0
for _ in range(100):
    mujoco.mj_step(model, data)
final_z_neg = data.qpos[2]
print(f"  Torso Z (ctrl=-1): {init_z:.4f} → {final_z_neg:.4f}  (delta={final_z_neg-init_z:+.4f})")

# --- Reward sanity at reset ---
print("\n--- Reward at reset (stand task) ---")
from training.env import RobotDogEnv

env = RobotDogEnv(task="stand")
obs, info = env.reset(seed=0)
# Zero action
action = np.zeros(4)
obs, reward, terminated, truncated, info = env.step(action)
print(f"  Zero-action step: reward={reward:.4f}")
for k, v in info.items():
    if isinstance(v, float):
        print(f"    {k}: {v:.6f}")
    else:
        print(f"    {k}: {v}")

# Check body_z vs target
print(f"\n  target_height={env.target_height}")
print(f"  body_z after reset+1step = {info.get('body_z', 'N/A')}")
print(f"  height_err = {info.get('height_err', 'N/A')}")

env.close()
print("\n" + "=" * 60)
print("DONE")
