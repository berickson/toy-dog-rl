"""
MJX GPU-vectorized environment for the toy dog.

This replaces training/env.py for GPU training. Instead of one MuJoCo
simulation on CPU, MJX runs thousands of sims in parallel on the GPU
via JAX. The env logic (obs, reward, reset) is written in JAX so
everything stays on-device — no CPU↔GPU transfers per step.

The reward/observation design matches training/env.py exactly so
policies are compatible.
"""

import os
import functools

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from typing import NamedTuple


# ── Model path ─────────────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "robot_dog.xml")


# ── State container ────────────────────────────────────────────────────────────

class EnvState(NamedTuple):
    """Carries everything needed between steps (all JAX arrays, on GPU)."""
    mjx_data: mjx.Data      # MJX simulation state
    step_count: jnp.ndarray  # scalar int32
    prev_action: jnp.ndarray # (4,) previous action for smoothness penalty
    prev_x: jnp.ndarray      # scalar — previous X position for velocity calc
    commands: jnp.ndarray     # (6,) command inputs [vx, yaw_rate, height, pitch, stride_freq, stride_height]
    rng: jnp.ndarray          # PRNG key


# Number of command dimensions (fixed across all phases).
_N_COMMANDS = 6
_OBS_DIM = 14 + _N_COMMANDS  # 20


# ── Environment config ─────────────────────────────────────────────────────────

class EnvConfig(NamedTuple):
    """Static hyperparameters (not traced by JAX)."""
    task: str = "stand"
    max_steps: int = 1000
    ctrl_dt: float = 0.02       # 50 Hz policy
    # Stand reward weights (match training/env.py defaults)
    height_reward_w: float = 4.0
    tilt_penalty_w: float = 4.0
    energy_penalty_w: float = 0.01
    velocity_penalty_w: float = 0.1
    target_height: float = 0.04
    min_height: float = 0.02
    # Walk reward weights
    forward_reward_w: float = 2.0
    target_vx: float = 0.05
    vx_sigma: float = 0.01
    walk_height_penalty_w: float = 2.0
    smoothness_penalty_w: float = 0.1
    leg_limit: float = 0.35
    leg_limit_penalty_w: float = 2.0
    # Perturbation
    push_interval: int = 50
    push_force: float = 0.05


# ── Load and compile model ─────────────────────────────────────────────────────

def load_model_and_mjx():
    """Load the MuJoCo model and put it on GPU via MJX."""
    mj_model = mujoco.MjModel.from_xml_path(_MODEL_PATH)
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mjx_model


# ── Quaternion utilities (JAX) ─────────────────────────────────────────────────

def _quat_to_roll_pitch(quat):
    """Convert [qw, qx, qy, qz] to (roll, pitch) in radians. Pure JAX."""
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)
    sinp = jnp.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = jnp.arcsin(sinp)
    return roll, pitch


# ── Observation ────────────────────────────────────────────────────────────────

def _get_obs(mjx_data, mj_model, commands):
    """
    Build the 20-dim observation from MJX sensordata + commands.

    Sensor layout (from robot_dog.xml):
      [0:4]  joint positions  (pos_fl, pos_fr, pos_rl, pos_rr)
      [4:8]  joint velocities (vel_fl, vel_fr, vel_rl, vel_rr)
      [8:11] gyro             (wx, wy, wz)
      [11:14] accelerometer   (ax, ay, az)
    Commands:
      [14] vx_cmd  [15] yaw_rate_cmd  [16] height_cmd
      [17] pitch_cmd  [18] stride_freq_cmd  [19] stride_height_cmd
    """
    sensors = mjx_data.sensordata[:14]
    return jnp.concatenate([sensors, commands])


# ── Reward functions ───────────────────────────────────────────────────────────

def _compute_stand_reward(mjx_data, action, cfg):
    """Stand reward — matches training/env.py._compute_stand_reward."""
    body_z = mjx_data.qpos[2]
    height_err = body_z - cfg.target_height
    sigma = 0.0004
    height_reward = cfg.height_reward_w * jnp.exp(-(height_err ** 2) / sigma)

    quat = mjx_data.qpos[3:7]
    roll, pitch = _quat_to_roll_pitch(quat)
    tilt_penalty = cfg.tilt_penalty_w * (roll ** 2 + pitch ** 2)

    energy_penalty = cfg.energy_penalty_w * jnp.sum(action ** 2)

    lin_vel = mjx_data.qvel[0:3]
    velocity_penalty = cfg.velocity_penalty_w * jnp.sum(lin_vel ** 2)

    reward = height_reward - tilt_penalty - energy_penalty - velocity_penalty
    return reward


def _compute_walk_reward(mjx_data, action, prev_action, prev_x, commands, cfg):
    """Walk reward — Gaussian velocity tracking + height + uprightness."""
    curr_x = mjx_data.qpos[0]
    vx = (curr_x - prev_x) / cfg.ctrl_dt
    vx_cmd = commands[0]
    vx_err = vx - vx_cmd
    forward_reward = cfg.forward_reward_w * jnp.exp(-(vx_err ** 2) / (2 * cfg.vx_sigma ** 2))

    # Height reward — full reward at or above target, drops off below.
    body_z = mjx_data.qpos[2]
    height_err = jnp.minimum(body_z - cfg.target_height, 0.0)  # only penalize below target
    height_sigma = 0.0004
    height_reward = cfg.walk_height_penalty_w * jnp.exp(-(height_err ** 2) / height_sigma)

    quat = mjx_data.qpos[3:7]
    roll, pitch = _quat_to_roll_pitch(quat)
    tilt_penalty = cfg.tilt_penalty_w * (roll ** 2 + pitch ** 2)

    energy_penalty = cfg.energy_penalty_w * jnp.sum(action ** 2)

    action_delta = action - prev_action
    smoothness_penalty = cfg.smoothness_penalty_w * jnp.sum(action_delta ** 2)

    # Penalize leg angles beyond ~20° from vertical.
    joint_angles = mjx_data.qpos[7:11]  # 4 hinge angles
    excess = jnp.maximum(jnp.abs(joint_angles) - cfg.leg_limit, 0.0)
    leg_limit_penalty = cfg.leg_limit_penalty_w * jnp.sum(excess ** 2)

    reward = forward_reward + height_reward - tilt_penalty - energy_penalty - smoothness_penalty - leg_limit_penalty
    return reward


# ── Core env functions (all pure JAX, vmappable) ───────────────────────────────

def reset(mjx_model, mj_model, cfg, rng):
    """Reset a single environment. Returns (EnvState, obs)."""
    # Start from default state.
    mjx_data = mjx.make_data(mjx_model)

    rng, k1, k2, k3 = jax.random.split(rng, 4)

    # Random joint angle perturbation.
    joint_noise = jax.random.uniform(k1, shape=(mj_model.nu,), minval=-0.1, maxval=0.1)
    qpos = mjx_data.qpos.at[7:].add(joint_noise)

    # Random velocity perturbation.
    vel_noise = jax.random.uniform(k2, shape=(mj_model.nv,), minval=-0.05, maxval=0.05)
    qvel = mjx_data.qvel + vel_noise

    # Random initial tilt.
    tilt = jax.random.uniform(k3, shape=(2,), minval=-0.15, maxval=0.15)
    qpos = qpos.at[4].add(tilt[0] * 0.5)  # qx
    qpos = qpos.at[5].add(tilt[1] * 0.5)  # qy
    # Re-normalize quaternion.
    quat = qpos[3:7]
    qpos = qpos.at[3:7].set(quat / jnp.linalg.norm(quat))

    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = mjx.forward(mjx_model, mjx_data)

    # Phase 2a: fixed commands (vx=target_vx, rest=0).
    commands = jnp.zeros(_N_COMMANDS).at[0].set(cfg.target_vx)

    state = EnvState(
        mjx_data=mjx_data,
        step_count=jnp.int32(0),
        prev_action=jnp.zeros(mj_model.nu),
        prev_x=qpos[0],
        commands=commands,
        rng=rng,
    )
    obs = _get_obs(mjx_data, mj_model, commands)
    return state, obs


def step(mjx_model, mj_model, cfg, state, action):
    """
    Step a single environment. Returns (EnvState, obs, reward, done).

    Physics is advanced for ctrl_dt seconds (multiple MJX substeps).
    """
    action = jnp.clip(action, -1.0, 1.0)
    mjx_data = state.mjx_data.replace(ctrl=action)

    # Advance physics for ctrl_dt / timestep steps.
    steps_per_ctrl = max(1, round(cfg.ctrl_dt / mj_model.opt.timestep))

    def physics_step(data, _):
        return mjx.step(mjx_model, data), None

    mjx_data, _ = jax.lax.scan(physics_step, mjx_data, None, length=steps_per_ctrl)

    # Compute reward.
    if cfg.task == "stand":
        reward = _compute_stand_reward(mjx_data, action, cfg)
    else:
        reward = _compute_walk_reward(mjx_data, action, state.prev_action, state.prev_x, state.commands, cfg)

    # Termination.
    body_z = mjx_data.qpos[2]
    fallen = body_z < cfg.min_height
    step_count = state.step_count + 1
    truncated = step_count >= cfg.max_steps
    done = fallen | truncated

    new_state = EnvState(
        mjx_data=mjx_data,
        step_count=step_count,
        prev_action=action,
        prev_x=mjx_data.qpos[0],
        commands=state.commands,
        rng=state.rng,
    )

    obs = _get_obs(mjx_data, mj_model, state.commands)
    return new_state, obs, reward, done


# ── Vectorized wrappers ───────────────────────────────────────────────────────

def make_vec_env(n_envs, cfg=None, seed=0):
    """
    Create a vectorized env: n_envs parallel sims on GPU.

    Returns:
        mj_model:    CPU MuJoCo model (for metadata)
        mjx_model:   GPU MJX model
        cfg:         EnvConfig
        vec_reset:   fn(rngs) -> (states, obs)  — batched
        vec_step:    fn(states, actions) -> (states, obs, rewards, dones)  — batched
    """
    if cfg is None:
        cfg = EnvConfig()

    mj_model, mjx_model = load_model_and_mjx()

    # Partially apply static args, then vmap over (rng,) and (state, action).
    _reset = functools.partial(reset, mjx_model, mj_model, cfg)
    _step = functools.partial(step, mjx_model, mj_model, cfg)

    vec_reset = jax.vmap(_reset)
    vec_step = jax.vmap(_step)

    # JIT compile.
    vec_reset = jax.jit(vec_reset)
    vec_step = jax.jit(vec_step)

    return mj_model, mjx_model, cfg, vec_reset, vec_step
