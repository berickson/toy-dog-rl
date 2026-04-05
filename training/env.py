"""
RobotDogEnv — Gymnasium environment for the toy dog MuJoCo model.

Observation (20 values):
  [0:4]   joint positions   FL, FR, RL, RR  (radians)
  [4:8]   joint velocities  FL, FR, RL, RR  (rad/s)
  [8:11]  gyroscope         wx, wy, wz       (rad/s)
  [11:14] accelerometer     ax, ay, az       (m/s²)
  [14]    vx_cmd            forward speed command
  [15]    yaw_rate_cmd      yaw rate command
  [16]    height_cmd        target height command
  [17]    pitch_cmd         body pitch command
  [18]    stride_freq_cmd   stride frequency command
  [19]    stride_height_cmd stride height command

Action (4 values):
  Motor torque commands in [-1, 1], one per leg (FL, FR, RL, RR).

Reward:
    Stand task:
        + stay near target body height
        + stay upright (small roll/pitch)
        - large actions
        - body motion while balancing

    Walk task:
        + forward velocity
        + maintain standing height
        - roll/pitch tilt penalty
        - large actions
        - jerky action changes (smoothness)

Episode ends when:
  - body height drops below a threshold (fallen over), or
  - time limit reached (max_steps)
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer


# Path to the model, relative to the repo root.
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "robot_dog.xml")

# Observation size and bounds.
_SENSOR_DIM = 14
_N_COMMANDS = 6
_OBS_DIM = _SENSOR_DIM + _N_COMMANDS  # 20
_SENSOR_HIGH = (
    [2 * np.pi] * 4    # joint pos (unbounded in practice, generous limit)
    + [50.0] * 4       # joint vel  (rad/s)
    + [20.0] * 3       # gyro       (rad/s)
    + [30.0] * 3       # accel      (m/s² — includes gravity ~9.81)
)
# Command bounds: [vx, yaw_rate, height, pitch, stride_freq, stride_height]
_COMMAND_HIGH = [1.0, 3.0, 0.1, 1.0, 5.0, 0.05]
_OBS_HIGH = _SENSOR_HIGH + _COMMAND_HIGH


class RobotDogEnv(gym.Env):
    """Gymnasium environment for sim-to-real locomotion training."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        max_steps=1000,
        task="stand",
        ctrl_dt=0.02,          # policy runs at 50 Hz (every 0.02s of sim time)
        forward_reward_w=2.0,  # weight on forward velocity reward
        tilt_penalty_w=4.0,    # weight on tilt penalty
        energy_penalty_w=0.01, # weight on energy penalty
        height_reward_w=4.0,   # stand task: reward for staying near target height
        velocity_penalty_w=0.1,# stand task: discourage drifting while balancing
        target_height=0.04,    # stand task: torso target height (meters)
        min_height=0.02,       # body Z below this → episode ends (fallen)
        push_interval=50,      # apply a random push every N steps (0=disabled)
        push_force=0.05,       # max force of random pushes (Newtons)
        # Walk task
        target_vx=0.05,                   # walk task: target forward velocity (m/s)
        vx_sigma=0.01,                    # walk task: Gaussian tracking width
        walk_height_penalty_w=2.0,         # walk task: stay near target height
        smoothness_penalty_w=0.1,          # penalize action changes between steps
        leg_limit=0.35,                    # walk task: max leg angle from vertical (rad, ~20°)
        leg_limit_penalty_w=2.0,           # walk task: penalty weight for exceeding leg limit
    ):
        super().__init__()

        self.render_mode = render_mode
        self.task = task
        self.max_steps = max_steps
        self.ctrl_dt = ctrl_dt
        self.forward_reward_w = forward_reward_w
        self.tilt_penalty_w = tilt_penalty_w
        self.energy_penalty_w = energy_penalty_w
        self.height_reward_w = height_reward_w
        self.velocity_penalty_w = velocity_penalty_w
        self.target_height = target_height
        self.min_height = min_height
        self.push_interval = push_interval
        self.push_force = push_force
        self.target_vx = target_vx
        self.vx_sigma = vx_sigma
        self.walk_height_penalty_w = walk_height_penalty_w
        self.smoothness_penalty_w = smoothness_penalty_w
        self.leg_limit = leg_limit
        self.leg_limit_penalty_w = leg_limit_penalty_w

        self._prev_action = np.zeros(4)

        if self.task not in ("stand", "walk"):
            raise ValueError("task must be one of: stand, walk")

        # Load model once; recreate data each episode.
        self.model = mujoco.MjModel.from_xml_path(_MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        # How many physics steps per policy step.
        self._steps_per_ctrl = max(1, round(ctrl_dt / self.model.opt.timestep))

        # Sensor address/dim cache (built once).
        self._sensor_info = self._build_sensor_cache()

        # Gymnasium spaces.
        obs_high = np.array(_OBS_HIGH, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Viewer (only created when render_mode="human").
        self._viewer = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Small random perturbation to joint angles and body orientation so
        # the policy learns to recover from diverse starting states.
        self._rng = np.random.default_rng(seed)

        self.data.qpos[7:] += self._rng.uniform(-0.1, 0.1, size=self.model.nu)  # joint angles
        self.data.qvel[:] += self._rng.uniform(-0.05, 0.05, size=self.model.nv)

        # Random initial tilt — simulate imperfect ground.
        # Perturb the quaternion [qw, qx, qy, qz] with small roll/pitch.
        init_roll  = self._rng.uniform(-0.15, 0.15)  # ~8.5° max
        init_pitch = self._rng.uniform(-0.15, 0.15)
        self.data.qpos[4] += init_roll  * 0.5   # qx
        self.data.qpos[5] += init_pitch * 0.5   # qy
        # Re-normalize quaternion.
        quat = self.data.qpos[3:7]
        self.data.qpos[3:7] = quat / np.linalg.norm(quat)

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_x = self.data.qpos[0]  # track forward progress
        self._prev_action = np.zeros(self.model.nu)

        # Phase 2a: fixed commands (vx=target_vx, rest=0).
        self._commands = np.zeros(_N_COMMANDS, dtype=np.float32)
        self._commands[0] = self.target_vx

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        self.data.ctrl[:] = action

        # Random push perturbation — simulates uneven ground, wind, bumps.
        if self.push_interval > 0 and self._step_count % self.push_interval == 0:
            fx = self._rng.uniform(-self.push_force, self.push_force)
            fy = self._rng.uniform(-self.push_force, self.push_force)
            # xfrc_applied[body_id] = [fx, fy, fz, tx, ty, tz]
            torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            self.data.xfrc_applied[torso_id, :3] = [fx, fy, 0]
        else:
            # Clear external forces so pushes are momentary.
            self.data.xfrc_applied[:] = 0

        # Advance physics for ctrl_dt seconds.
        for _ in range(self._steps_per_ctrl):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self._step_count >= self.max_steps

        info = reward_info
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                # Launch passive viewer (we drive the loop, not mujoco).
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        """Build the 20-dim observation vector from sensordata + commands."""
        s = self._sensor_info
        return np.concatenate([
            self.data.sensordata[s["pos_fl"]:s["pos_fl"] + 1],
            self.data.sensordata[s["pos_fr"]:s["pos_fr"] + 1],
            self.data.sensordata[s["pos_rl"]:s["pos_rl"] + 1],
            self.data.sensordata[s["pos_rr"]:s["pos_rr"] + 1],
            self.data.sensordata[s["vel_fl"]:s["vel_fl"] + 1],
            self.data.sensordata[s["vel_fr"]:s["vel_fr"] + 1],
            self.data.sensordata[s["vel_rl"]:s["vel_rl"] + 1],
            self.data.sensordata[s["vel_rr"]:s["vel_rr"] + 1],
            self.data.sensordata[s["body_gyro"]:s["body_gyro"] + 3],
            self.data.sensordata[s["body_acc"]:s["body_acc"] + 3],
            self._commands,
        ], dtype=np.float32)

    def _compute_reward(self, action):
        """Compute reward for the selected task (stand or walk)."""
        if self.task == "stand":
            return self._compute_stand_reward(action)
        return self._compute_walk_reward(action)

    def _compute_walk_reward(self, action):
        """
        Simple walk reward — go forward, stay upright, be smooth:
          + forward_reward_w      * forward_velocity
          - walk_height_penalty_w * (z - target_height)^2
          - tilt_penalty_w        * (roll^2 + pitch^2)
          - energy_penalty_w      * sum(action^2)
          - smoothness_penalty_w  * sum((action - prev_action)^2)
        """
        # Forward velocity — Gaussian tracking reward centered on target_vx.
        curr_x = self.data.qpos[0]
        dx = curr_x - self._prev_x
        self._prev_x = curr_x
        vx = dx / self.ctrl_dt
        vx_err = vx - self.target_vx
        forward_reward = self.forward_reward_w * float(np.exp(-(vx_err ** 2) / (2 * self.vx_sigma ** 2)))

        # Height reward — full reward at or above target, drops off below.
        body_z = float(self.data.qpos[2])
        height_err = min(body_z - self.target_height, 0.0)  # only penalize below target
        height_sigma = 0.0004
        height_reward = self.walk_height_penalty_w * float(np.exp(-(height_err ** 2) / height_sigma))

        # Tilt penalty from torso orientation.
        quat = self.data.qpos[3:7]  # [qw, qx, qy, qz]
        roll, pitch = _quat_to_roll_pitch(quat)
        tilt_penalty = self.tilt_penalty_w * (roll ** 2 + pitch ** 2)

        # Energy penalty.
        energy_penalty = self.energy_penalty_w * float(np.sum(action ** 2))

        # Smoothness penalty — discourage jerky lunges, encourage cyclic gaits.
        action_delta = action - self._prev_action
        smoothness_penalty = self.smoothness_penalty_w * float(np.sum(action_delta ** 2))
        self._prev_action = action.copy()

        # Penalize leg angles beyond ~20° from vertical.
        joint_angles = self.data.qpos[7:11]
        excess = np.maximum(np.abs(joint_angles) - self.leg_limit, 0.0)
        leg_limit_penalty = self.leg_limit_penalty_w * float(np.sum(excess ** 2))

        reward = float(forward_reward + height_reward - tilt_penalty
                       - energy_penalty - smoothness_penalty - leg_limit_penalty)

        info = {
            "task": "walk",
            "vx": vx,
            "body_z": body_z,
            "height_reward": height_reward,
            "roll": roll,
            "pitch": pitch,
            "tilt_penalty": tilt_penalty,
            "energy_penalty": energy_penalty,
            "smoothness_penalty": smoothness_penalty,
        }
        return reward, info

    def _compute_stand_reward(self, action):
        """
        Stand reward shaping:
          + height_reward_w    * exp(-((z-target)^2)/sigma)
          - tilt_penalty_w     * (roll^2 + pitch^2)
          - energy_penalty_w   * sum(action^2)
          - velocity_penalty_w * (vx^2 + vy^2 + vz^2)
        """
        # Height tracking reward around target standing height.
        body_z = float(self.data.qpos[2])
        height_err = body_z - self.target_height
        sigma = 0.0004  # ~2 cm tolerance band
        height_reward = self.height_reward_w * float(np.exp(-(height_err ** 2) / sigma))

        # Uprightness penalty from torso orientation.
        quat = self.data.qpos[3:7]  # [qw, qx, qy, qz]
        roll, pitch = _quat_to_roll_pitch(quat)
        tilt_penalty = self.tilt_penalty_w * (roll ** 2 + pitch ** 2)

        # Penalize motor effort.
        energy_penalty = self.energy_penalty_w * float(np.sum(action ** 2))

        # Penalize translational movement to encourage stable standing.
        lin_vel = self.data.qvel[0:3]
        velocity_penalty = self.velocity_penalty_w * float(np.sum(lin_vel ** 2))

        reward = float(height_reward - tilt_penalty - energy_penalty - velocity_penalty)

        info = {
            "task": "stand",
            "body_z": body_z,
            "height_err": height_err,
            "height_reward": height_reward,
            "roll": roll,
            "pitch": pitch,
            "tilt_penalty": tilt_penalty,
            "energy_penalty": energy_penalty,
            "velocity_penalty": velocity_penalty,
        }
        return reward, info

    def _is_terminated(self):
        """End the episode if the body has fallen (height too low)."""
        body_z = self.data.qpos[2]  # Z position of torso
        return bool(body_z < self.min_height)

    def _build_sensor_cache(self):
        """Map sensor names to their start index in sensordata."""
        cache = {}
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            adr = self.model.sensor_adr[i]
            cache[name] = adr
        return cache


# ------------------------------------------------------------------
# Quaternion utility
# ------------------------------------------------------------------

def _quat_to_roll_pitch(quat):
    """
    Convert MuJoCo quaternion [qw, qx, qy, qz] to roll and pitch (radians).
    Roll  = rotation around X (side tilt)
    Pitch = rotation around Y (fore-aft tilt)
    """
    qw, qx, qy, qz = quat
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))
    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = float(np.arcsin(sinp))
    return roll, pitch


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Smoke-testing RobotDogEnv...")
    env = RobotDogEnv()

    obs, info = env.reset(seed=42)
    print(f"  obs shape: {obs.shape}  (expected ({_SENSOR_DIM},))")
    print(f"  obs range: [{obs.min():.3f}, {obs.max():.3f}]")

    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print(f"  100 random steps done, total reward: {total_reward:.3f}")
    print("OK")
