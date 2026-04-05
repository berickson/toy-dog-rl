"""
Play back an MJX-trained policy in the MuJoCo viewer.

Loads JAX params from .npz, runs inference with the Flax network,
but steps the *CPU* MuJoCo sim for rendering (MJX has no viewer).

Examples:
    python -m training_mjx.play
    python -m training_mjx.play --task walk --episodes 3
    python -m training_mjx.play --model policies/robot_dog_walk_mjx.npz
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_mjx.ppo import ActorCritic
from training_mjx.train import load_params

# Reuse the CPU env for rendering + stepping.
from training.env import RobotDogEnv


def main():
    parser = argparse.ArgumentParser(description="Play MJX-trained policy")
    parser.add_argument("--task", type=str, default="stand", choices=["stand", "walk"])
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .npz params file")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (1.0=realtime, 0.25=slow-mo)")
    args = parser.parse_args()

    model_path = args.model or os.path.join("policies", f"robot_dog_{args.task}_mjx.npz")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load the Flax network and params.
    act_dim = 4
    network = ActorCritic(action_dim=act_dim)
    params = load_params(model_path)

    # JIT the forward pass (single obs, not batched).
    @jax.jit
    def get_action(params, obs):
        mean, log_std, value = network.apply(params, obs)
        return mean  # deterministic: use the mean

    # CPU env for rendering.
    env = RobotDogEnv(
        render_mode=None if args.no_render else "human",
        max_steps=args.max_steps,
        task=args.task,
    )

    print(f"Task: {args.task}")
    print(f"Loaded MJX model: {model_path}")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0
        step_num = 0

        while True:
            step_wall_start = time.time()

            # JAX inference: numpy obs → jax → action → numpy.
            obs_jax = jnp.array(obs)
            action_jax = get_action(params, obs_jax)
            action = np.array(action_jax)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_num += 1

            # Print telemetry every 10 steps.
            if step_num % 10 == 0:
                if info.get("task") == "stand":
                    print(
                        f"\r  step={step_num:4d}  "
                        f"z={info['body_z']:.4f} (target={env.target_height:.4f})  "
                        f"roll={info['roll']:+.3f}  pitch={info['pitch']:+.3f}  "
                        f"r={reward:+.3f}  R={total_reward:+.1f}",
                        end="", flush=True,
                    )
                elif info.get("task") == "walk":
                    print(
                        f"\r  step={step_num:4d}  "
                        f"vx={info['vx']:+.4f}  "
                        f"z={info['body_z']:.4f}  "
                        f"smooth={info['smoothness_penalty']:.3f}  "
                        f"r={reward:+.3f}  R={total_reward:+.1f}",
                        end="", flush=True,
                    )

            if not args.no_render:
                env.render()
                target_dt = env.ctrl_dt / args.speed
                elapsed = time.time() - step_wall_start
                sleep_dt = target_dt - elapsed
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

            if terminated or truncated:
                break

        print(f"\nEpisode {ep + 1}: reward={total_reward:.3f}  steps={step_num}")

    env.close()

    if not args.no_render:
        os._exit(0)


if __name__ == "__main__":
    main()
