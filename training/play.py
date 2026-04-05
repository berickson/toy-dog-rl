"""
Play back a trained PPO policy in the MuJoCo viewer.

Examples:
    python -m training.play
    python -m training.play --task walk --episodes 3
    python -m training.play --model policies/robot_dog_stand_ppo.zip --episodes 3
  python -m training.play --no-render --episodes 5
"""

import argparse
import os
import sys
import time

from stable_baselines3 import PPO

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.env import RobotDogEnv
else:
    from .env import RobotDogEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="stand",
        choices=["stand", "walk"],
        help="Task objective the model was trained for",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to saved SB3 model",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run headless (for quick validation)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier when rendering (1.0=realtime, 0.25=slow motion)",
    )
    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError("--speed must be > 0")

    model_path = args.model or os.path.join("policies", f"robot_dog_{args.task}_ppo.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = RobotDogEnv(
        render_mode=None if args.no_render else "human",
        max_steps=args.max_steps,
        task=args.task,
    )
    model = PPO.load(model_path)

    print(f"Task: {args.task}")
    print(f"Loaded model: {model_path}")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0
        step_num = 0

        while True:
            step_wall_start = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_num += 1

            # Print live telemetry every 10 steps (~5 Hz at 50 Hz policy).
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
                try:
                    env.render()
                except RuntimeError as exc:
                    msg = str(exc)
                    if "launch_passive" in msg and "mjpython" in msg:
                        raise SystemExit(
                            "MuJoCo viewer on macOS must run under mjpython.\n"
                            "Use: conda run -n toy-dog-rl mjpython -m training.play\n"
                            "(or activate env and run: mjpython -m training.play)"
                        ) from exc
                    raise

                # Keep playback near human-viewable speed.
                target_dt = env.ctrl_dt / args.speed
                elapsed = time.time() - step_wall_start
                sleep_dt = target_dt - elapsed
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

            if terminated or truncated:
                break

        print(f"\nEpisode {ep + 1}: reward={total_reward:.3f}  steps={step_num}")

    env.close()


if __name__ == "__main__":
    main()
