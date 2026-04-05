"""
Train a locomotion policy using PPO (Stable Baselines3).

Usage:
    python -m training.train                        # train stand task by default
    python -m training.train --timesteps 500000     # longer run
    python -m training.train --render               # watch during training (slow)
    python -m training.train --task walk            # switch to walk objective

Outputs:
    logs/  — TensorBoard event files  (tensorboard --logdir logs/)
    checkpoints/  — model saved every N steps
    policies/robot_dog_ppo.zip  — final SB3 model (also exported to ONNX)
"""

import argparse
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

if __package__ is None or __package__ == "":
    # Allow running as: python training/train.py
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.env import RobotDogEnv
else:
    # Allow running as: python -m training.train
    from .env import RobotDogEnv

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR         = os.path.join(REPO_ROOT, "logs")
CHECKPOINT_DIR  = os.path.join(REPO_ROOT, "checkpoints")
POLICY_DIR      = os.path.join(REPO_ROOT, "policies")

for d in (LOG_DIR, CHECKPOINT_DIR, POLICY_DIR):
    os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Total env steps to train (default: 200k, good for a quick Mac run)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Parallel envs (CPU cores to use)")
    parser.add_argument("--render", action="store_true",
                        help="Render one env during training (disables vectorization)")
    parser.add_argument("--task", type=str, default="stand", choices=["stand", "walk"],
                        help="Training objective")
    parser.add_argument("--check", action="store_true",
                        help="Run env checker and exit")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved model to resume training from (e.g. policies/robot_dog_stand_ppo.zip)")
    args = parser.parse_args()

    # ── Env checker ────────────────────────────────────────────────────────────
    if args.check:
        print("Running gymnasium env checker...")
        env = RobotDogEnv(task=args.task)
        check_env(env, warn=True)
        env.close()
        print("Env check passed.")
        return

    # ── Build training env ─────────────────────────────────────────────────────
    if args.render:
        # Single env with viewer — useful for debugging, but slow.
        train_env = RobotDogEnv(render_mode="human", task=args.task)
    else:
        # Vectorized envs for faster data collection on CPU.
        train_env = make_vec_env(
            RobotDogEnv,
            n_envs=args.n_envs,
            env_kwargs=dict(max_steps=1000, task=args.task),
        )

    # Separate eval env (single, no render) for EvalCallback.
    eval_env = make_vec_env(RobotDogEnv, n_envs=1, env_kwargs=dict(task=args.task))

    # ── Callbacks ──────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix=f"robot_dog_{args.task}_ppo",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(POLICY_DIR, f"best_{args.task}"),
        log_path=LOG_DIR,
        eval_freq=max(20_000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # ── PPO hyperparameters ────────────────────────────────────────────────────
    # These are sensible starting defaults for a small locomotion task.
    # Key things to tune later:
    #   - n_steps / batch_size: larger = more stable but slower updates
    #   - learning_rate:  try 1e-4 → 3e-4 range
    #   - ent_coef:       encourages exploration; reduce as training matures
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=train_env, tensorboard_log=LOG_DIR)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=LOG_DIR,
            verbose=1,
        )

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"Task: {args.task}")
    print(f"Training for {args.timesteps:,} steps across {args.n_envs} envs.")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}")
    print()

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # ── Save final policy ──────────────────────────────────────────────────────
    final_path = os.path.join(POLICY_DIR, f"robot_dog_{args.task}_ppo")
    model.save(final_path)
    print(f"\nSaved final policy → {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
