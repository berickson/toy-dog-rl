"""
GPU-accelerated training via MJX + JAX PPO.

Usage:
    python -m training_mjx.train                           # stand task, 1M steps
    python -m training_mjx.train --task stand --timesteps 5000000
    python -m training_mjx.train --n-envs 4096             # more parallel sims

This runs the physics AND the RL update entirely on the GPU.
Policies are saved as .npz files (JAX arrays) and can be exported to ONNX
for use with the existing SB3 play.py viewer or firmware deployment.
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np

from .env import make_vec_env, EnvConfig, EnvState
from .ppo import ActorCritic, Transition, compute_gae, make_ppo_update


# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO_ROOT, "logs_mjx")
POLICY_DIR = os.path.join(REPO_ROOT, "policies")

for d in (LOG_DIR, POLICY_DIR):
    os.makedirs(d, exist_ok=True)


def sample_action(rng, mean, log_std):
    """Sample from Gaussian policy."""
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, shape=mean.shape)
    action = mean + std * noise
    log_prob = -0.5 * jnp.sum(((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    return action, log_prob


def train(args):
    print(f"JAX devices: {jax.devices()}")

    cfg = EnvConfig(task=args.task, max_steps=args.max_steps)
    mj_model, mjx_model, cfg, vec_reset, vec_step = make_vec_env(
        n_envs=args.n_envs, cfg=cfg, seed=args.seed,
    )

    from training_mjx.env import _OBS_DIM
    obs_dim = _OBS_DIM  # 20 (14 sensors + 6 commands)
    act_dim = mj_model.nu
    n_envs = args.n_envs
    n_steps = args.n_steps  # rollout length before PPO update

    # ── Network + optimizer ────────────────────────────────────────────────────
    network = ActorCritic(action_dim=act_dim)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((obs_dim,))
    params = network.init(init_rng, dummy_obs)

    if args.resume:
        if args.resume == "auto":
            resume_path = os.path.join("policies", f"robot_dog_{args.task}_mjx.npz")
        else:
            resume_path = args.resume
        print(f"Resuming from: {resume_path}")
        params = load_params(resume_path)

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(args.lr),
    )
    opt_state = optimizer.init(params)

    ppo_update = make_ppo_update(
        network, optimizer,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        clip_range=args.clip_range,
        vf_coef=0.5,
        ent_coef=args.ent_coef,
    )

    # Vectorized network inference.
    batch_apply = jax.vmap(network.apply, in_axes=(None, 0))

    # ── Initial reset ──────────────────────────────────────────────────────────
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, n_envs)
    states, obs = vec_reset(reset_rngs)

    # ── Training loop ──────────────────────────────────────────────────────────
    total_steps = 0
    n_updates = args.timesteps // (n_envs * n_steps)
    print(f"Task: {args.task}")
    print(f"Envs: {n_envs}, rollout: {n_steps} steps, updates: {n_updates}")
    print(f"Total timesteps: {n_updates * n_envs * n_steps:,}")
    print(f"Logs: {LOG_DIR}")
    print()

    # For TensorBoard-compatible logging.
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(LOG_DIR)
    except ImportError:
        writer = None
        print("(tensorboard not available for logging, install tensorboard to enable)")

    wall_start = time.time()
    ep_returns = []  # track completed episode returns

    for update_i in range(n_updates):
        # ── Collect rollout ────────────────────────────────────────────────────
        transitions = []
        rollout_rewards = jnp.zeros(n_envs)

        for t in range(n_steps):
            rng, act_rng = jax.random.split(rng)

            # Forward pass: get action distribution + value.
            mean, log_std, value = batch_apply(params, obs)
            action, log_prob = jax.vmap(sample_action)(
                jax.random.split(act_rng, n_envs), mean,
                jnp.broadcast_to(log_std, mean.shape),
            )

            # Step all envs.
            next_states, next_obs, reward, done = vec_step(states, action)

            transitions.append(Transition(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value,
            ))

            rollout_rewards = rollout_rewards + reward

            # Auto-reset done envs.
            rng, reset_rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, n_envs)
            fresh_states, fresh_obs = vec_reset(reset_rngs)

            # Where done, log episode return and use fresh state.
            done_mask = done
            if jnp.any(done_mask):
                done_returns = jnp.where(done_mask, rollout_rewards, jnp.nan)
                for r in done_returns:
                    if not jnp.isnan(r):
                        ep_returns.append(float(r))
                rollout_rewards = jnp.where(done_mask, 0.0, rollout_rewards)

            # Select fresh or continued state per env.
            states = jax.tree.map(
                lambda fresh, old: jnp.where(
                    done_mask.reshape(-1, *([1] * (fresh.ndim - 1))),
                    fresh, old
                ),
                fresh_states, next_states,
            )
            obs = jnp.where(done_mask[:, None], fresh_obs, next_obs)

        # Stack transitions: (n_steps, n_envs, ...).
        rollout = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)

        # Bootstrap value for last obs.
        _, _, last_value = batch_apply(params, obs)

        # ── PPO update ────────────────────────────────────────────────────────
        rng, update_rng = jax.random.split(rng)
        params, opt_state, epoch_infos = ppo_update(
            params, opt_state, rollout, last_value, update_rng,
        )

        total_steps += n_envs * n_steps

        # ── Logging ───────────────────────────────────────────────────────────
        if (update_i + 1) % args.log_interval == 0 or update_i == 0:
            elapsed = time.time() - wall_start
            fps = total_steps / elapsed

            # Mean episode return from recent completed episodes.
            if ep_returns:
                mean_ret = np.mean(ep_returns[-100:])
            else:
                mean_ret = float("nan")

            # Last epoch's mean losses.
            policy_loss = float(epoch_infos["policy_loss"][-1].mean())
            value_loss = float(epoch_infos["value_loss"][-1].mean())

            print(
                f"update {update_i+1:5d}/{n_updates} | "
                f"steps {total_steps:>10,} | "
                f"FPS {fps:,.0f} | "
                f"mean_return {mean_ret:+8.1f} | "
                f"pi_loss {policy_loss:+.4f} | "
                f"v_loss {value_loss:.4f}"
            )

            if writer:
                writer.add_scalar("rollout/mean_return", mean_ret, total_steps)
                writer.add_scalar("rollout/fps", fps, total_steps)
                writer.add_scalar("loss/policy", policy_loss, total_steps)
                writer.add_scalar("loss/value", value_loss, total_steps)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (update_i + 1) % args.save_interval == 0:
            save_path = os.path.join(POLICY_DIR, f"robot_dog_{args.task}_mjx.npz")
            _save_params(params, save_path)
            print(f"  [checkpoint] → {save_path}")

    # ── Final save ─────────────────────────────────────────────────────────────
    final_path = os.path.join(POLICY_DIR, f"robot_dog_{args.task}_mjx.npz")
    _save_params(params, final_path)

    elapsed = time.time() - wall_start
    print(f"\nDone. {total_steps:,} steps in {elapsed:.1f}s ({total_steps/elapsed:,.0f} FPS)")
    print(f"Saved → {final_path}")

    if writer:
        writer.close()


def _save_params(params, path):
    """Save JAX params as .npz for portability."""
    flat_leaves, tree_def = jax.tree_util.tree_flatten(params)
    save_dict = {f"arr_{i}": np.array(leaf) for i, leaf in enumerate(flat_leaves)}
    np.savez(path, **save_dict)
    # Save tree def separately as a pickle alongside.
    import pickle
    with open(path.replace(".npz", "_treedef.pkl"), "wb") as f:
        pickle.dump(tree_def, f)


def load_params(path):
    """Load params saved by _save_params."""
    import pickle
    data = np.load(path)
    leaves = [jnp.array(data[f"arr_{i}"]) for i in range(len(data.files))]
    with open(path.replace(".npz", "_treedef.pkl"), "rb") as f:
        tree_def = pickle.load(f)
    return tree_def.unflatten(leaves)


def main():
    parser = argparse.ArgumentParser(description="MJX GPU training")
    parser.add_argument("--task", type=str, default="stand", choices=["stand", "walk"])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=2048,
                        help="Parallel envs on GPU (default 2048, try 4096 on 4090)")
    parser.add_argument("--n-steps", type=int, default=64,
                        help="Rollout length before each PPO update")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per episode")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print stats every N updates")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save checkpoint every N updates")
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None,
                        help="Resume training. Optionally specify path to .npz params (default: auto-detect from task)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
