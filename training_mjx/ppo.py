"""
Minimal PPO implementation in JAX for MJX environments.

This is a single-file, dependency-light PPO that runs entirely on GPU.
No SB3, no external RL library — just JAX + Flax + Optax.

Key design choices:
- Everything stays on GPU (obs, actions, advantages, gradients).
- The policy and value networks are small MLPs (matching SB3 defaults).
- Generalized Advantage Estimation (GAE) for variance reduction.
- Clipped surrogate objective (standard PPO).
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


# ── Policy / Value network ─────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """MLP policy + value head. Same architecture as SB3 MlpPolicy defaults."""
    action_dim: int
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x):
        # Shared feature extraction? No — SB3 default uses separate nets.
        # Policy head.
        pi = nn.Dense(self.hidden_size)(x)
        pi = nn.tanh(pi)
        pi = nn.Dense(self.hidden_size)(pi)
        pi = nn.tanh(pi)
        mean = nn.Dense(self.action_dim)(pi)

        # Log std as a free parameter (not state-dependent), like SB3.
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))

        # Value head.
        v = nn.Dense(self.hidden_size)(x)
        v = nn.tanh(v)
        v = nn.Dense(self.hidden_size)(v)
        v = nn.tanh(v)
        value = nn.Dense(1)(v)

        return mean, log_std, value.squeeze(-1)


# ── Trajectory buffer ──────────────────────────────────────────────────────────

class Transition(NamedTuple):
    obs: jnp.ndarray        # (n_envs, obs_dim)
    action: jnp.ndarray     # (n_envs, act_dim)
    reward: jnp.ndarray     # (n_envs,)
    done: jnp.ndarray       # (n_envs,)
    log_prob: jnp.ndarray   # (n_envs,)
    value: jnp.ndarray      # (n_envs,)


# ── GAE computation ────────────────────────────────────────────────────────────

def compute_gae(transitions, last_value, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation over a rollout."""

    def _scan_fn(carry, transition):
        next_value, next_advantage = carry
        delta = transition.reward + gamma * next_value * (1 - transition.done) - transition.value
        advantage = delta + gamma * gae_lambda * (1 - transition.done) * next_advantage
        return (transition.value, advantage), advantage

    _, advantages = jax.lax.scan(
        _scan_fn,
        (last_value, jnp.zeros_like(last_value)),
        transitions,
        reverse=True,
    )
    returns = advantages + transitions.value
    return advantages, returns


# ── PPO update ─────────────────────────────────────────────────────────────────

def ppo_loss(params, apply_fn, batch, clip_range=0.2, vf_coef=0.5, ent_coef=0.01):
    """PPO clipped surrogate loss + value loss + entropy bonus."""
    mean, log_std, value = apply_fn(params, batch.obs)
    std = jnp.exp(log_std)

    # Log prob of the action under the current policy.
    log_prob = -0.5 * jnp.sum(((batch.action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)

    # Policy loss (clipped surrogate).
    ratio = jnp.exp(log_prob - batch.log_prob)
    advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = jnp.mean(jnp.maximum(loss_unclipped, loss_clipped))

    # Value loss (clipped, like SB3).
    value_loss = 0.5 * jnp.mean((value - batch.returns) ** 2)

    # Entropy bonus.
    entropy = 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * std ** 2))

    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return total_loss, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}


class PPOBatch(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray


def make_ppo_update(network, optimizer, n_epochs=10, batch_size=64, clip_range=0.2,
                    vf_coef=0.5, ent_coef=0.01):
    """
    Returns a JIT-compiled function that performs PPO updates on a rollout.
    """

    @functools.partial(jax.jit)
    def update(params, opt_state, transitions, last_value, rng):
        # Compute advantages.
        advantages, returns = compute_gae(transitions, last_value)

        # Flatten (n_steps, n_envs, ...) -> (n_steps * n_envs, ...).
        def flatten(x):
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        flat = PPOBatch(
            obs=flatten(transitions.obs),
            action=flatten(transitions.action),
            log_prob=flatten(transitions.log_prob),
            advantages=flatten(advantages),
            returns=flatten(returns),
        )

        n_samples = flat.obs.shape[0]

        def _epoch(carry, rng_epoch):
            params, opt_state = carry
            # Shuffle and split into minibatches.
            perm = jax.random.permutation(rng_epoch, n_samples)
            n_batches = max(1, n_samples // batch_size)

            def _minibatch(carry, idx):
                params, opt_state = carry
                start = idx * batch_size
                mb_idx = jax.lax.dynamic_slice(perm, (start,), (batch_size,))
                mb = PPOBatch(
                    obs=flat.obs[mb_idx],
                    action=flat.action[mb_idx],
                    log_prob=flat.log_prob[mb_idx],
                    advantages=flat.advantages[mb_idx],
                    returns=flat.returns[mb_idx],
                )
                grads, info = jax.grad(ppo_loss, has_aux=True)(
                    params, network.apply, mb, clip_range, vf_coef, ent_coef,
                )
                updates, opt_state_new = optimizer.update(grads, opt_state, params)
                params_new = optax.apply_updates(params, updates)
                return (params_new, opt_state_new), info

            (params, opt_state), infos = jax.lax.scan(
                _minibatch, (params, opt_state), jnp.arange(n_batches)
            )
            return (params, opt_state), infos

        rngs = jax.random.split(rng, n_epochs)
        (params, opt_state), epoch_infos = jax.lax.scan(
            _epoch, (params, opt_state), rngs
        )

        return params, opt_state, epoch_infos

    return update
