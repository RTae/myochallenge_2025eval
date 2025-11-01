from config import Config
from loguru import logger
from typing import Tuple

import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax
import flax.linen as nn
import distrax
from tqdm.auto import trange

from myosuite.utils import gym
from utils.callbacks import JaxVideoMetricLogger


# =====================================================
#  SAFE OBSERVATION WRAPPER
# =====================================================
class SafeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.full(env.observation_space.shape, -np.inf, dtype=np.float32)
        high = np.full(env.observation_space.shape, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, obs):
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)


def make_env(env_id: str, seed: int):
    def _thunk():
        env = gym.make(env_id)
        env = SafeObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env
    return _thunk


# =====================================================
#  NETWORKS
# =====================================================
class MLP(nn.Module):
    features: Tuple[int, ...]

    @nn.compact
    def __call__(self, x):
        for f in self.features[:-1]:
            x = nn.relu(nn.Dense(f)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


class PolicyValueNet(nn.Module):
    hidden: int
    act_dim: int

    @nn.compact
    def __call__(self, x, rng_key):
        h = MLP((self.hidden, self.hidden))(x)
        mu = nn.Dense(self.act_dim)(h)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.act_dim,))
        std = jnp.exp(log_std)
        dist = distrax.Normal(mu, std)
        action = dist.sample(seed=rng_key)
        log_prob = jnp.sum(dist.log_prob(action), axis=-1)
        value = nn.Dense(1)(h)
        return jnp.tanh(action), log_prob, jnp.squeeze(value, -1)


class Manager(nn.Module):
    hidden: int
    skills: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        logits = nn.Dense(self.skills)(x)
        return logits


# =====================================================
#  PPO + ES MANAGER
# =====================================================
def tree_add(tree, upd):
    return jax.tree_util.tree_map(lambda a, b: a + b, tree, upd)


def tree_scale(tree, scale):
    return jax.tree_util.tree_map(lambda a: a * scale, tree)


def sample_noise_like(key, tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = random.split(key, len(leaves))
    noisy = [random.normal(k, shape=x.shape) for k, x in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(treedef, noisy)


def one_hot(i, n):
    v = np.zeros((n,), dtype=np.float32)
    v[i] = 1.0
    return v


def select_skill(manager_params, manager_net, obs_np, cfg: Config):
    logits = manager_net.apply(manager_params, jnp.expand_dims(jnp.asarray(obs_np), 0))
    sk = int(jnp.argmax(logits, axis=-1)[0])
    return sk


# =====================================================
#  EVOLUTION STRATEGY
# =====================================================
def es_evaluate(env, manager_params, manager_net, policy_params, policy_net, cfg: Config, steps: int):
    obs, _ = env.reset()
    total_r = 0.0
    skill_id = select_skill(manager_params, manager_net, obs, cfg)
    skill_oh = one_hot(skill_id, cfg.skills)
    H = cfg.horizon_H
    for t in range(steps):
        if t % H == 0:
            skill_id = select_skill(manager_params, manager_net, obs, cfg)
            skill_oh = one_hot(skill_id, cfg.skills)
        obs_skill = np.concatenate([obs.astype(np.float32), skill_oh], axis=-1)
        key = random.PRNGKey((t + 7) % 2**31)
        act, _, _ = policy_net.apply(policy_params, jnp.expand_dims(jnp.asarray(obs_skill), 0), key)
        next_obs, r, done, trunc, _ = env.step(np.asarray(act[0]))
        total_r += float(r)
        obs = next_obs
        if done or trunc:
            obs, _ = env.reset()
    return total_r


def es_update(env_maker, manager_params, manager_net, policy_params, policy_net, cfg: Config):
    key = random.PRNGKey(cfg.seed + 123)
    grads_acc = jax.tree_util.tree_map(jnp.zeros_like, manager_params)
    rewards = []
    for i in range(cfg.es_batch):
        key, k1 = random.split(key)
        eps = sample_noise_like(k1, manager_params)
        params_plus = tree_add(manager_params, tree_scale(eps, cfg.es_sigma))
        params_minus = tree_add(manager_params, tree_scale(eps, -cfg.es_sigma))
        env_p = env_maker()
        env_m = env_maker()
        r_plus = es_evaluate(env_p, params_plus, manager_net, policy_params, policy_net, cfg, steps=512)
        r_minus = es_evaluate(env_m, params_minus, manager_net, policy_params, policy_net, cfg, steps=512)
        rewards.append((r_plus, r_minus))
        g_i = tree_scale(eps, (r_plus - r_minus) / (2.0 * cfg.es_sigma))
        grads_acc = jax.tree_util.tree_map(lambda a, b: a + b, grads_acc, g_i)
        env_p.close(); env_m.close()
    grads_acc = tree_scale(grads_acc, cfg.es_alpha / float(cfg.es_batch))
    new_params = tree_add(manager_params, grads_acc)
    return new_params, rewards


# =====================================================
#  TRAIN
# =====================================================
def train(cfg: Config):
    # --- Auto-generate log path ./logs/exp1, exp2, exp3 ---
    base_dir = "./logs"
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("exp") and os.path.isdir(os.path.join(base_dir, d))]
    exp_nums = [int(d.replace("exp", "")) for d in existing if d.replace("exp", "").isdigit()]
    next_exp = max(exp_nums) + 1 if exp_nums else 1
    exp_dir = os.path.join(base_dir, f"exp{next_exp}")
    os.makedirs(exp_dir, exist_ok=True)
    cfg.logdir = exp_dir

    logger.info(f"📁 Starting new experiment: {exp_dir}")

    # === Env + setup ===
    env = make_env(cfg.env_id, cfg.seed)()
    obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # === Nets ===
    input_dim = obs_dim + cfg.skills
    policy_net = PolicyValueNet(cfg.policy_hidden, act_dim)
    key = random.PRNGKey(cfg.seed)
    policy_params = policy_net.init(key, jnp.zeros((1, input_dim)), key)
    optimizer = optax.adam(cfg.ppo_lr)
    opt_state = optimizer.init(policy_params)

    manager_net = Manager(cfg.policy_hidden, cfg.skills)
    manager_params = manager_net.init(random.PRNGKey(cfg.seed + 999), jnp.zeros((1, obs_dim)))

    gamma = cfg.ppo_gamma
    clip = cfg.ppo_clip
    logger_tb = JaxVideoMetricLogger(cfg)

    # === PPO loss/update ===
    def ppo_loss(params, obs_skill, actions, adv, old_logp, returns, key):
        a, logp, v = policy_net.apply(params, obs_skill, key)
        ratio = jnp.exp(logp - old_logp)
        clip_adv = jnp.clip(ratio, 1 - clip, 1 + clip) * adv
        policy_loss = -jnp.mean(jnp.minimum(ratio * adv, clip_adv))
        value_loss = jnp.mean((returns - v) ** 2)
        entropy = -jnp.mean(logp)
        return policy_loss + 0.5 * value_loss - 0.01 * entropy

    @jax.jit
    def ppo_update(params, opt_state, batch, key):
        grads = jax.grad(ppo_loss)(params, *batch, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    # === Training ===
    H = cfg.horizon_H
    total_return = 0.0
    skill_id = select_skill(manager_params, manager_net, obs, cfg)
    skill_oh = one_hot(skill_id, cfg.skills)

    logger.info(f"[Start] worker on {cfg.env_id}")
    for step in trange(cfg.total_timesteps):
        logger_tb.global_step = step

        if step % H == 0:
            skill_id = select_skill(manager_params, manager_net, obs, cfg)
            skill_oh = one_hot(skill_id, cfg.skills)

        obs_skill = np.concatenate([obs.astype(np.float32), skill_oh], axis=-1)
        key, sub = random.split(key)
        act, logp, v = policy_net.apply(policy_params, jnp.expand_dims(jnp.asarray(obs_skill), 0), sub)
        next_obs, r, done, trunc, _ = env.step(np.asarray(act[0]))
        total_return += float(r)

        next_obs_skill = np.concatenate([next_obs.astype(np.float32), skill_oh], axis=-1)
        _, _, v_next = policy_net.apply(policy_params, jnp.expand_dims(jnp.asarray(next_obs_skill), 0), sub)
        td_target = float(r) + (0.0 if (done or trunc) else gamma * float(v_next[0]))
        adv = td_target - float(v[0])

        batch = (
            jnp.expand_dims(jnp.asarray(obs_skill), 0),
            jnp.expand_dims(jnp.asarray(act[0]), 0),
            jnp.asarray([adv], dtype=jnp.float32),
            jnp.asarray([float(logp[0])], dtype=jnp.float32),
            jnp.asarray([td_target], dtype=jnp.float32),
        )
        policy_params, opt_state = ppo_update(policy_params, opt_state, batch, sub)

        logger_tb.log_scalar("train/return", total_return)
        logger_tb.log_scalar("train/advantage", float(adv))

        obs = next_obs
        if done or trunc:
            obs, _ = env.reset()
            skill_id = select_skill(manager_params, manager_net, obs, cfg)
            skill_oh = one_hot(skill_id, cfg.skills)

        if (step + 1) % 10000 == 0:
            logger.info("[ES] Updating manager with OpenES...")
            manager_params, es_stats = es_update(lambda: make_env(cfg.env_id, cfg.seed)(),
                                                 manager_params, manager_net,
                                                 policy_params, policy_net, cfg)
            mean_r = np.mean([0.5*(rp+rm) for rp, rm in es_stats]) if len(es_stats) else 0.0
            logger.info(f"[ES] mean pair return ~ {mean_r:.2f}")
            logger_tb.log_scalar("es/mean_pair_return", float(mean_r))
            logger_tb.record_eval_video(cfg.env_id, policy_net, policy_params, manager_net, manager_params)

        if (step + 1) % 5000 == 0:
            logger.info(f"Step {step+1} | running return ~ {total_return:.2f}")

    env.close()
    logger_tb.close()


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
