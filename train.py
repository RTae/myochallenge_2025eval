from config import Config
from loguru import logger
from typing import Tuple
import os, time, numpy as np, jax, jax.numpy as jnp
from jax import random
import optax, flax.linen as nn, distrax
from tqdm.auto import trange

from myosuite.utils import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.callbacks import VideoMetricLogger
from utils.model_helper import one_hot, sample_noise_like, tree_add, tree_scale, select_skill

# =====================================================
#  Runtime configuration
# =====================================================
os.environ["MUJOCO_GL"] = "egl"
os.environ.pop("DISPLAY", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# =====================================================
#  NETWORKS
# =====================================================
class MLP(nn.Module):
    features: Tuple[int, ...]
    @nn.compact
    def __call__(self, x):
        for f in self.features[:-1]:
            x = nn.relu(nn.Dense(f)(x))
        return nn.Dense(self.features[-1])(x)

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
        act = dist.sample(seed=rng_key)
        logp = jnp.sum(dist.log_prob(act), axis=-1)
        val = nn.Dense(1)(h)
        return jnp.tanh(act), logp, jnp.squeeze(val, -1)

class Manager(nn.Module):
    hidden: int
    skills: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.skills)(x)

# =====================================================
#  VECTOR ENV (multiprocess)
# =====================================================
def make_vec_env(env_id, seed, n_envs):
    """Create parallel MyoSuite envs using subprocesses for true CPU parallelism."""
    def make_one(i):
        def _factory():
            e = gym.make(env_id)
            e = gym.wrappers.RecordEpisodeStatistics(e)
            e.reset(seed=seed + i)
            return e
        return _factory
    return SubprocVecEnv([make_one(i) for i in range(n_envs)])

# =====================================================
#  EVOLUTION STRATEGY
# =====================================================
def es_evaluate(env, mgr_p, mgr_n, pol_p, pol_n, cfg, steps: int):
    obs, _ = env.reset()
    total = 0.0
    skill = int(jnp.argmax(mgr_n.apply(mgr_p, jnp.expand_dims(jnp.asarray(obs), 0))[0]))
    skill_oh = one_hot(np.array([skill]), cfg.skills)[0]
    H = cfg.horizon_H
    for t in range(steps):
        if t % H == 0:
            skill = int(jnp.argmax(mgr_n.apply(mgr_p, jnp.expand_dims(jnp.asarray(obs), 0))[0]))
            skill_oh = one_hot(np.array([skill]), cfg.skills)[0]
        obs_s = np.concatenate([obs.astype(np.float32), skill_oh])[None]
        key = random.PRNGKey((t + 7) % 2**31)
        act, _, _ = pol_n.apply(pol_p, jnp.asarray(obs_s), key)
        obs, r, done, trunc, _ = env.step(np.asarray(act[0]))
        total += float(r)
        if done or trunc:
            obs, _ = env.reset()
    return total

def es_update(make_env, mp, mn, pp, pn, cfg):
    k = random.PRNGKey(cfg.seed + 123)
    g_acc = jax.tree_util.tree_map(jnp.zeros_like, mp)
    rewards = []
    for _ in range(cfg.es_batch):
        k, k1 = random.split(k)
        eps = sample_noise_like(k1, mp)
        p_plus  = tree_add(mp, tree_scale(eps,  cfg.es_sigma))
        p_minus = tree_add(mp, tree_scale(eps, -cfg.es_sigma))
        envp, envm = make_env(), make_env()
        r_p = es_evaluate(envp, p_plus,  mn, pp, pn, cfg, steps=512)
        r_m = es_evaluate(envm, p_minus, mn, pp, pn, cfg, steps=512)
        rewards.append((r_p, r_m))
        g_i = tree_scale(eps, (r_p - r_m) / (2 * cfg.es_sigma))
        g_acc = jax.tree_util.tree_map(lambda a, b: a + b, g_acc, g_i)
        envp.close(); envm.close()
    g_acc = tree_scale(g_acc, cfg.es_alpha / float(cfg.es_batch))
    return tree_add(mp, g_acc), rewards

# =====================================================
#  TRAIN (PPO + ES + HRL)
# =====================================================
def train(cfg: Config):
    # --- Auto-increment logdir ---
    base_dir = "./logs"; os.makedirs(base_dir, exist_ok=True)
    exps = [int(d[3:]) for d in os.listdir(base_dir) if d.startswith("exp") and d[3:].isdigit()]
    exp_dir = os.path.join(base_dir, f"exp{max(exps)+1 if exps else 1}")
    os.makedirs(exp_dir, exist_ok=True)
    cfg.logdir = exp_dir

    cpus = os.cpu_count() or 4
    n_envs = cfg.n_envs or max(4, cpus)
    logger.info(f"📁 {exp_dir} | using {n_envs} envs")

    # --- Create vectorized MyoSuite env ---
    env = make_vec_env(cfg.env_id, cfg.seed, n_envs)
    obs = env.reset()
    if isinstance(obs, tuple):  # Gymnasium compatibility
        obs = obs[0]
    obs_dim = obs.shape[1]
    act_dim = env.action_space.shape[0]

    # --- Networks ---
    input_dim = obs_dim + cfg.skills
    key = random.PRNGKey(cfg.seed)
    pol_net = PolicyValueNet(cfg.policy_hidden, act_dim)
    pol_p   = pol_net.init(key, jnp.zeros((1, input_dim)), key)
    opt     = optax.adam(cfg.ppo_lr)
    opt_state = opt.init(pol_p)

    mgr_net = Manager(cfg.policy_hidden, cfg.skills)
    mgr_p   = mgr_net.init(random.PRNGKey(cfg.seed + 999), jnp.zeros((1, obs_dim)))

    gamma, clip = cfg.ppo_gamma, cfg.ppo_clip
    tb = VideoMetricLogger(cfg)

    # --- PPO loss ---
    def ppo_loss(params, obs_s, act, adv, oldlp, ret, rng):
        a, lp, v = pol_net.apply(params, obs_s, rng)
        ratio    = jnp.exp(lp - oldlp)
        clip_adv = jnp.clip(ratio, 1 - clip, 1 + clip) * adv
        actor    = -jnp.mean(jnp.minimum(ratio * adv, clip_adv))
        critic   = jnp.mean((ret - v) ** 2)
        ent      = -jnp.mean(lp)
        total    = actor + 0.5 * critic - 0.01 * ent
        return total, dict(actor_loss=actor, critic_loss=critic, entropy=ent, value_mean=jnp.mean(v))

    @jax.jit
    def ppo_update(p, s, b, k):
        (l, m), g = jax.value_and_grad(ppo_loss, has_aux=True)(p, *b, k)
        u, s = opt.update(g, s)
        return optax.apply_updates(p, u), s, m

    # --- Train loop ---
    H = cfg.horizon_H
    ep_ret = np.zeros((n_envs,), np.float32)
    sk_ids = select_skill(mgr_p, mgr_net, obs, cfg.skills)
    sk_oh  = one_hot(sk_ids, cfg.skills)

    t0 = time.time()
    steps_acc = 0

    for step in trange(cfg.total_timesteps, desc="training"):
        tb.global_step = step

        if step % H == 0:
            sk_ids = select_skill(mgr_p, mgr_net, obs, cfg.skills)
            sk_oh  = one_hot(sk_ids, cfg.skills)

        obs_s = np.concatenate([obs.astype(np.float32), sk_oh], 1)
        key, sub = random.split(key)
        acts, logp, v = pol_net.apply(pol_p, jnp.asarray(obs_s), sub)

        next_obs, rew, done, info = env.step(np.asarray(acts))
        ep_ret += rew

        obs_s2 = np.concatenate([next_obs.astype(np.float32), sk_oh], 1)
        _, _, v2 = pol_net.apply(pol_p, jnp.asarray(obs_s2), sub)
        td  = rew + gamma * np.asarray(v2) * (~done)
        adv = td - np.asarray(v)

        batch = (
            jnp.asarray(obs_s),
            jnp.asarray(acts),
            jnp.asarray(adv),
            jnp.asarray(logp),
            jnp.asarray(td),
        )
        pol_p, opt_state, met = ppo_update(pol_p, opt_state, batch, sub)

        tb.log_scalar("train/return", float(np.mean(ep_ret)))
        tb.log_scalar("train/actor_loss", float(met["actor_loss"]))
        tb.log_scalar("train/critic_loss", float(met["critic_loss"]))
        tb.log_scalar("train/entropy", float(met["entropy"]))
        tb.log_scalar("train/value_mean", float(met["value_mean"]))

        if np.any(done):
            tb.log_scalar("custom/episodic_return", float(np.mean(ep_ret[done])))
            ep_ret[done] = 0.0

        obs = next_obs

        # Performance log
        steps_acc += n_envs
        if steps_acc >= 5000:
            dt = time.time() - t0
            rate = steps_acc / max(dt, 1e-6)
            tb.log_scalar("sys/env_steps_per_sec", float(rate))
            logger.info(f"⚡ {rate:.0f} env-steps/sec @ {n_envs} envs")
            t0 = time.time()
            steps_acc = 0

        # Periodic ES update + video
        if (step + 1) % cfg.video_freq == 0:
            logger.info("[ES] manager update + video checkpoint")
            mgr_p, stats = es_update(lambda: gym.make(cfg.env_id), mgr_p, mgr_net, pol_p, pol_net, cfg)
            mean_r = np.mean([0.5 * (rp + rm) for rp, rm in stats]) if stats else 0.0
            tb.log_scalar("es/mean_pair_return", float(mean_r))
            tb.record_eval_video(cfg.env_id, pol_net, pol_p, mgr_net, mgr_p)

    env.close()
    tb.close()

if __name__ == "__main__":
    cfg = Config()
    train(cfg)
