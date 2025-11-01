from config import Config
from loguru import logger
from typing import Tuple
import os, numpy as np, jax, jax.numpy as jnp
from jax import random
import optax, flax.linen as nn, distrax
from tqdm.auto import trange

import envpool
from utils.callbacks import JaxVideoMetricLogger

os.environ["MUJOCO_GL"] = "egl"
os.environ["DISPLAY"] = ":0"

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

# Policy-Value Network
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

# Hierarchical RL
class Manager(nn.Module):
    hidden: int
    skills: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.skills)(x)

# =====================================================
#  HELPER FUNCTIONS
# =====================================================
def tree_add(a,b): return jax.tree_util.tree_map(lambda x,y:x+y,a,b)
def tree_scale(a,s): return jax.tree_util.tree_map(lambda x:x*s,a)
def sample_noise_like(k,t):
    leaves, td = jax.tree_util.tree_flatten(t)
    ks = random.split(k, len(leaves))
    noises = [random.normal(kk, shape=x.shape) for kk,x in zip(ks,leaves)]
    return jax.tree_util.tree_unflatten(td,noises)

def one_hot_np(ids, n):
    oh = np.zeros((ids.shape[0], n), dtype=np.float32)
    oh[np.arange(ids.shape[0]), ids] = 1.0
    return oh

def select_skill_batch(params, net, obs, nskills):
    logits = net.apply(params, jnp.asarray(obs))
    return np.asarray(jnp.argmax(logits, axis=-1))

# =====================================================
#  EVOLUTION STRATEGY
# =====================================================
def es_evaluate(env, mgr_p, mgr_n, pol_p, pol_n, cfg, steps: int):
    """
    Evaluate a given manager parameter set (mgr_p) within one environment instance.

    This function rolls out the environment for a fixed number of steps using:
        - The manager (mgr_n) to select a discrete skill every H steps
        - The policy (pol_n) conditioned on that skill to produce continuous actions
    It returns the total accumulated reward (episodic return).

    Args:
        env: environment (e.g., EnvPool vectorized with 1 env)
        mgr_p: current manager parameters (θ)
        mgr_n: manager network
        pol_p: current policy parameters
        pol_n: policy network
        cfg: configuration object (contains horizon_H, skills, etc.)
        steps: number of environment steps to simulate

    Returns:
        total (float): total reward accumulated during evaluation
    """
    obs = env.reset().obs
    total = 0.0

    # Initial skill selection from manager
    skill = int(jnp.argmax(mgr_n.apply(mgr_p, jnp.expand_dims(jnp.asarray(obs[0]), 0))[0]))
    skill_oh = one_hot_np(np.array([skill]), cfg.skills)[0]
    H = cfg.horizon_H

    # Rollout loop
    for t in range(steps):

        # Periodically resample the high-level skill every H steps
        if t % H == 0:
            skill = int(jnp.argmax(mgr_n.apply(mgr_p, jnp.expand_dims(jnp.asarray(obs[0]), 0))[0]))
            skill_oh = one_hot_np(np.array([skill]), cfg.skills)[0]

        # Concatenate observation with one-hot skill vector → hierarchical input
        obs_s = np.concatenate([obs[0].astype(np.float32), skill_oh])[None]

        # Sample low-level continuous action from PPO policy
        key = random.PRNGKey((t + 7) % 2**31)
        act, _, _ = pol_n.apply(pol_p, jnp.asarray(obs_s), key)

        # Step environment and collect reward
        obs = env.step(np.asarray(act))[0]
        total += float(obs.reward[0])

        # Reset if the episode terminates early
        if obs.done[0]:
            obs = env.reset()

    return total


def es_update(make_env, mp, mn, pp, pn, cfg):
    """
    Evolution Strategies update for the manager network.

    This performs n=cfg.es_batch perturbation evaluations and computes a
    finite-difference gradient estimate for the manager parameters.

    Args:
        make_env: callable returning a new single-environment instance
        mp: current manager parameters θ
        mn: manager network
        pp: policy parameters (fixed PPO-trained low-level controller)
        pn: policy network
        cfg: configuration with ES hyperparameters (es_batch, es_sigma, es_alpha)

    Returns:
        (new_mp, rewards): updated manager parameters and list of (r_plus, r_minus) tuples
    """
    k = random.PRNGKey(cfg.seed + 123)
    g_acc = jax.tree_util.tree_map(jnp.zeros_like, mp)  # accumulated ES gradient
    rewards = []

    # Sample perturbations (ε_i) and evaluate in parallel environments
    for _ in range(cfg.es_batch):
        k, k1 = random.split(k)

        # Sample Gaussian noise ε_i for all parameters
        eps = sample_noise_like(k1, mp)

        # Antithetic parameter sets: θ+σε and θ−σε
        p_plus = tree_add(mp, tree_scale(eps,  cfg.es_sigma))
        p_minus = tree_add(mp, tree_scale(eps, -cfg.es_sigma))

        # Create independent environment copies for both evaluations
        envp, envm = make_env(), make_env()

        # Evaluate rewards of perturbed managers
        r_p = es_evaluate(envp, p_plus, mn, pp, pn, cfg, steps=512)
        r_m = es_evaluate(envm, p_minus, mn, pp, pn, cfg, steps=512)
        rewards.append((r_p, r_m))

        # Compute finite-difference gradient contribution
        #   g_i = (r_plus - r_minus) / (2σ) * ε_i
        g_i = tree_scale(eps, (r_p - r_m) / (2 * cfg.es_sigma))

        # Accumulate estimated gradients across perturbations
        g_acc = jax.tree_util.tree_map(lambda a, b: a + b, g_acc, g_i)

        # Close environments to free resources
        envp.close(); envm.close()

    # Average and scale by learning rate α
    g_acc = tree_scale(g_acc, cfg.es_alpha / float(cfg.es_batch))

    # Update manager parameters: θ ← θ + α * g_acc
    new_mp = tree_add(mp, g_acc)

    return new_mp, rewards


# =====================================================
#  TRAIN : PPO + ES + HRL
# =====================================================
def train(cfg: Config):
    base_dir="./logs"; os.makedirs(base_dir,exist_ok=True)
    exps=[int(d[3:]) for d in os.listdir(base_dir) if d.startswith("exp") and d[3:].isdigit()]
    exp_dir=os.path.join(base_dir,f"exp{max(exps)+1 if exps else 1}")
    os.makedirs(exp_dir,exist_ok=True)
    cfg.logdir=exp_dir

    n_cpus=os.cpu_count() or 4
    n_envs=getattr(cfg,"n_envs",min(28,n_cpus))
    logger.info(f"📁 {exp_dir} | Using EnvPool {n_envs} envs on {n_cpus} cores")

    # --- EnvPool vectorized env ---
    env = envpool.make_gym(cfg.env_id, num_envs=n_envs, batch_size=n_envs)
    obs = env.reset().obs
    obs_dim = obs.shape[1]
    act_dim = env.action_space.shape[0]

    # --- Nets ---
    input_dim=obs_dim+cfg.skills
    key=random.PRNGKey(cfg.seed)
    pol_net=PolicyValueNet(cfg.policy_hidden, act_dim)
    pol_p=pol_net.init(key,jnp.zeros((1,input_dim)),key)
    opt=optax.adam(cfg.ppo_lr)
    opt_state=opt.init(pol_p)

    mgr_net=Manager(cfg.policy_hidden,cfg.skills)
    mgr_p=mgr_net.init(random.PRNGKey(cfg.seed+999),jnp.zeros((1,obs_dim)))

    gamma,clip=cfg.ppo_gamma,cfg.ppo_clip
    tb=JaxVideoMetricLogger(cfg)

    # PPO loss
    def ppo_loss(params, obs_s, act, adv, oldlp, ret, rng):
        a, lp, v = pol_net.apply(params, obs_s, rng)
        ratio = jnp.exp(lp - oldlp)
        clip_adv = jnp.clip(ratio, 1-clip, 1+clip)*adv
        actor = -jnp.mean(jnp.minimum(ratio*adv, clip_adv))
        critic = jnp.mean((ret-v)**2)
        ent = -jnp.mean(lp)
        total = actor + 0.5*critic - 0.01*ent
        return total, dict(actor_loss=actor, critic_loss=critic, entropy=ent, value_mean=jnp.mean(v))

    @jax.jit
    def ppo_update(p, s, b, k):
        (l,m),g=jax.value_and_grad(ppo_loss,has_aux=True)(p,*b,k)
        u,s=opt.update(g,s)
        return optax.apply_updates(p,u),s,m

    H=cfg.horizon_H
    ep_ret=np.zeros((n_envs,),np.float32)
    succ=np.zeros((n_envs,),np.float32)

    sk_ids=select_skill_batch(mgr_p,mgr_net,obs,cfg.skills)
    sk_oh=one_hot_np(sk_ids,cfg.skills)

    for step in trange(cfg.total_timesteps):
        tb.global_step=step
        if step%H==0:
            sk_ids=select_skill_batch(mgr_p,mgr_net,obs,cfg.skills)
            sk_oh=one_hot_np(sk_ids,cfg.skills)

        obs_s=np.concatenate([obs.astype(np.float32),sk_oh],1)
        key,sub=random.split(key)
        acts,logp,v=pol_net.apply(pol_p,jnp.asarray(obs_s),sub)
        trans=env.step(np.asarray(acts))
        next_obs,rew,done,trunc,info=trans.obs,trans.reward,trans.done,trans.trunc,trans.info
        done=np.logical_or(done,trunc)
        ep_ret+=rew
        obs_s2=np.concatenate([next_obs.astype(np.float32),sk_oh],1)
        _,_,v2=pol_net.apply(pol_p,jnp.asarray(obs_s2),sub)
        td=rew+gamma*np.asarray(v2)*(~done)
        adv=td-np.asarray(v)

        batch=(jnp.asarray(obs_s),jnp.asarray(acts),jnp.asarray(adv),
               jnp.asarray(logp),jnp.asarray(td))
        pol_p,opt_state,met=ppo_update(pol_p,opt_state,batch,sub)

        tb.log_scalar("train/return",float(np.mean(ep_ret)))
        tb.log_scalar("train/actor_loss",float(met["actor_loss"]))
        tb.log_scalar("train/critic_loss",float(met["critic_loss"]))
        tb.log_scalar("train/entropy",float(met["entropy"]))
        tb.log_scalar("train/value_mean",float(met["value_mean"]))

        if np.any(done):
            # reset done envs efficiently
            next_obs = env.reset_done(done).obs
            tb.log_scalar("custom/episodic_return",float(np.mean(ep_ret[done])))
            ep_ret[done]=0

        obs=next_obs

        if (step+1)%10000==0:
            logger.info("[ES] manager update")
            mgr_p,stats=es_update(lambda: envpool.make_gym(cfg.env_id,num_envs=1,batch_size=1),
                                  mgr_p,mgr_net,pol_p,pol_net,cfg)
            m_r=np.mean([0.5*(rp+rm) for rp,rm in stats]) if stats else 0
            tb.log_scalar("es/mean_pair_return",float(m_r))
            tb.record_eval_video(cfg.env_id,pol_net,pol_p,mgr_net,mgr_p)

    env.close()
    tb.close()

if __name__=="__main__":
    cfg=Config()
    train(cfg)
