import os
import numpy as np

from loguru import logger
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from lattice.ppo.policies import LatticeRecurrentActorCriticPolicy
from lattice.sac.policies import LatticeSACPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from myosuite.utils import gym

from config import Config
from utils.callbacks import VideoCallback

os.environ["MUJOCO_GL"] = "egl"
os.environ.pop("DISPLAY", None)  # ensure no X11 display is used

# =====================================================
#  Auto-increment log dir (unchanged)
# =====================================================
def next_exp_dir(base="./logs") -> str:
    os.makedirs(base, exist_ok=True)
    exps = [int(d[3:]) for d in os.listdir(base) if d.startswith("exp") and d[3:].isdigit()]
    exp_dir = os.path.join(base, f"exp{max(exps)+1 if exps else 1}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


# =====================================================
#  Vector Env Factory (supports optional wrapper)
# =====================================================
def make_vec_env(env_id: str, seed: int, n_envs: int, wrapper_fn=None) -> SubprocVecEnv:
    """
    Create parallel MyoSuite envs using subprocesses for true CPU parallelism.
    Optionally wraps each env with `wrapper_fn(env) -> env` (e.g., HRL/SHRL wrapper).
    """
    def thunk(rank: int):
        def _init():
            env = gym.make(env_id)
            if wrapper_fn is not None:
                env = wrapper_fn(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    set_random_seed(seed)
    return SubprocVecEnv([thunk(i) for i in range(n_envs)])


# =====================================================
#  Subgoal Reward Wrapper (HRL / SHRL)
# =====================================================
class SubgoalRewardWrapper(gym.Wrapper):
    """Adds intrinsic reward based on progress toward a periodically refreshed subgoal."""
    def __init__(self, env, subgoal_interval=10, intrinsic_coef=0.1):
        super().__init__(env)
        self.subgoal_interval = max(1, subgoal_interval)
        self.intrinsic_coef = intrinsic_coef
        self._t = 0
        self._subgoal = None
        self._ext_scale = 1.0  # running magnitude of extrinsic reward for simple normalization

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._t = 0
        self._subgoal = self._sample_subgoal(obs)
        return obs

    def step(self, action):
        result = self.env.step(action)

        # Gymnasium: (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            obs, ext_r, terminated, truncated, info = result
            done = terminated or truncated
        else:  # older Gym: (obs, reward, done, info)
            obs, ext_r, done, info = result

        self._t += 1
        if self._t % self.subgoal_interval == 0:
            self._subgoal = self._sample_subgoal(obs)

        intrinsic = -np.linalg.norm(self._obs_to_vec(obs) - self._subgoal)
        shaped = ext_r + self.intrinsic_coef * intrinsic

        info["ext_r"] = ext_r
        info["intr_r"] = intrinsic
        info["shaped_r"] = shaped
        return obs, shaped, done, info


    def _sample_subgoal(self, obs):
        v = self._obs_to_vec(obs)
        return v + np.random.normal(0, 0.05, size=v.shape)

    def _obs_to_vec(self, obs):
        """
        Safely flatten observation to 1D float32 vector.
        Works with dicts, tuples, lists, and nested arrays (MyoSuite-friendly).
        """
        if isinstance(obs, dict):
            parts = []
            for v in obs.values():
                parts.append(self._obs_to_vec(v))
            return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)

        elif isinstance(obs, (list, tuple)):
            parts = []
            for v in obs:
                parts.append(self._obs_to_vec(v))
            return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)

        else:
            arr = np.asarray(obs, dtype=np.float32).reshape(-1)
            return arr



# =====================================================
#  Main train()  — HRL (no ES)
# =====================================================
def train(cfg: Config):
    # --- Runtime env vars (headless EGL etc.) ---
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.pop("DISPLAY", None)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # --- Logs ---
    exp_dir = next_exp_dir("./logs")
    cfg.logdir = exp_dir
    logger.info(f"📁 {exp_dir}")

    # --- Vec envs ---
    cpus = os.cpu_count() or 4
    n_envs = max(cfg.n_envs, cpus)
    logger.info(f"Using {n_envs} envs")

    # HRL: wrap each env with subgoal-based intrinsic reward
    vec_env = make_vec_env(
        cfg.env_id, cfg.seed, n_envs,
        wrapper_fn=lambda e: SubgoalRewardWrapper(e, subgoal_interval=getattr(cfg, "horizon_H", 20), intrinsic_coef=0.1)
    )
    vec_env = VecMonitor(vec_env)

    # --- PPO Model (worker policy) ---
    policy_kwargs = dict(
        net_arch=dict(
            pi=[cfg.policy_hidden, cfg.policy_hidden],
            vf=[cfg.policy_hidden, cfg.policy_hidden],
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=cfg.ppo_lr,
        n_steps=getattr(cfg, "n_steps", 2048 // n_envs * n_envs),
        batch_size=getattr(cfg, "n_steps", 2048 // n_envs * n_envs) * n_envs,
        n_epochs=getattr(cfg, "ppo_epochs", 10),
        gamma=getattr(cfg, "ppo_gamma", 0.99),
        gae_lambda=getattr(cfg, "gae_lambda", 0.95),
        clip_range=getattr(cfg, "ppo_clip", 0.2),
        ent_coef=getattr(cfg, "ent_coef", 0.0),
        vf_coef=getattr(cfg, "vf_coef", 0.5),
        max_grad_norm=getattr(cfg, "max_grad_norm", 0.5),
        seed=cfg.seed,
        tensorboard_log=exp_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device='cpu',
    )

    # --- Lattice SAC ---
    # model = SAC(policy=LatticeSACPolicy,
    #     env=vec_env,
    #     device='auto',
    #     learning_rate=3e-4,
    #     buffer_size=300_000,
    #     learning_starts=10000,
    #     batch_size=256,
    #     tau=0.02,
    #     gamma=0.98,
    #     train_freq=(8, "step"),
    #     gradient_steps=8,
    #     action_noise=None,
    #     replay_buffer_class=None,
    #     ent_coef="auto",
    #     target_update_interval=1,
    #     target_entropy="auto",
    #     use_sde=False,
    #     sde_sample_freq=1,
    #     policy_kwargs=dict(
    #         **policy_kwargs,
    #         use_lattice=True,
    #         use_expln=True,
    #         log_std_init=0.0,
    #         # activation_fn=nn.GELU,
    #         std_clip=(1e-3, 1),
    #         expln_eps=1e-6,
    #         clip_mean=2.0,
    #         std_reg=0.0
    #     ),)
    
    # --- Lattice PPO ---
    # model = RecurrentPPO(policy=LatticeRecurrentActorCriticPolicy, 
    #     env=vec_env,
    #     device='auto',
    #     batch_size=32,
    #     n_steps=128,
    #     learning_rate=2.55673e-05,
    #     ent_coef=3.62109e-06,
    #     clip_range=0.3,
    #     gamma=0.99,
    #     gae_lambda=0.9,
    #     max_grad_norm=0.7,
    #     vf_coef=0.835671,
    #     n_epochs=10,
    #     use_sde=False,
    #     sde_sample_freq=1,
    #     policy_kwargs=dict(
    #         **policy_kwargs,
    #         use_lattice=True,
    #         use_expln=True,
    #         ortho_init=False,
    #         log_std_init=0.0,
    #         # activation_fn=nn.ReLU,
    #         std_clip=(1e-3, 10),
    #         expln_eps=1e-6,
    #         full_std=False,
    #         std_reg=0.0,
    #     ),)

    # --- Callbacks: Video + Evaluation ---
    eval_env = make_vec_env(cfg.env_id, cfg.seed + 999, 1,
                            wrapper_fn=lambda e: SubgoalRewardWrapper(e, subgoal_interval=getattr(cfg, "horizon_H", 20), intrinsic_coef=0.1))
    eval_env = VecMonitor(eval_env)

    video_cb = VideoCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        logdir=exp_dir,
        video_freq=getattr(cfg, "video_freq", 50_000),
        eval_episodes=getattr(cfg, "eval_episodes", 1),
        verbose=0,
    )

    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=exp_dir,
        log_path=exp_dir,
        eval_freq=getattr(cfg, "eval_freq", 25_000),
        deterministic=True,
        render=False,
    )

    callback_list = [video_cb, eval_cb]

    # --- Train ---
    total_timesteps = int(cfg.total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)

    # --- Save artifacts ---
    save_path = os.path.join(exp_dir, "hrl_policy.zip")
    model.save(save_path)
    logger.info(f"Saved model to {save_path}")

    # Close envs
    vec_env.close()

if __name__ == "__main__":
    cfg = Config()
    train(cfg)
