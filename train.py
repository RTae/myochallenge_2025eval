import os

from loguru import logger

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from myosuite.utils import gym

from config import Config
from utils.callbacks import VideoCallback

os.environ["MUJOCO_GL"] = "egl"
os.environ.pop("DISPLAY", None)  # ensure no X11 display is used

# =====================================================
#  Vector Env Factory
# =====================================================
def make_vec_env(env_id: str, seed: int, n_envs: int) -> SubprocVecEnv:
    """
    Create parallel MyoSuite envs using subprocesses for true CPU parallelism.
    """
    def thunk(rank: int):
        def _init():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # Optional: add other wrappers here (clip actions, obs norm, etc.)
            # Note: seed per-subproc
            env.reset(seed=seed + rank)
            return env
        return _init

    set_random_seed(seed)
    return SubprocVecEnv([thunk(i) for i in range(n_envs)])


# =====================================================
#  Auto-increment log dir
# =====================================================
def next_exp_dir(base="./logs") -> str:
    os.makedirs(base, exist_ok=True)
    exps = [int(d[3:]) for d in os.listdir(base) if d.startswith("exp") and d[3:].isdigit()]
    exp_dir = os.path.join(base, f"exp{max(exps)+1 if exps else 1}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


# =====================================================
#  Main train()
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

    vec_env = make_vec_env(cfg.env_id, cfg.seed, n_envs)
    # Keep VecMonitor (episode stats) and optional VecNormalize for obs/returns
    vec_env = VecMonitor(vec_env)
    
    # --- PPO Model (PyTorch) ---
    policy_kwargs = dict(
        net_arch=[dict(pi=[cfg.policy_hidden, cfg.policy_hidden],
                       vf=[cfg.policy_hidden, cfg.policy_hidden])]
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

    # --- Callbacks: Video + Evaluation ---
    eval_env = make_vec_env(cfg.env_id, cfg.seed + 999, 1)  # single-env eval
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

    # Combine both
    callback_list = [video_cb, eval_cb]

    # --- Train ---
    total_timesteps = int(cfg.total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)

    # --- Save artifacts ---
    save_path = os.path.join(exp_dir, "ppo_policy.zip")
    model.save(save_path)
    logger.info(f"Saved model to {save_path}")

    # Close envs
    vec_env.close()


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
