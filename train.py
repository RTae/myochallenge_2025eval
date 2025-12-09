# train_all.py
import os
import copy
import numpy as np

from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from config import Config
from env_factory import build_vec_env
from callbacks.video_callback import VideoCallback
from hrl_utils import build_worker_obs, flatten_myo_obs_worker, make_hierarchical_predictor
from stable_baselines3 import PPO as PPO_LOAD


# ============================================================
# Create experiment directory logs/expN/
# ============================================================
def prepare_experiment_directory(cfg: Config):
    """
    Creates logs/exp1, exp2, ... automatically.
    Updates cfg.logdir to the new experiment folder.
    """
    base = cfg.logdir
    os.makedirs(base, exist_ok=True)

    existing = [d for d in os.listdir(base) if d.startswith("exp")]
    exp_nums = []
    for e in existing:
        try:
            exp_nums.append(int(e.replace("exp", "")))
        except Exception:
            pass

    next_id = 1 if not exp_nums else max(exp_nums) + 1
    exp_dir = os.path.join(base, f"exp{next_id}")
    os.makedirs(exp_dir, exist_ok=True)

    cfg.logdir = exp_dir
    logger.info(f"📁 Created new experiment folder: {exp_dir}")
    return exp_dir


# ============================================================
# TRAIN WORKER
# ============================================================
def train_worker(cfg: Config):
    logger.info("🦵 Training worker (goal-conditioned low-level)...")

    worker_logdir = os.path.join(cfg.logdir, "worker")
    os.makedirs(worker_logdir, exist_ok=True)

    env = build_vec_env(worker=True, cfg=cfg)
    eval_env = build_vec_env(worker=True, cfg=cfg)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=worker_logdir,
        log_path=worker_logdir,
        eval_freq=cfg.eval_freq,
        deterministic=True,
        n_eval_episodes=cfg.eval_episodes,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=worker_logdir,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        learning_rate=cfg.ppo_lr,
        gae_lambda=cfg.ppo_lambda,
        clip_range=cfg.ppo_clip,
        n_epochs=cfg.ppo_epochs,
        seed=cfg.seed,
    )

    # --- Video callback (worker only) ---
    def worker_predict(_ignored_sb3_obs, env_instance):
        obs_dict = env_instance.unwrapped.obs_dict

        # Use zero goal + phase=0 just to visualize
        zero_goal = np.zeros(cfg.goal_dim, dtype=np.float32)
        worker_obs = build_worker_obs(
            obs_dict=obs_dict,
            goal=zero_goal,
            t_in_macro=0,
            cfg=cfg,
        ).reshape(1, -1)

        action, _ = model.predict(worker_obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    video_cb = VideoCallback(cfg, mode="worker", predict_fn=worker_predict)

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList([eval_callback, video_cb]),
    )

    worker_path = os.path.join(worker_logdir, "worker.zip")
    model.save(worker_path)
    logger.info(f"💾 Saved worker model → {worker_path}")

    env.close()
    eval_env.close()


# ============================================================
# TRAIN MANAGER
# ============================================================
def train_manager(cfg: Config):
    logger.info("🧠 Training manager (high-level HRL)...")

    manager_logdir = os.path.join(cfg.logdir, "manager")
    os.makedirs(manager_logdir, exist_ok=True)

    worker_model_path = os.path.join(cfg.logdir, "worker", "worker.zip")
    worker_model_path = os.path.abspath(worker_model_path)

    if not os.path.exists(worker_model_path):
        raise FileNotFoundError(f"Expected worker model at: {worker_model_path}")

    env = build_vec_env(worker=False, cfg=cfg, worker_model_path=worker_model_path)
    eval_env = build_vec_env(worker=False, cfg=cfg, worker_model_path=worker_model_path)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=manager_logdir,
        log_path=manager_logdir,
        eval_freq=cfg.eval_freq,
        deterministic=True,
        n_eval_episodes=cfg.eval_episodes,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=manager_logdir,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        learning_rate=cfg.ppo_lr,
        gae_lambda=cfg.ppo_lambda,
        clip_range=cfg.ppo_clip,
        n_epochs=cfg.ppo_epochs,
        seed=cfg.seed,
    )

    # HRL predictor for visualization (manager + worker on raw env)
    worker_model = PPO_LOAD.load(worker_model_path)
    hrl_predict = make_hierarchical_predictor(cfg, model, worker_model)

    video_cb = VideoCallback(cfg, mode="manager", predict_fn=hrl_predict)

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList([eval_callback, video_cb]),
    )

    manager_path = os.path.join(manager_logdir, "manager.zip")
    model.save(manager_path)
    logger.info(f"💾 Saved manager model → {manager_path}")

    env.close()
    eval_env.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    base_cfg = Config()
    prepare_experiment_directory(base_cfg)

    # You can override these per-stage if you want different budgets
    worker_cfg = copy.deepcopy(base_cfg)
    manager_cfg = copy.deepcopy(base_cfg)

    # Example: different timesteps / lrs
    worker_cfg.total_timesteps = 1_000_000
    worker_cfg.ppo_lr = 1e-4
    
    manager_cfg.total_timesteps = 1_000_000
    manager_cfg.ppo_lr = 3e-4

    train_worker(worker_cfg)
    train_manager(manager_cfg)

    logger.info("🎉 HRL Training Complete!")
