# train_all.py
import os
import copy
import numpy as np
from loguru import logger

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from config import Config
from env_factory import build_vec_env
from hrl_utils import flatten_myo_obs_worker, make_hierarchical_predictor
from callbacks.video_callback import VideoCallback


# ============================================================
# Create experiment directory logs/expN/
# ============================================================

def prepare_experiment_directory(cfg: Config):
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
# TRAIN WORKER (LOW-LEVEL CONTROLLER)
# ============================================================

def train_worker(cfg: Config):

    logger.info("🚀 Training Worker (low-level controller)")
    worker_logdir = os.path.join(cfg.logdir, "worker")
    os.makedirs(worker_logdir, exist_ok=True)

    # Parallel environments
    env = build_vec_env(worker=True, cfg=cfg)
    eval_env = build_vec_env(worker=True, cfg=cfg, eval_env=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=worker_logdir,
        log_path=worker_logdir,
        eval_freq=cfg.eval_freq,
        deterministic=True,
        n_eval_episodes=cfg.eval_episodes,
    )

    # PPO Model
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

    # Video callback uses a simple predictor
    def worker_predict(_ignored_sb3_obs, env_instance):
        obs_dict = env_instance.unwrapped.obs_dict
        obs_vec = flatten_myo_obs_worker(obs_dict).reshape(1, -1)
        action, _ = model.predict(obs_vec, deterministic=True)
        return action.reshape(-1)

    video_cb = VideoCallback(cfg, mode="worker", predict_fn=worker_predict)

    # Train
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList([eval_callback, video_cb]),
    )

    # Save
    model.save(os.path.join(worker_logdir, "worker.zip"))
    env.save(os.path.join(worker_logdir, "worker_norm.pkl"))

    logger.info("💾 Worker saved → worker.zip + worker_norm.pkl")

    env.close()
    eval_env.close()


# ============================================================
# TRAIN MANAGER (HIGH-LEVEL CONTROLLER)
# ============================================================

def train_manager(cfg: Config):

    logger.info("🚀 Training Manager (high-level controller)")
    manager_logdir = os.path.join(cfg.logdir, "manager")
    os.makedirs(manager_logdir, exist_ok=True)

    # Parallel environments
    env = build_vec_env(worker=False, cfg=cfg)
    eval_env = build_vec_env(worker=False, cfg=cfg, eval_env=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=manager_logdir,
        log_path=manager_logdir,
        eval_freq=cfg.eval_freq,
        deterministic=True,
        n_eval_episodes=cfg.eval_episodes,
    )

    # PPO Model
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

    # Load trained worker
    from stable_baselines3 import PPO as PPO_LOAD
    worker_path = os.path.join(cfg.logdir, "worker", "worker.zip")
    worker_model = PPO_LOAD.load(worker_path)

    # Build HRL predictor for video recording
    hrl_predict = make_hierarchical_predictor(cfg, model, worker_model)

    video_cb = VideoCallback(cfg, mode="manager", predict_fn=hrl_predict)

    # Train
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList([eval_callback, video_cb]),
    )

    # Save
    model.save(os.path.join(manager_logdir, "manager.zip"))
    env.save(os.path.join(manager_logdir, "manager_norm.pkl"))

    logger.info("💾 Manager saved → manager.zip + manager_norm.pkl")

    env.close()
    eval_env.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    base_cfg = Config()
    prepare_experiment_directory(base_cfg)

    # ------------------------------
    # WORKER CONFIG
    # ------------------------------
    worker_cfg = copy.deepcopy(base_cfg)
    worker_cfg.high_level_period = 1
    worker_cfg.total_timesteps = 100_000
    worker_cfg.ppo_lr = 1e-4

    train_worker(worker_cfg)

    # ------------------------------
    # MANAGER CONFIG
    # ------------------------------
    manager_cfg = copy.deepcopy(base_cfg)
    manager_cfg.high_level_period = 15
    manager_cfg.total_timesteps = 100_000
    manager_cfg.ppo_lr = 3e-4

    train_manager(manager_cfg)

    logger.success("🎉 HRL Training Complete!")
