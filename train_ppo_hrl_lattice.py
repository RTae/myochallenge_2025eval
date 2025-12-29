import os
from typing import Callable, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize
from lattice.ppo.policies import LatticeActorCriticPolicy

from config import Config
from env_factory import build_manager_vec, build_worker_vec
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback
from hrl.worker_env import TableTennisWorker
from hrl.manager_env import TableTennisManager
from utils import make_predict_fn, prepare_experiment_directory, resume_vecnormalize_on_training_env
from loguru import logger
from torch import nn
import math


# ==================================================
# Worker loaders
# ==================================================
def load_worker_model(path: str):
    return PPO.load(
            path,
            device="cuda",
            policy=LatticeActorCriticPolicy, 
    )


def load_worker_vecnormalize(path: str, venv: TableTennisWorker) -> VecNormalize:
    """
    Keep your existing interface for manager/video usage:
    load VecNormalize stats onto a fresh env built by env_fn.
    """
    vecnorm = VecNormalize.load(path, venv)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm

def main():
    cfg = Config()
    prepare_experiment_directory(cfg)

    worker_total_timesteps = 15_000_000
    manager_total_timesteps = 2_000_000

    # ==================================================
    # LOAD paths
    # ==================================================
    LOAD_WORKER_MODEL_PATH: Optional[str] = None
    LOAD_WORKER_ENV_PATH: Optional[str] = None
    LOAD_MANAGER_MODEL_PATH: Optional[str] = None

    # ==================================================
    # SAVE dirs/paths (always write to your experiment logdir)
    # ==================================================
    WORKER_DIR = os.path.join(cfg.logdir, "worker")
    MANAGER_DIR = os.path.join(cfg.logdir, "manager")
    os.makedirs(WORKER_DIR, exist_ok=True)
    os.makedirs(MANAGER_DIR, exist_ok=True)

    SAVE_WORKER_MODEL_PATH = os.path.join(WORKER_DIR, "worker_model.pkl")
    SAVE_WORKER_ENV_PATH = os.path.join(WORKER_DIR, "vecnormalize.pkl")
    SAVE_MANAGER_MODEL_PATH = os.path.join(MANAGER_DIR, "manager_model.pkl")

    info_cb = InfoLoggerCallback()

    # ==================================================
    # WORKER
    # ==================================================
    cfg.logdir = WORKER_DIR

    # ---- Build training env exactly like your original ----
    worker_env = build_worker_vec(cfg=cfg, num_envs=cfg.num_envs)

    # ---- If LOAD_WORKER_ENV_PATH exists, load VecNormalize stats onto worker_env ----
    worker_env = resume_vecnormalize_on_training_env(
        worker_env,
        LOAD_WORKER_ENV_PATH,
        training=True,
        norm_reward=False,
    )

    # ---- Build model: load if provided else create new ----
    worker_resumed = bool(LOAD_WORKER_MODEL_PATH and os.path.exists(LOAD_WORKER_MODEL_PATH))
    
    worker_args = {
        # ---------------------------
        # Env + VecNormalize
        # ---------------------------
        "env": worker_env,                # this should be a VecNormalize-wrapped env
        "verbose": 1,
        "tensorboard_log": os.path.join(cfg.logdir),
        "device": "cuda",

        # ---------------------------
        # PPO Batch & Rollout Settings
        # ---------------------------
        "batch_size": 32,
        "n_steps": 128,
        "n_epochs": cfg.ppo_epochs,

        # ---------------------------
        # Learning Rate Scheduler
        # ---------------------------
        "learning_rate": lambda progress: cfg.ppo_lr * 0.5 * (1 + math.cos(math.pi * progress)),

        # ---------------------------
        # PPO Hyperparameters
        # ---------------------------
        "ent_coef": 3.62109e-06,
        "clip_range": cfg.ppo_clip_range,
        "gamma": cfg.ppo_gamma,
        "gae_lambda": cfg.ppo_lambda,
        "max_grad_norm": cfg.ppo_max_grad_norm,
        "vf_coef": 0.835671,
        "clip_range_vf": cfg.ppo_clip_range,

        # ---------------------------
        # SDE Exploration
        # ---------------------------
        "use_sde": True,
        "sde_sample_freq": 1,

        # ---------------------------
        # Reproducibility
        # ---------------------------
        "seed": cfg.seed,

        # ---------------------------
        # Policy Network Architecture
        # ---------------------------
        "policy_kwargs": dict(
            # ===== Lattice Noise Settings =====
            use_lattice=True,
            use_expln=True,
            full_std=False,
            ortho_init=False,
            log_std_init=-0.5,
            std_clip=(1e-3, 10),
            expln_eps=1e-6,
            std_reg=0.0,

            # ===== Pi & V Network Sizes =====
            net_arch=[
                dict(
                    pi=[256, 256],   # policy layers
                    vf=[256, 256],   # value layers
                )
            ],
            activation_fn=nn.Tanh,  # smooth control
        ),
    }

    if worker_resumed:
        logger.info(f"[Worker] Loading pretrained model from: {LOAD_WORKER_MODEL_PATH}")
        worker_model = PPO.load(
            LOAD_WORKER_MODEL_PATH,
            policy=LatticeActorCriticPolicy, 
            **worker_args
        )
    else:
        logger.info("[Worker] No pretrained worker model given/found. Training from scratch.")        
        worker_model = PPO(
            policy=LatticeActorCriticPolicy,
            **worker_args
        )

    # ---- Worker evaluation ----
    eval_worker_env = build_worker_vec(cfg=cfg, num_envs=1)

    if LOAD_WORKER_ENV_PATH:
        eval_worker_env = resume_vecnormalize_on_training_env(
            eval_worker_env,
            LOAD_WORKER_ENV_PATH,
            training=False,
            norm_reward=False,
        )

    if isinstance(eval_worker_env, VecNormalize):
        eval_worker_env.training = False
        eval_worker_env.norm_reward = False

    eval_worker_cb = EvalCallback(
        eval_worker_env,
        best_model_save_path=os.path.join(cfg.logdir, "best"),
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=int(cfg.eval_freq // cfg.num_envs),
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )

    # ---- Worker video ---
    video_worker_cb = VideoCallback(
        env_func=TableTennisWorker,
        env_args={"config": cfg},
        cfg=cfg,
        predict_fn=make_predict_fn(worker_model),
    )

    logger.info("Starting WORKER training...")
    logger.info(f"Worker total timesteps: {worker_total_timesteps}")

    worker_model.learn(
        total_timesteps=worker_total_timesteps,
        reset_num_timesteps=not worker_resumed,  # continue curves if resumed
        callback=CallbackList([eval_worker_cb, info_cb, video_worker_cb]),
    )

    # ---- Save worker to SAVE paths  ----
    worker_model.save(SAVE_WORKER_MODEL_PATH)
    if isinstance(worker_env, VecNormalize):
        worker_env.save(SAVE_WORKER_ENV_PATH)
    else:
        logger.warning("[Worker] Training env is not VecNormalize; skipping vecnormalize save.")

    logger.info(f"Saved WORKER model to: {SAVE_WORKER_MODEL_PATH}")
    logger.info(f"Saved WORKER VecNormalize to: {SAVE_WORKER_ENV_PATH}")

    worker_env.close()
    eval_worker_env.close()

    # ==================================================
    # MANAGER
    # ==================================================
    cfg.logdir = MANAGER_DIR
    
    if LOAD_MANAGER_MODEL_PATH:
        assert os.path.exists(SAVE_WORKER_MODEL_PATH)
        assert os.path.exists(SAVE_WORKER_ENV_PATH)

    def worker_env_loader(path: str):
        # Manager loads a frozen worker env via VecNormalize + fresh env_fn
        return load_worker_vecnormalize(path, TableTennisWorker(cfg))

    # Manager should use the worker produced by this run
    manager_env = build_manager_vec(
        cfg=cfg,
        num_envs=cfg.num_envs,
        worker_model_loader=load_worker_model,
        worker_env_loader=worker_env_loader,
        worker_model_path=SAVE_WORKER_MODEL_PATH,
        worker_env_path=SAVE_WORKER_ENV_PATH,
        decision_interval=5,
        max_episode_steps=cfg.episode_len,
    )

    manager_resumed = bool(LOAD_MANAGER_MODEL_PATH and os.path.exists(LOAD_MANAGER_MODEL_PATH))
    
    manager_args = {
        "device":"cpu",
        "verbose":1,
        "tensorboard_log":cfg.logdir,
        "n_steps":128,
        "batch_size":256,
        "learning_rate":1e-4,
        "gamma":0.995,
        "gae_lambda":0.97,
        "clip_range":cfg.ppo_clip_range,
        "n_epochs":cfg.ppo_epochs,
        "max_grad_norm":cfg.ppo_max_grad_norm,
        "policy_kwargs":dict(net_arch=[256, 256]),
        "seed":cfg.seed,
    }

    if manager_resumed:
        logger.info(f"[Manager] Loading pretrained model from: {LOAD_MANAGER_MODEL_PATH}")
        manager_model = PPO.load(
            LOAD_MANAGER_MODEL_PATH,
            env=manager_env,
            **manager_args
        )
    else:
        logger.info("[Manager] No pretrained manager model given/found. Training from scratch.")
        manager_model = PPO(
            "MlpPolicy",
            manager_env,
            **manager_args
        )

    # ---- Manager evaluation ----
    eval_manager_env = build_manager_vec(
        cfg=cfg,
        num_envs=1,
        worker_model_loader=load_worker_model,
        worker_env_loader=worker_env_loader,
        worker_model_path=SAVE_WORKER_MODEL_PATH,
        worker_env_path=SAVE_WORKER_ENV_PATH,
        decision_interval=5,
        max_episode_steps=cfg.episode_len,
    )

    eval_manager_cb = EvalCallback(
        eval_manager_env,
        best_model_save_path=os.path.join(cfg.logdir, "best"),
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=int(cfg.eval_freq // cfg.num_envs),
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )

    # ---- Manager video (unchanged pattern) ----
    video_worker_env = worker_env_loader(SAVE_WORKER_ENV_PATH)
    frozen_worker_model = load_worker_model(SAVE_WORKER_MODEL_PATH)

    video_manager_cb = VideoCallback(
        env_func=TableTennisManager,
        env_args={
            "worker_env": video_worker_env,
            "worker_model": frozen_worker_model,
            "config": cfg,
            "decision_interval": 1,
            "max_episode_steps": cfg.episode_len,
        },
        cfg=cfg,
        predict_fn=make_predict_fn(manager_model),
    )

    logger.info("Starting MANAGER training...")
    logger.info(f"Manager total timesteps: {manager_total_timesteps}")

    manager_model.learn(
        total_timesteps=manager_total_timesteps,
        reset_num_timesteps=not manager_resumed,
        callback=CallbackList([eval_manager_cb, info_cb, video_manager_cb]),
    )

    # ---- Save manager to SAVE paths ----
    manager_model.save(SAVE_MANAGER_MODEL_PATH)
    logger.info(f"Saved MANAGER model to: {SAVE_MANAGER_MODEL_PATH}")

    # ---- Close ----
    manager_env.close()
    eval_manager_env.close()
    video_worker_env.close()


if __name__ == "__main__":
    main()