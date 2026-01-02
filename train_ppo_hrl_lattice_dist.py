import os
from typing import Optional

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from lattice.ppo.policies import LatticeActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor

from config import Config
from env_factory import build_worker_vec
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback
from hrl.worker_env import TableTennisWorker
from hrl.manager_env import TableTennisManager
from hrl.noise_ann_cb import WorkerNoiseAnnealCallback
from utils import make_predict_fn, prepare_experiment_directory, resume_vecnormalize_on_training_env
from loguru import logger
from torch import nn
import math


# ==================================================
# Worker loaders
# ==================================================
def load_worker_model(path: str):
    return RecurrentPPO.load(
            path,
            device="cuda",
            policy=LatticeActorCriticPolicy, 
    )

def load_worker_on_specific_gpu(path: str, device_id: int):
    """
    Loads the worker model onto a specific GPU.
    Used by the Manager environments to shard memory usage.
    """
    return RecurrentPPO.load(path, device=f"cuda:{device_id}", policy=LatticeActorCriticPolicy)

def load_worker_vecnormalize(path: str, venv: TableTennisWorker) -> VecNormalize:
    """
    Keep your existing interface for manager/video usage:
    load VecNormalize stats onto a fresh env built by env_fn.
    """
    env = DummyVecEnv([lambda: venv])
    vecnorm = VecNormalize.load(path, env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm

def build_sharded_manager_vec(cfg: Config, num_envs: int, worker_model_path: str, worker_env_path: str, num_gpus: int = 4):
    """
    Spawns 'num_envs' processes distributed across 'num_gpus'.
    """
    def make_env(rank: int):
        def _init():
            import torch
            # Prevent threads from fighting over CPU resources
            torch.set_num_threads(1)
            
            # 1. Assign this environment to a specific GPU based on rank
            gpu_id = rank // (num_envs // num_gpus)
            gpu_id = min(gpu_id, num_gpus - 1)
            
            # 2. Local Loaders for this specific GPU
            def specific_model_loader(p):
                return load_worker_on_specific_gpu(p, gpu_id)
            
            def specific_env_loader(p):
                return load_worker_vecnormalize(p, TableTennisWorker(cfg))

            worker_vec = specific_env_loader(worker_env_path),
            worker_model = specific_model_loader(worker_model_path),

            env = TableTennisManager(
                worker_env=worker_vec,
                worker_model=worker_model,
                config=cfg,
                decision_interval=5,
                max_episode_steps=cfg.episode_len,
            )
            return env
        return _init

    env_fns = [make_env(i) for i in range(num_envs)]
    
    logger.info(f"Spawning {num_envs} Manager environments sharded across {num_gpus} GPUs...")
    venv = SubprocVecEnv(env_fns)
    venv = VecMonitor(venv, info_keywords=("is_success",))
    
    return VecNormalize(venv, norm_obs=False, norm_reward=False)

def main():
    cfg = Config()
    prepare_experiment_directory(cfg)

    worker_total_timesteps = 20_000_000
    manager_total_timesteps = 4_000_000

    # ==================================================
    # LOAD paths
    # ==================================================
    LOAD_WORKER_MODEL_PATH: Optional[str] = None
    LOAD_WORKER_ENV_PATH: Optional[str] = None

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
        "batch_size": 2048,
        "n_steps": 256,
        "n_epochs": 5,

        # ---------------------------
        # Scheduler
        # ---------------------------
        "learning_rate": lambda p: cfg.ppo_lr * 0.5 * (1 + math.cos(math.pi * (1 - p))),
        "clip_range": lambda p: cfg.ppo_clip_range * p,

        # ---------------------------
        # PPO Hyperparameters
        # ---------------------------
        "ent_coef": 3.62109e-06,
        "clip_range": cfg.ppo_clip_range,
        "gamma": cfg.ppo_gamma,
        "gae_lambda": cfg.ppo_lambda,
        "max_grad_norm": 0.3,
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
            ortho_init=True,
            
            log_std_init=-2.0,
            std_clip=(0.01, 1.0),
            expln_eps=1e-6,
            std_reg=0.0,

            # ===== Pi & V Network Sizes =====
            net_arch=
                dict(
                    pi=[256, 256],
                    vf=[256, 256],
                ),
            activation_fn=nn.Tanh,  # smooth control
            lstm_hidden_size=128,
        ),
    }
    
    # Log hyperparameters
    logger.info("#"*100)
    logger.info("Worker hyperparameters:")
    logger.info("#"*100)
    for k, v in worker_args.items():
        logger.info(f"{k} = {v}")

    if worker_resumed:
        logger.info(f"[Worker] Loading pretrained model from: {LOAD_WORKER_MODEL_PATH}")
        worker_model = RecurrentPPO.load(
            LOAD_WORKER_MODEL_PATH,
            policy=LatticeActorCriticPolicy, 
            **worker_args
        )
    else:
        logger.info("[Worker] No pretrained worker model given/found. Training from scratch.")        
        worker_model = RecurrentPPO(
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
    
    ann_worker_cb = WorkerNoiseAnnealCallback(
        worker_env=worker_env,
        check_freq=50_000 // cfg.num_envs,
        log_every_steps=1_000_000,
        verbose=1,
    )

    logger.info("Starting WORKER training...")
    logger.info(f"Worker total timesteps: {worker_total_timesteps}")

    worker_model.learn(
        total_timesteps=worker_total_timesteps,
        reset_num_timesteps=not worker_resumed,  # continue curves if resumed
        callback=CallbackList([eval_worker_cb, info_cb, video_worker_cb, ann_worker_cb]),
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
        
    # Ensure Worker Exists
    if not os.path.exists(SAVE_WORKER_MODEL_PATH):
        logger.error(f"Worker model not found at {SAVE_WORKER_MODEL_PATH}")
        return

    # 1. Build Multi-GPU Sharded Environment
    # 160 Envs / 4 GPUs = 40 Envs per GPU
    MANAGER_NUM_ENVS = 160 
    
    manager_env = build_sharded_manager_vec(
        cfg=cfg,
        num_envs=MANAGER_NUM_ENVS,
        worker_model_path=SAVE_WORKER_MODEL_PATH,
        worker_env_path=SAVE_WORKER_ENV_PATH,
        num_gpus=4
    )
    
    # 2. Manager Hyperparameters
    manager_args = {
        "device": "cpu",
        "verbose": 1,
        "tensorboard_log": cfg.logdir,
        
        # Optimized for 160 Envs
        # Buffer = 160 * 256 = 40960
        "n_steps": 256,
        "batch_size": 4096,
        
        "learning_rate": lambda p: 3e-4 * 0.5 * (1 + math.cos(math.pi * (1 - p))),
        "clip_range": lambda p: 0.2 * p,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "max_grad_norm": 0.5,
        "n_epochs": 10,
        "policy_kwargs": dict(net_arch=[256, 256]),
        "seed": cfg.seed,
    }
    
    logger.info("#"*100)
    logger.info(f"Manager training on {MANAGER_NUM_ENVS} envs across 4 GPUs")
    logger.info("#"*100)

    # 3. Load or Create Manager
    manager_model = PPO("MlpPolicy", manager_env, **manager_args)

    # 4. Evaluation (Single Env on GPU 0)
    eval_manager_env = build_sharded_manager_vec(
        cfg, num_envs=1, 
        worker_model_path=SAVE_WORKER_MODEL_PATH, 
        worker_env_path=SAVE_WORKER_ENV_PATH, 
        num_gpus=1
    )

    eval_manager_cb = EvalCallback(
        eval_manager_env,
        best_model_save_path=os.path.join(cfg.logdir, "best"),
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # 5. Video Callback
    # Load a dedicated worker on GPU 0 for video generation
    video_worker_vec = load_worker_vecnormalize(SAVE_WORKER_ENV_PATH, TableTennisWorker(cfg))
    video_worker_model = load_worker_on_specific_gpu(SAVE_WORKER_MODEL_PATH, 0)
    
    video_manager_cb = VideoCallback(
        env_func=TableTennisManager,
        env_args={
            "worker_env": video_worker_vec,
            "worker_model": video_worker_model,
            "config": cfg,
            "decision_interval": 1,
            "max_episode_steps": cfg.episode_len,
        },
        cfg=cfg,
        predict_fn=make_predict_fn(manager_model),
    )

    # 6. Train
    logger.info("Starting MANAGER training...")
    manager_model.learn(
        total_timesteps=manager_total_timesteps,
        callback=CallbackList([eval_manager_cb, info_cb, video_manager_cb]),
    )

    manager_model.save(SAVE_MANAGER_MODEL_PATH)
    logger.info(f"Saved MANAGER model to: {SAVE_MANAGER_MODEL_PATH}")

    manager_env.close()
    eval_manager_env.close()

if __name__ == "__main__":
    main()