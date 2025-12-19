import copy
import os
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3 import PPO

from config import Config
from env_factory import build_env
from utils import prepare_experiment_directory, make_predict_fn
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback

from hrl.worker_env import TableTennisWorker
from hrl.manager_env import TableTennisManager
from dr_spcrl.dr_spcrl import DRSPCRLRecurrentPPO
from loguru import logger


def main():
    cfg = Config()
    prepare_experiment_directory(cfg)

    # # Train worker
    worker_cfg = copy.deepcopy(cfg)
    worker_cfg.logdir = os.path.join(cfg.logdir, "worker")
    env_worker = build_env(worker_cfg, env_type="worker")

    worker_model = PPO(
        "MlpPolicy",
        env_worker,
        verbose=1,
        tensorboard_log=os.path.join(cfg.logdir),
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_n_steps//cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        learning_rate=cfg.ppo_lr,
        gae_lambda=cfg.ppo_lambda,
        n_epochs=cfg.ppo_epochs,
        max_grad_norm=cfg.ppo_max_grad_norm,
        clip_range=cfg.ppo_clip_range,
        seed=cfg.seed,
        clip_range_vf=cfg.ppo_clip_range,
        policy_kwargs=dict(net_arch=[cfg.ppo_hidden_dim]),
    )

    # # Callback Share
    info_cb = InfoLoggerCallback(prefix="train/info")
    eval_cfg = Config()
    eval_cfg.num_envs = 1
    
    # Callback Worker
    eval_worker_env = build_env(eval_cfg, env_type="worker")
    eval_cb = EvalCallback(
        eval_worker_env,
        best_model_save_path=os.path.join(worker_cfg.logdir, "best_model"),
        log_path=os.path.join(worker_cfg.logdir, "eval"),
        eval_freq=worker_cfg.eval_freq,
        n_eval_episodes=worker_cfg.eval_episodes,
        deterministic=True,
        render=False,
    )
    video_cb = VideoCallback(TableTennisWorker, worker_cfg, make_predict_fn(worker_model))

    # Learn
    logger.info("🚧 Training Worker Policy 🚧")
    worker_model.learn(
        total_timesteps=worker_cfg.worker_total_timesteps,
        callback=CallbackList([info_cb, eval_cb, video_cb]),
    )

    worker_model_path = os.path.join(worker_cfg.logdir, "model.pkl")
    worker_model.save(worker_model_path)
    eval_worker_env.close()
    video_worker_env.close()
    env_worker.close()
    
    # Train Manager
    manager_cfg = copy.deepcopy(cfg)
    
    manager_cfg.ppo_lr = 3e-4
    manager_cfg.ppo_gamma = 0.995
    manager_cfg.ppo_n_steps = 512
    manager_cfg.ppo_batch_size = 256
    manager_cfg.logdir = os.path.join(cfg.logdir, "manager")
    
    worker_model = DRSPCRLRecurrentPPO.load(
        worker_model_path,
    )
    
    env_manager = build_env(
        manager_cfg, 
        env_type="manager",
        worker_model=worker_model
    )

    manager_model = PPO(
        "MlpPolicy",
        env_manager,
        verbose=1,
        tensorboard_log=os.path.join(cfg.logdir, "manager"),
        n_steps=manager_cfg.ppo_n_steps,
        batch_size=manager_cfg.ppo_batch_size,
        gamma=manager_cfg.ppo_gamma,
        learning_rate=manager_cfg.ppo_lr,
        gae_lambda=manager_cfg.ppo_lambda,
        clip_range=manager_cfg.ppo_clip,
        n_epochs=manager_cfg.ppo_epochs,
        seed=manager_cfg.seed,
    )
    
    eval_manager_env = build_env(eval_cfg, env_type="manager", worker_model=worker_model)
    eval_cb = EvalCallback(
        eval_manager_env,
        best_model_save_path=os.path.join(manager_cfg.logdir, "best_model"),
        log_path=os.path.join(manager_cfg.logdir, "eval"),
        eval_freq=manager_cfg.eval_freq,
        n_eval_episodes=manager_cfg.eval_episodes,
        deterministic=True,
        render=False,
    )
    video_worker_env = TableTennisWorker(worker_cfg)
    video_cb = VideoCallback(
        env_func=TableTennisManager,
        env_args={
            "worker_env": video_worker_env,
            "worker_model": worker_model,
            "config": manager_cfg,
        },
        cfg=manager_cfg,
        predict_fn=make_predict_fn(manager_model)
    )
        
    # Learn
    logger.info("🚧 Training Manager Policy 🚧")
    manager_model.learn(
        total_timesteps=manager_cfg.manager_total_timesteps,
        callback=CallbackList([info_cb, eval_cb, video_cb]),
    )

    manager_model.save(os.path.join(manager_cfg.logdir, "model.pkl"))
    
    eval_manager_env.close()
    video_worker_env.close()
    env_manager.close()
    
    logger.info("🏁 Training Complete 🏁")

if __name__ == "__main__":
    main()
