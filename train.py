import copy
import os
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from lattice.ppo.policies import LatticeRecurrentActorCriticPolicy

from config import Config
from env_factory import build_env
from utils import prepare_experiment_directory, make_predict_fn
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback
from worker_env import TableTennisWorker


def main():
    cfg = Config()
    #prepare_experiment_directory(cfg)
    cfg.logdir = "./logs/exp15"

    # # Train worker
    worker_cfg = copy.deepcopy(cfg)
    # worker_cfg.logdir = os.path.join(cfg.logdir, "worker")
    # env_worker = build_env(worker_cfg, worker=True)

    # worker_model = RecurrentPPO(
    #     policy=LatticeRecurrentActorCriticPolicy,
    #     env=env_worker,
    #     tensorboard_log=worker_cfg.logdir,
    #     verbose=1,
    #     device="auto",
    #     batch_size=worker_cfg.ppo_batch_size,
    #     n_steps=worker_cfg.ppo_n_steps,
    #     n_epochs=worker_cfg.ppo_epochs,
    #     learning_rate=worker_cfg.ppo_lr,
    #     clip_range=worker_cfg.ppo_clip,
    #     gamma=worker_cfg.ppo_gamma,
    #     gae_lambda=worker_cfg.ppo_lambda,
    #     ent_coef=3.62109e-06,
    #     max_grad_norm=0.7,
    #     vf_coef=0.835671,
    #     policy_kwargs=dict(
    #         use_lattice=True,
    #         use_expln=True,
    #         ortho_init=False,
    #         log_std_init=0.0,
    #         std_clip=(1e-3, 10),
    #     ),
    # )

    # # Callback Share
    info_cb = InfoLoggerCallback(prefix="train/info")
    eval_cfg = Config()
    eval_cfg.num_envs = 1
    
    # # Callback Worker
    # eval_worker_env = build_env(eval_cfg, worker=True)
    # eval_cb = EvalCallback(
    #     eval_worker_env,
    #     best_model_save_path=os.path.join(worker_cfg.logdir, "best_model"),
    #     log_path=os.path.join(worker_cfg.logdir, "eval"),
    #     eval_freq=worker_cfg.eval_freq,
    #     n_eval_episodes=worker_cfg.eval_episodes,
    #     deterministic=True,
    #     render=False,
    # )
    # video_cb = VideoCallback(worker_cfg, make_predict_fn(worker_model))

    # # Learn
    # worker_model.learn(
    #     total_timesteps=worker_cfg.total_timesteps,
    #     callback=CallbackList([info_cb, eval_cb, video_cb]),
    # )

    worker_model_path = os.path.join(worker_cfg.logdir, "model.pkl")
    # worker_model.save(worker_model_path)
    # eval_worker_env.close()
    # env_worker.close()
    
    # Train Manager
    
    manager_cfg = copy.deepcopy(cfg)
    
    manager_cfg.total_timesteps = 10_000
    manager_cfg.ppo_lr = 3e-4
    manager_cfg.ppo_gamma = 0.995
    manager_cfg.ppo_n_steps = 512
    manager_cfg.ppo_batch_size = 256
    manager_cfg.logdir = os.path.join(cfg.logdir, "manager")
    
    worker_model = RecurrentPPO.load(
        worker_model_path,
    )
    
    env_manager = build_env(
        manager_cfg, 
        worker=False,
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
    
    eval_manager_env = build_env(eval_cfg, worker=False, worker_model=worker_model)
    eval_cb = EvalCallback(
        eval_manager_env,
        best_model_save_path=os.path.join(manager_cfg.logdir, "best_model"),
        log_path=os.path.join(manager_cfg.logdir, "eval"),
        eval_freq=manager_cfg.eval_freq,
        n_eval_episodes=manager_cfg.eval_episodes,
        deterministic=True,
        render=False,
    )
    video_cb = VideoCallback(manager_cfg, make_predict_fn(manager_model))
    
    # Learn
    manager_model.learn(
        total_timesteps=manager_cfg.total_timesteps,
        callback=CallbackList([info_cb, eval_cb, video_cb]),
    )

    manager_model.save(os.path.join(manager_cfg.logdir, "model.pkl"))
    eval_manager_env.close()
    env_manager.close()

if __name__ == "__main__":
    main()
