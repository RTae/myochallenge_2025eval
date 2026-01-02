from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback
from stable_baselines3 import PPO
from config import Config
from loguru import logger
import os

from utils import prepare_experiment_directory, make_predict_fn
from env_factory import create_default_env
from custom_env import CustomEnv
from torch import nn
import math


def main():
    cfg = Config()
    prepare_experiment_directory(cfg)
    
    env = create_default_env(cfg, num_envs=cfg.num_envs)
    
    args = {
        "env": env,
        "verbose": 1,
        "tensorboard_log": os.path.join(cfg.logdir),
        "device": "cuda",
        "batch_size": 4096,
        "n_steps": 256,
        "n_epochs": 5,
        "learning_rate": lambda p: cfg.ppo_lr * 0.5 * (1 + math.cos(math.pi * (1 - p))),
        "clip_range": 0.2,
        "ent_coef": 3.62109e-06,
        "clip_range": cfg.ppo_clip_range,
        "gamma": cfg.ppo_gamma,
        "gae_lambda": cfg.ppo_lambda,
        "max_grad_norm": 0.3,
        "vf_coef": 0.835671,
        "clip_range_vf": cfg.ppo_clip_range,
        "use_sde": True,
        "sde_sample_freq": 1,
        "seed": cfg.seed,
        "policy_kwargs": dict(
            # ===== Pi & V Network Sizes =====
            net_arch=
                dict(
                    pi=[256, 256],
                    vf=[256, 256],
                ),
            activation_fn=nn.Tanh,  # smooth control
        ),
    }
    
    model = PPO(
        "MlpPolicy",
        **args
    )
    
    info_cb = InfoLoggerCallback(prefix="train/info")
    
    eval_env = create_default_env(cfg, num_envs=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg.logdir, "best_model"),
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=int(cfg.eval_freq//cfg.num_envs),
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )
    
    video_cb = VideoCallback(
        env_func=CustomEnv,
        env_args={"config": cfg},
        cfg=cfg,
        predict_fn=make_predict_fn(model)
    )
    
    model.learn(
        total_timesteps=cfg.ppo_total_timesteps,
        callback=CallbackList([info_cb, eval_cb, video_cb]),
    )
    
    model_path = os.path.join(cfg.logdir, "model.pkl")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}, closing environments...")
    
    eval_env.close()
    env.close()

if __name__ == "__main__":
    main()
