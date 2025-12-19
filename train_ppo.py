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

def main():
    cfg = Config()
    prepare_experiment_directory(cfg)
    
    env = create_default_env(cfg, num_envs=cfg.num_envs)
    model = PPO(
        "MlpPolicy",
        env,
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
    
    info_cb = InfoLoggerCallback(prefix="train/info")
    
    eval_env = create_default_env(cfg, num_envs=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg.logdir, "best_model"),
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=cfg.eval_freq//cfg.num_envs,
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
