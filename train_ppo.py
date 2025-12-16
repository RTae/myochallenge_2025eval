from myosuite.utils import gym as myo_gym
import os
from config import Config

from loguru import logger

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from config import Config
from utils import prepare_experiment_directory, make_predict_fn
from custom_env import CustomEnv


def create_env(cfg: Config,  num_envs: int) -> VecNormalize:
    def make_env(rank: int):
        def _init():
            env = CustomEnv(cfg, device="cpu")
            return Monitor(env, info_keywords=("is_success",))
        return _init
    
    logger.info(f"Creating {num_envs} parallel Manager environments")
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    # Add normalization
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=False, 
        clip_obs=10.0,
        gamma=cfg.ppo_gamma
    )
    
    return env

def main():
    cfg = Config()
    prepare_experiment_directory(cfg)
    
    env = create_env

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(cfg.logdir, "manager"),
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        learning_rate=cfg.ppo_lr,
        gae_lambda=cfg.ppo_lambda,
        clip_range=cfg.ppo_clip,
        n_epochs=cfg.ppo_epochs,
        seed=cfg.seed,
    )
    
    info_cb = InfoLoggerCallback(prefix="train/info")
    
    eval_env = create_env(cfg, num_envs=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg.logdir, "best_model"),
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )
    
    video_env = CustomEnv(cfg)
    video_cb = VideoCallback(video_env, cfg, make_predict_fn(model))
    
    model.learn(
        total_timesteps=cfg.worker_total_timesteps,
        callback=CallbackList([info_cb, eval_cb, video_cb]),
    )

if __name__ == "__main__":
    main()
