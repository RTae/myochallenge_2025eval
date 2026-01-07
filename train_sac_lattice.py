from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback
from stable_baselines3 import SAC
from lattice.sac.policies import LatticeSACPolicy
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
    model = SAC(policy=LatticeSACPolicy,
        env=env,
        device='auto',
        learning_rate=cfg.sac_lr,
        tensorboard_log=os.path.join(cfg.logdir, "manager"),
        batch_size=cfg.sac_batch_size,
        tau=cfg.sac_tau,
        gamma=cfg.sac_gamma,
        # action_noise=None,
        # replay_buffer_class=None,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        # use_sde=False,
        # sde_sample_freq=1,
        seed=cfg.seed+1,
        policy_kwargs=dict(
            use_lattice=True,
            use_expln=True,
            log_std_init=1.0,
            std_clip=(1e-3, 1),
            expln_eps=1e-6,
            clip_mean=None,
            std_reg=1e-4
        ),)
    
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
        progress_bar=True,
        callback=CallbackList([info_cb, eval_cb, video_cb]),
    )
    
    model_path = os.path.join(cfg.logdir, "model.pkl")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}, closing environments...")
    
    eval_env.close()
    env.close()

if __name__ == "__main__":
    main()
