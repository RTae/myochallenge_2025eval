from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from lattice.ppo.policies import LatticeRecurrentActorCriticPolicy
from config import Config
from loguru import logger
import os

from utils import prepare_experiment_directory, make_predict_fn
from custom_env import CustomEnv


def create_env(cfg: Config, num_envs: int) -> VecNormalize:    
    def make_env(rank: int):
        def _init():
            # Create plain environment
            env = CustomEnv(cfg)
            return Monitor(env, info_keywords=("is_success",))
        return _init
    
    logger.info(f"Creating {num_envs} parallel plain environments")
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    # Add normalization
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=cfg.ppo_gamma,
    )
    
    return env

def main():
    cfg = Config()
    prepare_experiment_directory(cfg)
            
    env = create_env(cfg, num_envs=cfg.num_envs)
    model = RecurrentPPO(policy=LatticeRecurrentActorCriticPolicy, 
        env=env,
        verbose=1,
        tensorboard_log=os.path.join(cfg.logdir),
        device='auto',
        batch_size=32,
        n_steps=128,
        learning_rate=cfg.ppo_lr,
        ent_coef=3.62109e-06,
        clip_range=cfg.ppo_clip_range,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_lambda,
        max_grad_norm=cfg.ppo_max_grad_norm,
        vf_coef=0.835671,
        n_epochs=cfg.ppo_epochs,
        use_sde=False,
        sde_sample_freq=1,
        clip_range_vf=cfg.ppo_clip_range,
        seed=cfg.seed,
        policy_kwargs=dict(
            use_lattice=True,
            use_expln=True,
            ortho_init=False,
            log_std_init=0.0,
            std_clip=(1e-3, 10),
            expln_eps=1e-6,
            full_std=False,
            std_reg=0.0,
        )
    )
    
    
    info_cb = InfoLoggerCallback(prefix="train/info")
    
    eval_env = create_env(cfg, num_envs=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg.logdir, "best_model"),
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=cfg.eval_freq//cfg.num_envs,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )
    
    video_cb = VideoCallback(CustomEnv, cfg, make_predict_fn(model))
    
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
