import copy
import os
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from lattice.ppo.policies import LatticeRecurrentActorCriticPolicy

from config import Config
from env_factory import build_env
from utils import prepare_experiment_directory, make_predict_fn
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback

from custom_env import CustomEnv
from dr_spcrl import DRSPCRLRecurrentPPO


def main():
    cfg = Config()
    prepare_experiment_directory(cfg)

    # # Train worker
    cfg = copy.deepcopy(cfg)
    cfg.logdir = os.path.join(cfg.logdir)
    env = build_env(cfg, env_type="worker")

    worker_model = DRSPCRLRecurrentPPO(
        policy=LatticeRecurrentActorCriticPolicy,
        env=env,
        tensorboard_log=cfg.logdir,
        verbose=1,
        device="auto",
        batch_size=cfg.ppo_batch_size,
        n_steps=cfg.ppo_n_steps,
        n_epochs=cfg.ppo_epochs,
        learning_rate=cfg.ppo_lr,
        clip_range=cfg.ppo_clip,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_lambda,
        ent_coef=3.62109e-06,
        max_grad_norm=0.7,
        vf_coef=0.835671,
        policy_kwargs=dict(
            use_lattice=True,
            use_expln=True,
            ortho_init=False,
            log_std_init=0.0,
            std_clip=(1e-3, 10),
        ),
        eps_start=0.0,
        eps_budget=1.0,
        lr_beta=5e-4,
        beta_updates=5,
        lr_curr=1e-3,
        alpha=0.1,
    )

    # # Callback Share
    info_cb = InfoLoggerCallback(prefix="train/info")
    eval_cfg = Config()
    eval_cfg.num_envs = 1
    
    # Callback Worker
    eval_env = build_env(eval_cfg, env_type="worker")
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
    video_cb = VideoCallback(video_env, cfg, make_predict_fn(worker_model))

    worker_model.learn(
        total_timesteps=cfg.worker_total_timesteps,
        callback=CallbackList([info_cb, eval_cb, video_cb]),
    )

    model_path = os.path.join(cfg.logdir, "model.pkl")
    worker_model.save(model_path)
    eval_env.close()
    video_env.close()
    env.close()
    

if __name__ == "__main__":
    main()
