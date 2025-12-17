import copy
import os
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from lattice.ppo.policies import LatticeRecurrentActorCriticPolicy

from config import Config
from env_factory import build_env
from utils import prepare_experiment_directory, make_predict_fn
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback
from callbacks.curriculum_callback import CurriculumCallback

from curriculum_env import CurriculumEnv
from dr_spcrl import DRSPCRLRecurrentPPO


def main():
    cfg = Config()
    prepare_experiment_directory(cfg)

    # # Train worker
    cfg = copy.deepcopy(cfg)
    cfg.logdir = os.path.join(cfg.logdir)
    env = build_env(cfg, env_type="curriculum", eval_mode=False)

    model = DRSPCRLRecurrentPPO(
        policy=LatticeRecurrentActorCriticPolicy,
        env=env,
        tensorboard_log=cfg.logdir,
        verbose=1,
        device="auto",
        batch_size=cfg.ppo_batch_size,
        n_steps=cfg.ppo_n_steps,
        n_epochs=5,
        learning_rate=5e-5,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_lambda,
        max_grad_norm=0.5,
        clip_range = 0.2,
        ent_coef = 3e-6,
        vf_coef=0.25,
        clip_range_vf=0.2,
        policy_kwargs=dict(
            use_lattice=True,
            use_expln=True,
            ortho_init=False,
            log_std_init=0.0,
            std_clip=(1e-3, 10),
        ),
        target_beta=5,
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
    eval_env = build_env(eval_cfg, env_type="curriculum", eval_mode=True)
    eval_env.obs_rms = env.obs_rms
    eval_env.training = False
    eval_env.norm_reward = False
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg.logdir, "best_model"),
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )
    video_env = CurriculumEnv(cfg, eval_mode=True)
    video_cb = VideoCallback(video_env, cfg, make_predict_fn(model))
    # curriculum_cb = CurriculumCallback(
    #     cfg,
    #     total_steps=cfg.worker_total_timesteps,
    #     freeze_patience=5,
    #     freeze_threshold=0.05,
    #     verbose=1,
    # )

    model.learn(
        total_timesteps=cfg.worker_total_timesteps,
        callback=CallbackList([info_cb, eval_cb, video_cb]),
    )

    model_path = os.path.join(cfg.logdir, "model.pkl")
    model.save(model_path)
    
    eval_env.close()
    video_env.close()
    env.close()
    

if __name__ == "__main__":
    main()
