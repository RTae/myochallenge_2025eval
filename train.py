import os
from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CallbackList
from sb3_contrib import RecurrentPPO
from lattice.ppo.policies import LatticeRecurrentActorCriticPolicy

from config import Config
from env_factory import build_env
from utils import prepare_experiment_directory
from callbacks.infologger_callback import InfoLoggerCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecVideoRecorder


def main():
    cfg = Config()
    prepare_experiment_directory(cfg)

    env = build_env(cfg)

    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     device='auto',
    #     tensorboard_log=cfg.logdir,
    #     n_steps=cfg.ppo_n_steps,
    #     batch_size=cfg.ppo_batch_size,
    #     gamma=cfg.ppo_gamma,
    #     learning_rate=cfg.ppo_lr,
    #     gae_lambda=cfg.ppo_lambda,
    #     clip_range=cfg.ppo_clip,
    #     n_epochs=cfg.ppo_epochs,
    #     seed=cfg.seed,
    # )
    
    # --- Lattice PPO ---
    model = RecurrentPPO(
        policy=LatticeRecurrentActorCriticPolicy, 
        env=env,
        tensorboard_log=cfg.logdir,
        verbose=1,
        device='auto',
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
        use_sde=False,
        sde_sample_freq=1,
        policy_kwargs=dict(
            use_lattice=True,
            use_expln=True,
            ortho_init=False,
            log_std_init=0.0,
            # activation_fn=nn.ReLU,
            std_clip=(1e-3, 10),
            expln_eps=1e-6,
            full_std=False,
            std_reg=0.0,
        ),
    )

    info_cb = InfoLoggerCallback(prefix="train/info")
    
    # ---- eval env (single env only) ----
    eval_cfg = Config()
    eval_cfg.num_envs = 1
    eval_env = build_env(eval_cfg)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.logdir,
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )
    
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder= os.path.join(cfg.logdir, "videos"),
        record_video_trigger=lambda step: step % cfg.video_freq == 0,
        video_length=cfg.video_frames,
        name_prefix="eval"
    )

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList([eval_env, info_cb, eval_callback]),
    )
    

    model.save(os.path.join(cfg.logdir, "model"))
    env.close()


if __name__ == "__main__":
    main()
