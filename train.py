import os
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from sb3_contrib import RecurrentPPO
from lattice.ppo.policies import LatticeRecurrentActorCriticPolicy

from config import Config
from env_factory import build_env
from utils import prepare_experiment_directory, make_predict_fn
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback


def main():
    cfg = Config()
    prepare_experiment_directory(cfg)

    # ========================
    # Training env (NO render)
    # ========================
    env = build_env(cfg)

    model = RecurrentPPO(
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
    )

    info_cb = InfoLoggerCallback(prefix="train/info")

    # ========================
    # Eval callback (NO video)
    # ========================
    eval_cfg = Config()
    eval_cfg.num_envs = 1
    eval_env = build_env(eval_cfg)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=cfg.logdir,
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )

    # ========================
    # Video callback (manual)
    # ========================
    video_cb = VideoCallback(cfg, make_predict_fn(model))

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList([info_cb, eval_cb, video_cb]),
    )

    model.save(os.path.join(cfg.logdir, "model"))
    env.close()


if __name__ == "__main__":
    main()
