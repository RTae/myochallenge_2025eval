import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from config import Config
from env_factory import build_env
from utils import make_predict_fn, prepare_experiment_directory
from callbacks.video_callback import VideoCallback
from callbacks.infologger_callback import InfoLoggerCallback
from stable_baselines3.common.callbacks import EvalCallback


def main():
    cfg = Config()
    prepare_experiment_directory(cfg)

    env = build_env(cfg)
    eval_env = build_env(cfg)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=cfg.logdir,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        learning_rate=cfg.ppo_lr,
        gae_lambda=cfg.ppo_lambda,
        clip_range=cfg.ppo_clip,
        n_epochs=cfg.ppo_epochs,
        seed=cfg.seed,
    )

    video_cb = VideoCallback(cfg, predict_fn=make_predict_fn(model))
    info_cb = InfoLoggerCallback(prefix="train/info")
    eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=cfg.logdir,
                    log_path=os.path.join(cfg.logdir, "eval"),
                    eval_freq=cfg.eval_freq,
                    deterministic=True, render=False)

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList([video_cb, info_cb, eval_callback]),
    )
    

    model.save(os.path.join(cfg.logdir, "model"))
    env.close()


if __name__ == "__main__":
    main()
