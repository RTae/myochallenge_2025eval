import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from config import Config
from env_factory import build_worker_vec, build_manager_vec
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback
from hrl.worker_env import TableTennisWorker
from hrl.manager_env import TableTennisManager
from utils import prepare_experiment_directory, make_predict_fn


# ==================================================
# Worker loader
# ==================================================
def load_worker_model(path: str):
    return PPO.load(path, device="cpu")


def main():
    # ==================================================
    # Config & directories
    # ==================================================
    cfg = Config()
    prepare_experiment_directory(cfg)

    env_id = cfg.env_id

    WORKER_DIR = os.path.join(cfg.logdir, "worker")
    MANAGER_DIR = os.path.join(cfg.logdir, "manager")
    os.makedirs(WORKER_DIR, exist_ok=True)
    os.makedirs(MANAGER_DIR, exist_ok=True)

    info_cb = InfoLoggerCallback()

    # ==================================================
    # 1) Train WORKER
    # ==================================================
    worker_env = build_worker_vec(
        env_id=env_id,
        num_envs=cfg.num_envs,
    )

    worker_model = PPO(
        "MlpPolicy",
        worker_env,
        verbose=1,
        tensorboard_log=WORKER_DIR,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        learning_rate=cfg.ppo_lr,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_lambda,
        clip_range=cfg.ppo_clip_range,
        n_epochs=cfg.ppo_epochs,
        max_grad_norm=cfg.ppo_max_grad_norm,
        policy_kwargs=dict(net_arch=[cfg.ppo_hidden_dim]),
        seed=cfg.seed,
    )

    # ---- Worker evaluation ----
    eval_worker_env = build_worker_vec(env_id=env_id, num_envs=1)
    eval_worker_env.training = False
    eval_worker_env.norm_reward = False

    eval_worker_cb = EvalCallback(
        eval_worker_env,
        best_model_save_path=os.path.join(WORKER_DIR, "best"),
        log_path=os.path.join(WORKER_DIR, "eval"),
        eval_freq=int(cfg.eval_freq // cfg.num_envs),
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )

    # ---- Worker video ----
    video_worker_cb = VideoCallback(
        env_func=TableTennisWorker,
        env_args={"config": cfg},
        cfg=cfg,
        predict_fn=make_predict_fn(worker_model),
    )

    worker_model.learn(
        total_timesteps=cfg.worker_total_timesteps,
        callback=CallbackList([
            eval_worker_cb,
            info_cb,
            video_worker_cb,
        ]),
    )

    # ---- Save WORKER (.pkl) ----
    worker_model_path = os.path.join(WORKER_DIR, "worker_model.pkl")
    worker_model.save(worker_model_path)
    worker_env.save(os.path.join(WORKER_DIR, "vecnormalize.pkl"))

    worker_env.close()
    eval_worker_env.close()

    # ==================================================
    # 2) Train MANAGER (frozen worker)
    # ==================================================
    manager_env = build_manager_vec(
        env_id=env_id,
        num_envs=cfg.num_envs,
        worker_model_path=worker_model_path,
        decision_interval=cfg.episode_len // 10,
        max_episode_steps=cfg.episode_len,
        worker_model_loader=load_worker_model,
    )

    manager_model = PPO(
        "MlpPolicy",
        manager_env,
        verbose=1,
        tensorboard_log=MANAGER_DIR,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        learning_rate=cfg.ppo_lr,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_lambda,
        clip_range=cfg.ppo_clip_range,
        n_epochs=cfg.ppo_epochs,
        max_grad_norm=cfg.ppo_max_grad_norm,
        policy_kwargs=dict(net_arch=[cfg.ppo_hidden_dim]),
        seed=cfg.seed,
    )

    # ---- Manager evaluation ----
    eval_manager_env = build_manager_vec(
        env_id=env_id,
        num_envs=1,
        worker_model_path=worker_model_path,
        decision_interval=cfg.episode_len // 10,
        max_episode_steps=cfg.episode_len,
        worker_model_loader=load_worker_model,
    )

    eval_manager_cb = EvalCallback(
        eval_manager_env,
        best_model_save_path=os.path.join(MANAGER_DIR, "best"),
        log_path=os.path.join(MANAGER_DIR, "eval"),
        eval_freq=int(cfg.eval_freq // cfg.num_envs),
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )

    # ---- Manager video ----
    video_worker_env = TableTennisWorker(cfg)
    frozen_worker_model = load_worker_model(worker_model_path)

    video_manager_cb = VideoCallback(
        env_func=TableTennisManager,
        env_args={
            "worker_env": video_worker_env,
            "worker_model": frozen_worker_model,
            "config": cfg,
        },
        cfg=cfg,
        predict_fn=make_predict_fn(manager_model),
    )

    manager_model.learn(
        total_timesteps=cfg.manager_total_timesteps,
        callback=CallbackList([
            eval_manager_cb,
            info_cb,
            video_manager_cb,
        ]),
    )

    # ---- Save MANAGER (.pkl) ----
    manager_model.save(os.path.join(MANAGER_DIR, "manager_model.pkl"))

    manager_env.close()
    eval_manager_env.close()
    video_worker_env.close()


if __name__ == "__main__":
    main()
