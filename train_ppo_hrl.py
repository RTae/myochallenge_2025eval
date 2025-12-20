import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from config import Config
from env_factory import build_worker_vec, build_manager_vec
from callbacks.infologger_callback import InfoLoggerCallback
from utils import prepare_experiment_directory


def load_worker_model(path: str):
    """
    Worker is trained with normal PPO.
    Must be loaded INSIDE each SubprocVecEnv subprocess.
    """
    return PPO.load(path, device="cpu")


def main():
    # ==================================================
    # Config & directories
    # ==================================================
    cfg = Config()
    prepare_experiment_directory(cfg)

    env_id = cfg.env_id  # ✅ use config, not hard-coded

    WORKER_DIR = os.path.join(cfg.logdir, "worker")
    MANAGER_DIR = os.path.join(cfg.logdir, "manager")
    os.makedirs(WORKER_DIR, exist_ok=True)
    os.makedirs(MANAGER_DIR, exist_ok=True)
    
    # Share Callback
    info_cb = InfoLoggerCallback()

    # ==================================================
    # 1) Train Worker
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
    eval_worker_env = build_worker_vec(
        env_id=env_id,
        num_envs=1,
    )
    eval_worker_env.training = False
    eval_worker_env.norm_reward = False

    eval_cb = EvalCallback(
        eval_worker_env,
        best_model_save_path=os.path.join(WORKER_DIR, "best"),
        log_path=os.path.join(WORKER_DIR, "eval"),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )

    worker_model.learn(
        total_timesteps=cfg.worker_total_timesteps,
        callback=CallbackList([eval_cb, info_cb]),
    )

    worker_model_path = os.path.join(WORKER_DIR, "worker_model.zip")
    worker_model.save(worker_model_path)
    worker_env.save(os.path.join(WORKER_DIR, "vecnormalize.pkl"))

    worker_env.close()
    eval_worker_env.close()

    # ==================================================
    # 2) Train Manager
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

    eval_cb2 = EvalCallback(
        eval_manager_env,
        best_model_save_path=os.path.join(MANAGER_DIR, "best"),
        log_path=os.path.join(MANAGER_DIR, "eval"),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )

    manager_model.learn(
        total_timesteps=cfg.manager_total_timesteps,
        callback=CallbackList([eval_cb2, info_cb]),
    )

    manager_model.save(os.path.join(MANAGER_DIR, "manager_model.zip"))

    manager_env.close()
    eval_manager_env.close()


if __name__ == "__main__":
    main()
