# train_all.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from config import Config
from env_factory import build_vec_env


# -----------------------------
# Train Worker
# -----------------------------

def train_worker(cfg: Config):

    print("\n=== TRAINING WORKER (LOW-LEVEL PPO) ===\n")

    worker_logdir = os.path.join(cfg.logdir, "worker")
    os.makedirs(worker_logdir, exist_ok=True)

    # Build parallel VecEnv
    env = build_vec_env(worker=True, cfg=cfg)
    eval_env = build_vec_env(worker=True, cfg=cfg, eval_env=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=worker_logdir,
        log_path=worker_logdir,
        eval_freq=cfg.eval_freq,
        deterministic=True,
        n_eval_episodes=cfg.eval_episodes,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=worker_logdir,
        n_steps=cfg.ppo_n_steps,       # applied per-env inside VecEnv
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        learning_rate=cfg.ppo_lr,
        gae_lambda=cfg.ppo_lambda,
        clip_range=cfg.ppo_clip,
        n_epochs=cfg.ppo_epochs,
        seed=cfg.seed,
    )

    # Train worker
    model.learn(total_timesteps=cfg.total_timesteps, callback=eval_callback)

    model.save("worker.zip")
    env.save("worker_norm.pkl")      # IMPORTANT: save VecNormalize stats

    print("Worker saved to worker.zip + worker_norm.pkl")

    env.close()
    eval_env.close()


# -----------------------------
# Train Manager
# -----------------------------

def train_manager(cfg: Config):

    print("\n=== TRAINING MANAGER (HIGH-LEVEL PPO) ===\n")

    manager_logdir = os.path.join(cfg.logdir, "manager")
    os.makedirs(manager_logdir, exist_ok=True)

    # parallel vec env
    env = build_vec_env(worker=False, cfg=cfg)
    eval_env = build_vec_env(worker=False, cfg=cfg, eval_env=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=manager_logdir,
        log_path=manager_logdir,
        eval_freq=cfg.eval_freq,
        deterministic=True,
        n_eval_episodes=cfg.eval_episodes,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=manager_logdir,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        learning_rate=cfg.ppo_lr,
        gae_lambda=cfg.ppo_lambda,
        clip_range=cfg.ppo_clip,
        n_epochs=cfg.ppo_epochs,
        seed=cfg.seed,
    )

    model.learn(total_timesteps=cfg.total_timesteps, callback=eval_callback)

    model.save("manager.zip")
    env.save("manager_norm.pkl")

    print("Manager saved to manager.zip + manager_norm.pkl")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    cfg = Config()

    train_worker(cfg)
    train_manager(cfg)

    print("\n🎉 HRL Training Complete with Parallel Env + VecNormalize!")
