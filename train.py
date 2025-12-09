# train_all.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from config import Config
from worker_env import WorkerEnv
from manager_env import ManagerEnv


# -----------------------------
# Helper functions
# -----------------------------

def make_worker_env(cfg: Config):
    """Factory for DummyVecEnv."""
    def _init():
        env = WorkerEnv(cfg)
        return Monitor(env)  # enables logging
    return _init


def make_manager_env(cfg: Config):
    """Factory for DummyVecEnv."""
    def _init():
        env = ManagerEnv(cfg, worker_model_path="worker.zip")
        return Monitor(env)
    return _init


# -----------------------------
# Training Worker (low-level)
# -----------------------------

def train_worker(cfg: Config):
    print("\n========================")
    print(" TRAINING WORKER (LOW-LEVEL PPO)")
    print("========================\n")

    # Logging directory
    worker_logdir = os.path.join(cfg.logdir, "worker")
    os.makedirs(worker_logdir, exist_ok=True)

    # Environment
    env = DummyVecEnv([make_worker_env(cfg)])

    # Eval environment
    eval_env = DummyVecEnv([make_worker_env(cfg)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=worker_logdir,
        log_path=worker_logdir,
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
    )

    # PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=worker_logdir,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_lambda,
        learning_rate=cfg.ppo_lr,
        clip_range=cfg.ppo_clip,
        n_epochs=cfg.ppo_epochs,
        seed=cfg.seed,
    )

    # Train
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=eval_callback,
    )

    # Save
    model.save("worker.zip")
    print("Worker saved → worker.zip")

    env.close()
    eval_env.close()


# -----------------------------
# Training Manager (high-level)
# -----------------------------

def train_manager(cfg: Config):
    print("\n========================")
    print(" TRAINING MANAGER (HIGH-LEVEL PPO)")
    print("========================\n")

    # Logging directory
    manager_logdir = os.path.join(cfg.logdir, "manager")
    os.makedirs(manager_logdir, exist_ok=True)

    # Env (manager)
    env = DummyVecEnv([make_manager_env(cfg)])

    # Eval env
    eval_env = DummyVecEnv([make_manager_env(cfg)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=manager_logdir,
        log_path=manager_logdir,
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
    )

    # PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=manager_logdir,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_lambda,
        learning_rate=cfg.ppo_lr,
        clip_range=cfg.ppo_clip,
        n_epochs=cfg.ppo_epochs,
        seed=cfg.seed,
    )

    # Train
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=eval_callback,
    )

    # Save
    model.save("manager.zip")
    print("Manager saved → manager.zip")

    env.close()
    eval_env.close()


# -----------------------------
# MAIN — Train Worker then Manager
# -----------------------------

if __name__ == "__main__":
    cfg = Config()

    print("\n====================================")
    print("        HIERARCHICAL TRAINING")
    print("====================================")

    # 1) Train low-level controller first
    train_worker(cfg)

    # 2) Train high-level manager (with frozen worker)
    train_manager(cfg)

    print("\n🎉 HRL training complete!")
    print("worker.zip and manager.zip are ready.")
