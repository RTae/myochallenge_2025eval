import os
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from env_factory import build_worker_vec, build_manager_vec


ENV_ID = "myoChallengeTableTennisP2-v0"

LOGDIR = "./runs/hrl"
WORKER_DIR = os.path.join(LOGDIR, "worker")
MANAGER_DIR = os.path.join(LOGDIR, "manager")
os.makedirs(WORKER_DIR, exist_ok=True)
os.makedirs(MANAGER_DIR, exist_ok=True)


def load_worker_model(path: str):
    # If worker was trained with RecurrentPPO:
    return RecurrentPPO.load(path, device="cpu")
    # If worker was trained with PPO instead:
    # return PPO.load(path, device="cpu")


def main():
    # -------------------------
    # 1) Train Worker
    # -------------------------
    worker_env = build_worker_vec(env_id=ENV_ID, num_envs=8)

    worker_model = RecurrentPPO(
        "MlpLstmPolicy",
        worker_env,
        verbose=1,
        tensorboard_log=WORKER_DIR,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )

    eval_env = build_worker_vec(env_id=ENV_ID, num_envs=1)
    eval_env.training = False
    eval_env.norm_reward = False

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(WORKER_DIR, "best"),
        log_path=os.path.join(WORKER_DIR, "eval"),
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    worker_model.learn(
        total_timesteps=2_000_000,
        callback=CallbackList([eval_cb]),
    )

    worker_path = os.path.join(WORKER_DIR, "worker_model.zip")
    worker_model.save(worker_path)
    worker_env.save(os.path.join(WORKER_DIR, "vecnormalize.pkl"))

    worker_env.close()
    eval_env.close()

    # -------------------------
    # 2) Train Manager (frozen worker)
    # -------------------------
    # NOTE: manager env loads the worker model inside each subprocess
    manager_env = build_manager_vec(
        env_id=ENV_ID,
        num_envs=8,
        worker_model_path=worker_path,
        decision_interval=10,
        max_episode_steps=800,
        worker_model_loader=load_worker_model,
    )

    manager_model = PPO(
        "MlpPolicy",
        manager_env,
        verbose=1,
        tensorboard_log=MANAGER_DIR,
        n_steps=512,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.995,
    )

    # Eval manager
    eval_manager_env = build_manager_vec(
        env_id=ENV_ID,
        num_envs=1,
        worker_model_path=worker_path,
        decision_interval=10,
        max_episode_steps=800,
        worker_model_loader=load_worker_model,
    )

    eval_cb2 = EvalCallback(
        eval_manager_env,
        best_model_save_path=os.path.join(MANAGER_DIR, "best"),
        log_path=os.path.join(MANAGER_DIR, "eval"),
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    manager_model.learn(
        total_timesteps=2_000_000,
        callback=CallbackList([eval_cb2]),
    )

    manager_model.save(os.path.join(MANAGER_DIR, "manager_model.zip"))

    manager_env.close()
    eval_manager_env.close()


if __name__ == "__main__":
    main()
