import os
from dataclasses import dataclass


def getenv(name, default, cast_fn=str):
    """
    Helper to read environment variables with type casting.
    Example:
        NUM_ENVS=16 python train_all.py
    """
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return cast_fn(val)
    except Exception:
        return default


@dataclass
class Config:
    # === Environment ===
    env_id: str = getenv("ENV_ID", "myoChallengeTableTennisP2-v0", str)
    seed: int = getenv("SEED", 42, int)
    num_envs: int = getenv("NUM_ENVS", 24, int)

    # === HRL ===
    high_level_period: int = getenv("HL_PERIOD", 15, int)
    goal_dim: int = getenv("GOAL_DIM", 3, int)
    goal_std: float = getenv("GOAL_STD", 0.10, float)
    goal_bound: float = getenv("GOAL_BOUND", 0.25, float)
    worker_episode_len = 1000

    # === Training ===
    total_timesteps: int = getenv("TOTAL_TIMESTEPS", 10_000_000, int)
    logdir: str = getenv("LOGDIR", "./logs", str)
    train_log_freq: int = getenv("TRAIN_LOG_FREQ", 2000, int)

    # === PPO ===
    ppo_n_steps: int = getenv("PPO_N_STEPS", 2048, int)
    ppo_batch_size: int = getenv("PPO_BATCH_SIZE", 4096, int)
    ppo_gamma: float = getenv("PPO_GAMMA", 0.99, float)
    ppo_lambda: float = getenv("PPO_LAMBDA", 0.95, float)
    ppo_lr: float = getenv("PPO_LR", 1e-4, float)
    ppo_epochs: int = getenv("PPO_EPOCHS", 5, int)
    ppo_clip: float = getenv("PPO_CLIP", 0.2, float)

    # === Video ===
    video_freq: int = getenv("VIDEO_FREQ", 1_000_000, int)
    eval_episodes: int = getenv("EVAL_EPISODES", 3, int)
    video_w: int = getenv("VIDEO_W", 640, int)
    video_h: int = getenv("VIDEO_H", 480, int)
    camera_id: int = getenv("CAMERA_ID", 1, int)
    video_frames: int = getenv("VIDEO_FRAMES", 300, int)

    # === Eval ===
    eval_freq: int = getenv("EVAL_FREQ", 500_000, int)
