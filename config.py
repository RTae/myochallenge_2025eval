# config.py
import os
from dataclasses import dataclass


def getenv(name, default, cast_fn=str):
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return cast_fn(val)
    except Exception:
        return default


@dataclass
class Config:
    # ==================================================
    # Environment
    # ==================================================
    env_id: str = getenv("ENV_ID", "myoChallengeTableTennisP2-v0", str)
    seed: int = getenv("SEED", 42, int)

    # IMPORTANT: keep this small early
    num_envs: int = getenv("NUM_ENVS", 24, int)

    # Episode length (~3 seconds)
    episode_len: int = getenv("EPISODE_LEN", 300, int)

    # ==================================================
    # Training
    # ==================================================
    worker_total_timesteps: int = getenv("WORKER_TOTAL_TIMESTEPS", 10_000_000, int)
    manager_total_timesteps: int = getenv("MANAGER_TOTAL_TIMESTEPS", 5_000_000, int)
    logdir: str = getenv("LOGDIR", "./logs", str)

    # ==================================================
    # PPO (FAST FEEDBACK)
    # ==================================================
    ppo_n_steps: int = getenv("PPO_N_STEPS", 128, int)
    ppo_batch_size: int = getenv("PPO_BATCH_SIZE", 32, int)
    ppo_epochs: int = getenv("PPO_EPOCHS", 5, int)

    ppo_lr: float = getenv("PPO_LR", 2.5e-5, float)

    ppo_gamma: float = getenv("PPO_GAMMA", 0.99, float)
    ppo_lambda: float = getenv("PPO_LAMBDA", 0.95, float)
    ppo_clip: float = getenv("PPO_CLIP", 0.2, float)

    # ==================================================
    # Evaluation & Video
    # ==================================================
    eval_freq: int = getenv("EVAL_FREQ", 50_000, int)
    eval_episodes: int = getenv("EVAL_EPISODES", 3, int)

    video_freq: int = getenv("VIDEO_FREQ", 1_000_000, int)
    video_w: int = getenv("VIDEO_W", 640, int)
    video_h: int = getenv("VIDEO_H", 480, int)
    camera_id: int = getenv("CAMERA_ID", 1, int)
    video_frames: int = getenv("VIDEO_FRAMES", 300, int)

    # ==================================================
    # Physics constants (from MyoChallenge spec)
    # ==================================================
    BALL_MASS = 0.0027
    BALL_RADIUS = 0.02
    PADDLE_MASS = 0.150
    PADDLE_FACE_RADIUS = 0.093
    PADDLE_HANDLE_RADIUS = 0.016
    TABLE_HALF_WIDTH = 1.37
    NET_HEIGHT = 0.305
