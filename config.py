# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # === Environment ===
    env_id: str = "myoChallengeTableTennisP2-v0"
    seed: int = 42

    # === HRL ===
    high_level_period: int = 10       # steps per manager decision
    goal_dim: int = 3                 # dx,dy,dz offset
    goal_std: float = 0.15            # worker goal sampling
    goal_bound: float = 0.30          # manager goal action bound

    # === Training ===
    total_timesteps: int = 1_000_000
    logdir: str = "./logs"
    train_log_freq: int = 1000

    # === PPO Hyperparameters ===
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 256
    ppo_gamma: float = 0.99
    ppo_lambda: float = 0.95
    ppo_lr: float = 3e-4
    ppo_epochs: int = 10
    ppo_clip: float = 0.2

    # === Video Callback ===
    video_freq: int = 10000
    eval_episodes: int = 2
    video_w: int = 640
    video_h: int = 480
    camera_id: int = 0

    # === Eval Callback ===
    eval_freq: int = 5000
