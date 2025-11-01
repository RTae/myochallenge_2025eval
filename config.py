from dataclasses import dataclass

@dataclass
class Config:
    env_id: str = "myoChallengeTableTennisP2-v0"
    total_timesteps: int = 10_000_000
    seed: int = 42
    n_envs: int = 4
    ppo_lr: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_clip: float = 0.2
    ppo_update_epochs: int = 4
    ppo_batch_size: int = 256
    ppo_lambda: float = 0.95
    policy_hidden: int = 256
    value_hidden: int = 256
    norm_clip: float = 10.0
    skills: int = 4
    es_batch: int = 8
    es_sigma: float = 0.1
    es_alpha: float = 0.02
    horizon_H: int = 20
    logdir: str = "./logs/exp_temp"
    video_freq: int = 10_000
    eval_episodes: int = 2
    max_eval_frames: int = 300
    video_w: int = 640
    video_h: int = 480
    camera_id: int = 0