from dataclasses import dataclass

@dataclass
class Config:
    # === Environment ===
    env_id: str = "myoChallengeTableTennisP2-v0"
    seed: int = 42

    total_timesteps: int = 1_000_000 
    logdir: str = "./logs"
    train_log_freq: int = 1000

    es_sigma: float = 0.1
    elites: int = 6

    es_batch: int = 4   
    horizon_H: int = 5
    cem_workers: int = 6
    mppi_lambda: float = 1.0

    # === Video Callback ===
    video_freq: int = 10_000
    eval_episodes: int = 2
    video_w: int = 640
    video_h: int = 480
    camera_id: int = 0

    # === Evaluation Callback ===
    eval_freq: int = 5_000
