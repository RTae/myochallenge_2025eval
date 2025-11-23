from dataclasses import dataclass

@dataclass
class Config:
    # === Environment ===
    env_id: str = "myoChallengeTableTennisP2-v0"
    seed: int = 42

    total_timesteps: int = 1_000_000
    logdir: str = "./logs"
    train_log_freq: int = 1000


    # === MPPI Parameters ===
    horizon_H: int = 14             # bigger horizon = better TT control
    pop_size: int = 64             # total MPPI samples per iteration
    es_sigma: float = 0.06         # exploration noise
    mppi_lambda: float = 4.0       # smoother weighting, much more stable
    cem_workers: int = 22           # parallel workers
    es_batch:int = 8               # number of samples per batch
    w_track = 0.02
    w_task = 1.0
    plan_internal:int = 5

    # === Video Callback ===
    video_freq: int = 10_000
    eval_episodes: int = 2
    video_w: int = 640
    video_h: int = 480
    camera_id: int = 0

    # === Evaluation Callback ===
    eval_freq: int = 5_000
