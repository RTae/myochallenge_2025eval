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
    horizon_H: int = 8              # fast + stable
    pop_size: int = 48              # fits 8–12 workers well
    es_sigma: float = 0.03          # safe; no instability
    mppi_lambda: float = 5.0        # smoother weights
    cem_workers: int = 12           # for 28-core CPU
    w_track: float = 0.10           # DO NOT make too small
    w_task: float = 1.00
    plan_internal: int = 10         # less replanning → faster

    # === Video Callback ===
    video_freq: int = 10000
    eval_episodes: int = 2
    video_w: int = 640
    video_h: int = 480
    camera_id: int = 0

    # === Evaluation Callback ===
    eval_freq: int = 5000
