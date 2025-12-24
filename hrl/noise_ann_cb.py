import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger


class WorkerNoiseAnnealCallback(BaseCallback):
    """
    Anneal goal noise for the worker based on EMA goal success.

    Noise is applied ONLY to the goal (not physics),
    and is controlled via worker.set_goal_noise_scale().
    """

    def __init__(
        self,
        worker_env,
        start_success=0.60,
        end_success=0.85,
        max_noise=0.10,
        check_freq=2_000,
        log_every_steps=1_000_000,
        verbose=1,
    ):
        super().__init__(verbose)

        self.worker_env = worker_env

        self.start_success = start_success
        self.end_success = end_success
        self.max_noise = max_noise

        self.check_freq = check_freq
        self.log_every_steps = log_every_steps

        # EMA of goal success
        self.success_ema = 0.0
        self.alpha = 0.01

        self.last_log_step = 0

    def _on_step(self) -> bool:
        # --------------------------------------------------
        # 1) Update EMA success
        # --------------------------------------------------
        infos = self.locals.get("infos", [])
        if infos:
            success = infos[0].get("is_goal_success", False)
            self.success_ema = (
                (1.0 - self.alpha) * self.success_ema
                + self.alpha * float(success)
            )

        # --------------------------------------------------
        # 2) Only update noise occasionally
        # --------------------------------------------------
        if self.n_calls % self.check_freq != 0:
            return True

        # --------------------------------------------------
        # 3) Compute progress based on success
        # --------------------------------------------------
        progress = np.clip(
            (self.success_ema - self.start_success)
            / (self.end_success - self.start_success),
            0.0,
            1.0,
        )

        # Quadratic schedule = gentler early
        noise_scale = self.max_noise * (progress ** 2)

        # --------------------------------------------------
        # 4) Apply noise to worker
        # --------------------------------------------------
        worker = self.worker_env.envs[0]
        worker.set_goal_noise_scale(noise_scale)

        # --------------------------------------------------
        # 5) Periodic logging
        # --------------------------------------------------
        if (
            self.verbose > 0
            and self.n_calls - self.last_log_step >= self.log_every_steps
        ):
            self.last_log_step = self.n_calls
            logger.info(
                f"[WorkerNoiseAnneal @ {self.n_calls:,} steps] "
                f"success_ema={self.success_ema:.3f} | "
                f"goal_noise_scale={noise_scale:.4f}"
            )

        return True