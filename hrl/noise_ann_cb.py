import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger


class WorkerNoiseAnnealCallback(BaseCallback):
    """
    Curriculum controller for the worker.

    Responsibilities:
    1) Track EMA of SOFT success
    2) Compute progress ∈ [0,1]
    3) Anneal goal noise
    4) Unlock hard success at a fixed threshold
    """

    def __init__(
        self,
        worker_env,
        soft_start=0.30,        # when progress starts
        soft_end=0.60,          # when progress == 1
        unlock_hard_at=0.60,    # exact unlock point
        max_noise=0.10,
        check_freq=2_000,
        log_every_steps=1_000_000,
        ema_alpha=0.01,
        verbose=1,
    ):
        super().__init__(verbose)

        self.worker_env = worker_env

        self.soft_start = soft_start
        self.soft_end = soft_end
        self.unlock_hard_at = unlock_hard_at
        self.max_noise = max_noise

        self.check_freq = check_freq
        self.log_every_steps = log_every_steps

        self.success_ema = 0.0
        self.alpha = ema_alpha

        self.last_log_step = 0

    def _on_step(self) -> bool:
        # --------------------------------------------------
        # 1) Update EMA of *SOFT* success
        # --------------------------------------------------
        infos = self.locals.get("infos", [])
        if infos:
            soft_successes = [
                float(info.get("is_goal_success", 0.0))
                for info in infos
            ]
            mean_soft = float(np.mean(soft_successes))

            self.success_ema = (
                (1.0 - self.alpha) * self.success_ema
                + self.alpha * mean_soft
            )

        # --------------------------------------------------
        # 2) Update only every check_freq
        # --------------------------------------------------
        if self.n_calls % self.check_freq != 0:
            return True

        # --------------------------------------------------
        # 3) Compute progress ∈ [0,1]
        # --------------------------------------------------
        progress = np.clip(
            (self.success_ema - self.soft_start)
            / max(1e-6, self.soft_end - self.soft_start),
            0.0,
            1.0,
        )

        # --------------------------------------------------
        # 4) Noise schedule (gentle early)
        # --------------------------------------------------
        noise_scale = self.max_noise * (progress ** 2)

        # --------------------------------------------------
        # 5) Unlock hard success
        # --------------------------------------------------
        allow_hard = self.success_ema >= self.unlock_hard_at

        # --------------------------------------------------
        # 6) Apply to all worker envs (SB3-safe)
        # --------------------------------------------------
        #self.worker_env.env_method("set_goal_noise_scale", noise_scale)
        self.worker_env.env_method("set_progress", progress)
        
        self.logger.record("info/soft_success_ema", self.success_ema)
        self.logger.record("info/progress", progress)
        self.logger.record("info/goal_noise", noise_scale)
        self.logger.record("info/hard_enabled", float(allow_hard))

        # --------------------------------------------------
        # 7) Logging
        # --------------------------------------------------
        if (
            self.verbose > 0
            and self.n_calls - self.last_log_step >= self.log_every_steps
        ):
            self.last_log_step = self.n_calls
            logger.info(
                f"[WorkerCurriculum @ {self.n_calls:,} steps] "
                f"soft_success_ema={self.success_ema:.3f} | "
                f"progress={progress:.3f} | "
                f"noise={noise_scale:.3f} | "
                f"hard_enabled={allow_hard}"
            )

        return True