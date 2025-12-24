import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger

class WorkerNoiseAnnealCallback(BaseCallback):
    def __init__(
        self,
        worker_env,
        start_success=0.6,
        end_success=0.85,
        check_freq=2000,
        log_every_steps=1_000_000,
        verbose=1,
    ):
        super().__init__(verbose)
        self.worker_env = worker_env
        self.start_success = start_success
        self.end_success = end_success
        self.check_freq = check_freq
        self.log_every_steps = log_every_steps

        self.success_ema = 0.0
        self.alpha = 0.01
        self.last_log_step = 0

        self.pos_start = np.array([0.03, 0.03, 0.02])
        self.pos_end   = np.array([0.08, 0.08, 0.04])

        self.normal_start = 0.06
        self.normal_end   = 0.15

        self.time_start = 0.04
        self.time_end   = 0.10

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            success = infos[0].get("worker/is_goal_success", False)
            self.success_ema = (
                (1 - self.alpha) * self.success_ema
                + self.alpha * float(success)
            )

        if self.n_calls % self.check_freq != 0:
            return True

        progress = np.clip(
            (self.success_ema - self.start_success)
            / (self.end_success - self.start_success),
            0.0,
            1.0,
        )

        def lerp(a, b, t):
            return a + t * (b - a)

        worker = self.worker_env.envs[0]

        worker.pos_noise_scale = lerp(self.pos_start, self.pos_end, progress)
        worker.normal_noise_scale = lerp(self.normal_start, self.normal_end, progress)
        worker.time_noise_scale = lerp(self.time_start, self.time_end, progress ** 2)

        if (
            self.verbose > 0
            and self.n_calls - self.last_log_step >= self.log_every_steps
        ):
            self.last_log_step = self.n_calls
            logger.info(
                f"[Anneal @ {self.n_calls:,} steps] "
                f"success_ema={self.success_ema:.3f} | "
                f"pos_noise={worker.pos_noise_scale} | "
                f"normal_noise={worker.normal_noise_scale:.3f} | "
                f"time_noise={worker.time_noise_scale:.3f}"
            )

        return True