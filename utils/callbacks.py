import os
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from loguru import logger

from utils.model_helper import make_video_env


class VideoCallback(BaseCallback):
    """
    - Logs env-steps/sec (like your JAX loop).
    - Periodically records evaluation video.
    - (Optional) A hook point to run ES/HRL manager-side updates without blocking PPO too long.

    You can replace the stubbed ES/HManager bits with your own logic later.
    """
    def __init__(
        self,
        env_id: str,
        seed: int,
        logdir: str,
        video_freq: int = 50_000,
        eval_episodes: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_id = env_id
        self.seed = seed
        self.logdir = logdir
        self.video_freq = video_freq
        self.eval_episodes = eval_episodes

        self._t0 = None
        self._last_steps = 0
        self._video_dir = os.path.join(logdir, "videos")
        os.makedirs(self._video_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        self._t0 = time.time()
        self._last_steps = 0
        logger.info(f"🎬 Video output => {self._video_dir}")

    def _on_step(self) -> bool:
        total_steps = self.model.num_timesteps

        if (total_steps > 0) and (total_steps % self.video_freq == 0):
            self._record_eval_video_and_metrics()

        return True

    def _record_eval_video_and_metrics(self):
        def trigger(ep_id: int) -> bool:
            return ep_id == 1

        eval_env = make_video_env(self.env_id, self.seed + 42, self._video_dir, trigger)
        try:
            mean_r, std_r = evaluate_policy(
                self.model,
                eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=False,
                render=True,
                warn=False,
            )
            if self.logger is not None:
                self.logger.record("eval/return_mean", float(mean_r))
                self.logger.record("eval/return_std", float(std_r))
            logger.info(f"[Eval] meanR={mean_r:.2f} ± {std_r:.2f}")
        finally:
            eval_env.close()