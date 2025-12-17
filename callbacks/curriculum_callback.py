from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
import numpy as np


class CurriculumCallback(BaseCallback):
    """
    Curriculum scheduler (SubprocVecEnv-safe)
    """

    def __init__(
        self,
        total_steps: int,
        freeze_patience: int = 5,
        freeze_threshold: float = 10.0,
        window: int = 10,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.total_steps = total_steps
        self.freeze_patience = freeze_patience
        self.freeze_threshold = freeze_threshold
        self.window = window

        self.recent_rewards = []
        self.best_mean_reward = -np.inf
        self.freeze_counter = 0
        self.frozen = False
        self.curriculum_level = 0.0

    def _on_step(self) -> bool:
        # -------------------------------------------------
        # 1) Curriculum ramp
        # -------------------------------------------------
        if not self.frozen:
            self.curriculum_level = min(
                1.0,
                self.model.num_timesteps / (0.6 * self.total_steps),
            )
            self.training_env.set_attr(
                "curriculum_level", float(self.curriculum_level)
            )

        # -------------------------------------------------
        # 2) Episode rewards
        # -------------------------------------------------
        ep_rewards = []
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                if "episode" in info:
                    ep_rewards.append(info["episode"]["r"])

        if len(ep_rewards) == 0:
            return True

        mean_ep_reward = safe_mean(ep_rewards)
        self.recent_rewards.append(mean_ep_reward)
        if len(self.recent_rewards) > self.window:
            self.recent_rewards.pop(0)

        # -------------------------------------------------
        # 3) Freeze logic
        # -------------------------------------------------
        if len(self.recent_rewards) == self.window:
            mean_r = np.mean(self.recent_rewards)

            if mean_r > self.best_mean_reward + self.freeze_threshold:
                self.best_mean_reward = mean_r
                self.freeze_counter = 0
            else:
                self.freeze_counter += 1

            if self.freeze_counter >= self.freeze_patience:
                self.frozen = True

        # -------------------------------------------------
        # 4) Logging
        # -------------------------------------------------
        self.logger.record("train/curriculum_level", self.curriculum_level)
        self.logger.record("train/curriculum_frozen", float(self.frozen))
        self.logger.record("train/episode_reward_mean", mean_ep_reward)

        return True
