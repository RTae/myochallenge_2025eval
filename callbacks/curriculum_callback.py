from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class CurriculumCallback(BaseCallback):
    """
    - Linearly increases curriculum_level
    - Freezes curriculum when reward plateaus/improves
    """

    def __init__(
        self,
        cfg,
        total_steps,
        freeze_patience: int = 5,
        freeze_threshold: float = 0.05,
        window: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.cfg = cfg
        self.total_steps = total_steps

        self.freeze_patience = freeze_patience
        self.freeze_threshold = freeze_threshold
        self.window = window

        self.recent_rewards = []
        self.best_mean_reward = -np.inf
        self.freeze_counter = 0
        self.frozen = False

    def _on_step(self) -> bool:
        # ---------------- Curriculum ramp ----------------
        if not self.frozen:
            self.cfg.curriculum_level = min(
                1.0,
                self.model.num_timesteps / (0.6 * self.total_steps)
            )

        # ---------------- Reward tracking ----------------
        if "rollout/ep_rew_mean" in self.logger.name_to_value:
            r = self.logger.name_to_value["rollout/ep_rew_mean"]
            self.recent_rewards.append(r)

            if len(self.recent_rewards) > self.window:
                self.recent_rewards.pop(0)

            mean_r = np.mean(self.recent_rewards)

            # Check improvement
            if mean_r > self.best_mean_reward * (1 + self.freeze_threshold):
                self.best_mean_reward = mean_r
                self.freeze_counter = 0
            else:
                self.freeze_counter += 1

            # Freeze curriculum
            if (
                self.freeze_counter >= self.freeze_patience
                and not self.frozen
            ):
                self.frozen = True
                if self.verbose:
                    print(
                        f"🧊 Curriculum frozen at level "
                        f"{self.cfg.curriculum_level:.3f}"
                    )

        return True
