from typing import Tuple, Dict, Optional, Any
import numpy as np
from myosuite.utils import gym
from stable_baselines3.common.vec_env import VecEnv

from config import Config
from custom_env import CustomEnv


class TableTennisManager(CustomEnv):
    """
    High-level manager for HRL.

    Learns to output goal6 that conditions a frozen worker.
    """

    def __init__(
        self,
        worker_env: VecEnv,
        worker_model: Any,
        config: Config,
        decision_interval: int = 10,
        max_episode_steps: int = 800,
    ):
        super().__init__(config)

        self.worker_env = worker_env
        self.worker_model = worker_model
        self.decision_interval = decision_interval
        self.max_episode_steps = max_episode_steps

        # worker obs (21) + time proxy
        self.worker_obs_dim = 21
        self.observation_dim = 22

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        base_worker = self.worker_env.envs[0]
        self.goal_low = base_worker.goal_low
        self.goal_high = base_worker.goal_high

        self.action_space = gym.spaces.Box(
            low=self.goal_low,
            high=self.goal_high,
            shape=(6,),
            dtype=np.float32,
        )

        self.current_step = 0
        self._worker_obs = None

        # curriculum
        self.curriculum_stage = 0
        self.success_window = []
        self.window_size = 50
        self.advance_threshold = 0.7

    @property
    def sim(self):
        return self.worker_env.envs[0].sim

    # ============================================================
    # Reset
    # ============================================================
    def reset(self, *, seed=None, options=None):
        self._worker_obs = self.worker_env.reset()
        self.current_step = 0
        return self._build_obs(), {"is_success": False}

    # ============================================================
    # Step
    # ============================================================
    def step(self, action: np.ndarray):
        goal = np.clip(action, self.goal_low, self.goal_high)
        self.worker_env.env_method("set_goal", goal, indices=0)

        paddle_hit = False
        is_success = False

        for _ in range(self.decision_interval):
            a, _ = self.worker_model.predict(self._worker_obs[0], deterministic=True)
            obs, _, dones, infos = self.worker_env.step([a])
            self._worker_obs = obs
            self.current_step += 1

            info0 = infos[0]
            paddle_hit |= info0.get("hit", False)
            is_success |= info0.get("is_success", False)

            if dones[0] or self.current_step >= self.max_episode_steps:
                break

        self._update_curriculum(paddle_hit, is_success)
        reward = self._calculate_reward(paddle_hit, is_success)

        return self._build_obs(), reward, False, False, {
            "is_success": is_success,
            "is_paddle_hit": paddle_hit,
            "curriculum_stage": self.curriculum_stage,
        }

    # ============================================================
    # Observation
    # ============================================================
    def _build_obs(self):
        t = np.array([self.current_step / self.max_episode_steps], np.float32)
        return np.concatenate([self._worker_obs[0], t])

    # ============================================================
    # Curriculum
    # ============================================================
    def _update_curriculum(self, hit, success):
        signal = hit if self.curriculum_stage == 0 else success
        self.success_window.append(int(signal))
        if len(self.success_window) > self.window_size:
            self.success_window.pop(0)
        if (
            len(self.success_window) == self.window_size
            and np.mean(self.success_window) > self.advance_threshold
            and self.curriculum_stage < 2
        ):
            self.curriculum_stage += 1
            self.success_window.clear()

    # ============================================================
    # Low-variance reward
    # ============================================================
    def _calculate_reward(self, hit: bool, success: bool) -> float:
        reward = -0.05  # small living penalty

        if hit:
            reward += 1.0

        if self.curriculum_stage >= 1 and success:
            reward += 10.0

        if self.curriculum_stage == 2 and success:
            reward += 25.0

        return reward