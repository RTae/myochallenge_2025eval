# hrl/manager_env.py
from typing import Tuple, Dict, Optional, Any
import numpy as np
from myosuite.utils import gym
from stable_baselines3.common.vec_env import VecEnv

from config import Config
from custom_env import CustomEnv


class TableTennisManager(CustomEnv):
    """
    High-level HRL manager.

    Issues goals to a frozen worker.
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

        # Worker obs = 21, + time proxy = 22
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

    # ------------------------------------------------
    # Expose sim for video callback
    # ------------------------------------------------
    @property
    def sim(self):
        return self.worker_env.envs[0].sim

    # ------------------------------------------------
    # Gym API
    # ------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self._worker_obs = self.worker_env.reset()
        self.current_step = 0
        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        goal = np.clip(action, self.goal_low, self.goal_high)
        self.worker_env.env_method("set_goal", goal, indices=0)

        hit = False
        success = False

        for _ in range(self.decision_interval):
            obs_1d = self._worker_obs[0]
            worker_action, _ = self.worker_model.predict(obs_1d, deterministic=True)

            obs, _, dones, infos = self.worker_env.step([worker_action])
            self._worker_obs = obs
            self.current_step += 1

            info = infos[0]
            hit |= info.get("is_paddle_hit", False)
            success |= info.get("is_success", False)

            if dones[0] or self.current_step >= self.max_episode_steps:
                break

        reward = self._reward(hit, success)

        return self._build_obs(), reward, False, False, {
            "is_success": success,
            "is_paddle_hit": hit,
        }

    # ------------------------------------------------
    # Observation
    # ------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        worker_obs = np.asarray(self._worker_obs[0], dtype=np.float32)
        t = np.asarray(
            [self.current_step / max(1, self.max_episode_steps)],
            dtype=np.float32,
        )

        obs = np.hstack([worker_obs, t])
        assert obs.shape == (22,)
        return obs

    # ------------------------------------------------
    # Manager reward (LOW VARIANCE)
    # ------------------------------------------------
    def _reward(self, hit: bool, success: bool) -> float:
        if success:
            return 10.0
        if hit:
            return 1.0
        return -0.1