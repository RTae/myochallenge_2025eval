from typing import Tuple, Dict, Optional
import numpy as np
from myosuite.utils import gym
from stable_baselines3.common.vec_env import VecEnv

from config import Config
from custom_env import CustomEnv


class TableTennisManager(CustomEnv):
    """
    Manager issues goals to a frozen Worker.

    Observation (19):
      worker_obs (18) + time_progress (1)

    Action:
      goal (6)
    """

    def __init__(
        self,
        worker_env: VecEnv,
        worker_model,
        config: Config,
        decision_interval=10,
        max_episode_steps=800,
    ):
        super().__init__(config)

        self.worker_env = worker_env
        self.worker_model = worker_model
        self.decision_interval = decision_interval
        self.max_episode_steps = max_episode_steps

        self.obs_dim = 19
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.obs_dim,), np.float32
        )

        base_worker = self.worker_env.envs[0]
        self.goal_low = base_worker.goal_low
        self.goal_high = base_worker.goal_high

        self.action_space = gym.spaces.Box(
            self.goal_low, self.goal_high, dtype=np.float32
        )

        self.current_step = 0
        self.worker_obs = None

    @property
    def sim(self):
        return self.worker_env.envs[0].sim

    def reset(self, *, seed=None, options=None):
        self.worker_obs = self.worker_env.reset()
        self.current_step = 0
        return self._build_obs(), {"is_success": False}

    def step(self, action):
        goal = np.clip(action, self.goal_low, self.goal_high)
        self.worker_env.env_method("set_goal", goal)

        hit = False
        success = False

        for _ in range(self.decision_interval):
            obs = self.worker_obs[0]
            act, _ = self.worker_model.predict(obs, deterministic=True)

            self.worker_obs, _, dones, infos = self.worker_env.step([act])
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

    def _build_obs(self):
        w = self.worker_obs[0]
        t = np.array([self.current_step / self.max_episode_steps], np.float32)
        return np.concatenate([w, t])

    def _reward(self, hit, success):
        if success:
            return 50.0
        if hit:
            return 5.0
        return -0.1