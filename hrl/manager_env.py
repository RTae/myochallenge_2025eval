from typing import Tuple, Dict, Optional, Any
import numpy as np
from myosuite.utils import gym
from stable_baselines3.common.vec_env import VecEnv

from config import Config
from custom_env import CustomEnv


class TableTennisManager(CustomEnv):
    """
    High-level controller.
    Learns WHICH goal to give to the worker.
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

        # ===============================
        # Observation: worker obs + ball_vel + time
        # ===============================
        self.observation_dim = 18 + 3 + 1

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

    @property
    def sim(self):
        return self.worker_env.envs[0].sim

    # ------------------------------------------------
    def reset(self, seed=None, options=None):
        self._worker_obs = self.worker_env.reset()
        self.current_step = 0
        return self._build_obs(), {}

    def step(self, action):
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
            hit |= info.get("hit", False)
            success |= info.get("is_success", False)

            if dones[0]:
                break

        reward = self._reward(hit, success)
        return self._build_obs(), reward, False, False, {
            "is_success": success,
            "is_paddle_hit": hit,
        }

    # ------------------------------------------------
    def _build_obs(self):
        worker_obs = self._worker_obs[0]

        obs = self.worker_env.envs[0].env.unwrapped.obs_dict
        ball_vel = np.asarray(obs["ball_vel"], np.float32)
        t = np.array([self.current_step / self.max_episode_steps], np.float32)

        return np.concatenate([worker_obs, ball_vel, t])

    def _reward(self, hit: bool, success: bool) -> float:
        reward = -0.1
        if hit:
            reward += 5.0
        if success:
            reward += 50.0
        return reward