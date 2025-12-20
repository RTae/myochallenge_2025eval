from typing import Tuple, Dict, Optional, Any
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal


class TableTennisManager(CustomEnv):
    def __init__(
        self,
        worker_env: Any,
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

        self.observation_dim = 25
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=self.worker_env.goal_low,
            high=self.worker_env.goal_high,
            shape=(6,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.total_hits = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        self.worker_env.reset(seed)
        self.worker_env.reset_hrl_state()
        self.current_step = 0
        return self._augment_observation(), {"is_success": False}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.worker_env.goal_low, self.worker_env.goal_high)
        self.worker_env.set_goal(action)

        hit = False
        terminated = False

        for _ in range(self.decision_interval):
            obs = self.worker_env._augment_observation()
            a, _ = self.worker_model.predict(obs, deterministic=True)
            _, _, term, trunc, _ = self.worker_env.step(a)
            self.current_step += 1

            touching = self.worker_env.env.obs_dict.get("touching_info", [0])
            if touching[0] > 0.5:
                hit = True
                self.total_hits += 1

            if term or trunc:
                terminated = True
                break

        reward = self._calculate_reward(hit)
        truncated = self.current_step >= self.max_episode_steps

        obs = self._augment_observation() if not (terminated or truncated) else np.zeros(self.observation_dim, np.float32)

        return obs, reward, terminated, truncated, {"is_success": hit}

    def _augment_observation(self):
        obs = self.worker_env.env.obs_dict
        return np.hstack([
            obs["ball_pos"], obs["ball_vel"],
            obs["paddle_pos"], obs["paddle_vel"],
            quat_to_paddle_normal(obs["paddle_ori"]),
            obs["reach_err"], obs["touching_info"],
            [obs["time"]],
        ]).astype(np.float32)

    def _calculate_reward(self, hit: bool):
        return 30.0 if hit else -1.0
