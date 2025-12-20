from typing import Tuple, Dict, Optional, Any
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal
from hrl.worker_env import TableTennisWorker


class TableTennisManager(CustomEnv):
    def __init__(
        self,
        worker_env: TableTennisWorker,
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
            low=-np.inf, high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
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
        self.worker_env.reset(seed=seed)
        self.worker_env.reset_hrl_state()
        self.current_step = 0
        self.total_hits = 0
        return self._augment_observation(), {"is_success": False}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.worker_env.goal_low, self.worker_env.goal_high)
        self.worker_env.set_goal(action)

        hit = False
        terminated = False
        truncated = False

        for _ in range(self.decision_interval):
            worker_obs = self.worker_env._augment_observation()
            a, _ = self.worker_model.predict(worker_obs, deterministic=True)

            _, _, term, trunc, _ = self.worker_env.step(a)
            self.current_step += 1

            touching = self.worker_env.env.obs_dict.get("touching_info", [0])
            if touching[0] > 0.5:
                hit = True
                self.total_hits += 1

            if term:
                terminated = True
                break
            if trunc:
                truncated = True
                break

        truncated = truncated or (self.current_step >= self.max_episode_steps)
        reward = self._calculate_reward(hit)

        obs = (
            self._augment_observation()
            if not (terminated or truncated)
            else np.zeros(self.observation_dim, dtype=np.float32)
        )

        return obs, float(reward), terminated, truncated, {"is_success": hit}

    def _augment_observation(self) -> np.ndarray:
        obs = self.worker_env.env.unwrapped.obs_dict

        def flat(x):
            return np.asarray(x, dtype=np.float32).reshape(-1)

        return np.concatenate([
            flat(obs["ball_pos"]),            # 3
            flat(obs["ball_vel"]),            # 3
            flat(obs["paddle_pos"]),          # 3
            flat(obs["paddle_vel"]),          # 3
            flat(quat_to_paddle_normal(obs["paddle_ori"])),  # 3
            flat(obs["reach_err"]),           # 3
            flat(obs["touching_info"]),       # 6
            np.array([float(obs["time"])], dtype=np.float32),  # 1
        ], axis=0)

    def _calculate_reward(self, hit: bool) -> float:
        return 30.0 if hit else -1.0
