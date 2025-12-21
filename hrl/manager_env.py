# hrl/manager_env.py
from typing import Tuple, Dict, Optional, Any
from collections import deque
import numpy as np
from myosuite.utils import gym
from stable_baselines3.common.vec_env import VecEnv
from loguru import logger

from config import Config
from custom_env import CustomEnv


class TableTennisManager(CustomEnv):
    """
    High-level HRL manager.

    - Observes worker obs (18) + normalized time (1) = 19
    - Action = 6D goal
    - Runs frozen worker for decision_interval steps
    - Terminates on goal success or max steps
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
        self.decision_interval = int(decision_interval)
        self.max_episode_steps = int(max_episode_steps)

        self.worker_obs_dim = 18
        self.observation_dim = 19

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.current_step = 0
        self._worker_obs = None

        # Smoothed success buffer
        self.success_buffer = deque(maxlen=15)
        self.success_rate_thred = 0.8

    # --------------------------------------------------
    # Expose sim (for video callback)
    # --------------------------------------------------
    @property
    def sim(self):
        return self.worker_env.envs[0].sim

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:

        self._worker_obs = self.worker_env.reset()
        self.current_step = 0
        self.success_buffer.clear()

        obs = self._build_obs()
        return obs, {}

    # --------------------------------------------------
    # Step
    # --------------------------------------------------
    def step(self, action: np.ndarray):

        # Send goal to worker
        goal_norm = np.clip(action, -1.0, 1.0)
        self.worker_env.env_method("set_goal", goal_norm, indices=0)
        
        assert np.all(goal_norm >= -1.01) and np.all(goal_norm <= 1.01)
        

        hit = False
        is_goal_success = False
        is_success = False

        # Run worker for K steps
        for _ in range(self.decision_interval):
            obs_1d = np.asarray(self._worker_obs[0], dtype=np.float32)

            worker_action, _ = self.worker_model.predict(
                obs_1d, deterministic=True
            )

            obs, _, dones, infos = self.worker_env.step([worker_action])
            self._worker_obs = obs
            self.current_step += 1

            info = infos[0]

            hit |= bool(info.get("is_paddle_hit", False))
            is_goal_success |= bool(info.get("is_goal_success", False))
            is_success |= bool(info.get("is_success", False))

            if dones[0]:
                break

        # --------------------------------------------------
        # Reward (smoothed)
        # --------------------------------------------------
        self.success_buffer.append(1.0 if is_goal_success else 0.0)
        success_rate = float(np.mean(self.success_buffer))

        reward = self._reward_smoothed(hit, is_goal_success, is_success)

        # --------------------------------------------------
        # Termination
        # --------------------------------------------------
        terminated = bool(is_goal_success)
        truncated = bool(self.current_step >= self.max_episode_steps)

        obs = self._build_obs()

        info_out = {
            "is_success": is_success,
            "is_goal_success": is_goal_success,
            "is_paddle_hit": hit,
            "success_rate_smooth": success_rate,
            "goal_dx": float(goal_norm[0]),
            "goal_dy": float(goal_norm[1]),
            "goal_dz": float(goal_norm[2]),
        }

        return obs, reward, terminated, truncated, info_out

    # --------------------------------------------------
    # Observation
    # --------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        worker_obs = np.asarray(
            self._worker_obs[0], dtype=np.float32
        ).reshape(self.worker_obs_dim)

        t_frac = np.array(
            [self.current_step / max(1, self.max_episode_steps)],
            dtype=np.float32,
        )

        obs = np.hstack([worker_obs, t_frac])
        assert obs.shape == (self.observation_dim,)
        return obs

    # --------------------------------------------------
    # Reward
    # --------------------------------------------------
    def _reward_smoothed(self, is_hit: bool, is_goal_success: bool, is_success: bool) -> float:
        success_rate = float(np.mean(self.success_buffer)) if self.success_buffer else 0.0

        r = -0.05                     # small living cost
        r += 1.0 * success_rate       # consistency matters most

        if is_goal_success:
            r += 3.0                  # core signal

        if is_success:
            r += 2.0                  # downstream success bonus

        return float(r)