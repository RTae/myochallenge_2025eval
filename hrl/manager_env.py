from __future__ import annotations

from typing import Tuple, Dict, Optional, Any
from collections import deque
import numpy as np
from myosuite.utils import gym
from stable_baselines3.common.vec_env import VecEnv

from config import Config
from custom_env import CustomEnv


class TableTennisManager(CustomEnv):
    """
    High-level manager.

    Observation = worker_obs (18) + normalized time (1) = 19
    Action      = goal_norm (6) in [-1,1]^6

    At each manager step:
      - set worker goal
      - run frozen worker policy for decision_interval environment steps
      - aggregate info and compute manager reward
      - terminate on goal_success or max steps
    """

    def __init__(
        self,
        worker_env: VecEnv,
        worker_model: Any,
        config: Config,
        decision_interval: int = 10,
        max_episode_steps: int = 800,
        success_buffer_len: int = 15,
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

        # Smooth goal success history (stabilizes reward)
        self.success_buffer = deque(maxlen=int(success_buffer_len))

    @property
    def sim(self):
        # for video callback
        return self.worker_env.envs[0].sim

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        self._worker_obs = self.worker_env.reset()
        self.current_step = 0
        self.success_buffer.clear()
        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        # ---- 1) set a new goal for worker ----
        goal_norm = np.clip(np.asarray(action, dtype=np.float32).reshape(6,), -1.0, 1.0)
        self.worker_env.env_method("set_goal", goal_norm, indices=0)

        # ---- 2) run frozen worker for K steps ----
        hit_any = False
        goal_success_any = False
        env_success_any = False

        for _ in range(self.decision_interval):
            obs_1d = np.asarray(self._worker_obs[0], dtype=np.float32)

            worker_action, _ = self.worker_model.predict(obs_1d, deterministic=True)

            obs, _, dones, infos = self.worker_env.step([worker_action])
            self._worker_obs = obs
            self.current_step += 1

            info = infos[0]
            hit_any |= bool(info.get("is_paddle_hit", False))
            goal_success_any |= bool(info.get("is_goal_success", False))
            env_success_any |= bool(info.get("is_success", False))

            if dones[0]:
                break

        # ---- 3) reward (smoothed by recent goal_success) ----
        self.success_buffer.append(1.0 if goal_success_any else 0.0)
        success_rate = float(np.mean(self.success_buffer)) if self.success_buffer else 0.0

        reward = self._reward_manager(
            hit=hit_any,
            goal_success=goal_success_any,
            env_success=env_success_any,
            success_rate=success_rate,
        )

        # ---- 4) termination/truncation ----
        # Recommended: terminate on goal_success (gives cleaner manager episodes)
        terminated = bool(goal_success_any)
        truncated = bool(self.current_step >= self.max_episode_steps)

        obs_out = self._build_obs()

        info_out = {
            "is_success": bool(env_success_any),
            "is_goal_success": bool(goal_success_any),
            "is_paddle_hit": bool(hit_any),
            "success_rate_smooth": float(success_rate),
            "goal_dx": float(goal_norm[0]),
            "goal_dy": float(goal_norm[1]),
            "goal_dz": float(goal_norm[2]),
        }

        return obs_out, float(reward), terminated, truncated, info_out

    def _build_obs(self) -> np.ndarray:
        worker_obs = np.asarray(self._worker_obs[0], dtype=np.float32).reshape(self.worker_obs_dim)
        t_frac = np.array([self.current_step / max(1, self.max_episode_steps)], dtype=np.float32)
        obs = np.hstack([worker_obs, t_frac])
        assert obs.shape == (self.observation_dim,)
        return obs

    def _reward_manager(self, hit: bool, goal_success: bool, env_success: bool, success_rate: float) -> float:
        """
        Manager reward:
        - small living cost
        - reward consistency (success_rate)
        - bonus for immediate goal_success
        - bigger bonus if env solved
        """
        r = -0.05
        r += 1.0 * success_rate

        if goal_success:
            r += 3.0

        if env_success:
            r += 2.0

        if hit:
            r += 0.2

        return float(r)