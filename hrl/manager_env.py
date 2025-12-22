# hrl/manager_env.py
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
    High-level HRL manager.

    Observation:
      - worker_obs (worker_obs_dim) + normalized time fraction (1)

    Action:
      - 6D residual in normalized goal space [-1, 1]^6

    Control:
      - computes oracle goal from worker's calculate_prediction-based helper
      - applies residual: goal = clip(oracle + residual_scale * action, [-1, 1])
      - sets goal in worker via env_method("set_goal", goal_norm)

    Termination:
      - terminated on worker goal success (is_goal_success)
      - truncated on max manager steps
    """

    def __init__(
        self,
        worker_env: VecEnv,
        worker_model: Any,
        config: Config,
        decision_interval: int = 10,
        max_episode_steps: int = 800,
        residual_scale: float = 0.3,
        success_buffer_len: int = 15,
    ):
        super().__init__(config)

        self.worker_env = worker_env
        self.worker_model = worker_model
        self.decision_interval = int(decision_interval)
        self.max_episode_steps = int(max_episode_steps)
        self.residual_scale = float(residual_scale)

        # NOTE: this must match what your worker env actually returns as obs dimension
        self.worker_obs_dim = 18
        self.observation_dim = self.worker_obs_dim + 1

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # Manager always outputs normalized residuals
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.current_step = 0
        self._worker_obs = None

        # Smoothed goal-success buffer (tracks worker goal success, not env solved)
        self.success_buffer = deque(maxlen=int(success_buffer_len))

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

        # Reset worker env; this DOES call worker.reset() inside the subprocess
        self._worker_obs = self.worker_env.reset()
        self.current_step = 0
        self.success_buffer.clear()

        obs = self._build_obs()
        return obs, {}

    # --------------------------------------------------
    # Step
    # --------------------------------------------------
    def step(self, action: np.ndarray):
        """
        action: residual (6,) in [-1, 1]^6
        """
        # Ensure correct shape
        residual = np.asarray(action, dtype=np.float32).reshape(6,)
        residual = np.clip(residual, -1.0, 1.0)

        # --------------------------------------------------
        # 1) Oracle goal from current worker/base state
        # --------------------------------------------------
        base_worker = self.worker_env.envs[0]  # worker wrapper instance inside vec env
        obs_dict = base_worker.env.unwrapped.obs_dict  # raw dict from myosuite env

        # predict_goal_from_state should return normalized goal in [-1, 1]^6
        oracle_goal = np.asarray(base_worker.predict_goal_from_state(obs_dict), dtype=np.float32).reshape(6,)
        oracle_goal = np.clip(oracle_goal, -1.0, 1.0)

        # --------------------------------------------------
        # 2) Residual goal = oracle + scaled residual
        # --------------------------------------------------
        goal_norm = np.clip(oracle_goal + self.residual_scale * residual, -1.0, 1.0)

        # Send goal to worker; worker.set_goal resets goal_start_time per-goal (in your latest worker)
        self.worker_env.env_method("set_goal", goal_norm, indices=0)

        # --------------------------------------------------
        # 3) Rollout worker for decision_interval steps
        # --------------------------------------------------
        is_hit = False
        is_goal_success = False
        is_success = False  # env-level success ("solved")

        for _ in range(self.decision_interval):
            # worker obs for SB3 policy
            obs_1d = np.asarray(self._worker_obs[0], dtype=np.float32)

            worker_action, _ = self.worker_model.predict(obs_1d, deterministic=True)

            obs, _, dones, infos = self.worker_env.step([worker_action])
            self._worker_obs = obs
            self.current_step += 1

            info = infos[0]
            is_hit |= bool(info.get("is_paddle_hit", False))
            is_goal_success |= bool(info.get("is_goal_success", False))
            is_success |= bool(info.get("is_success", False))

            if dones[0]:
                break

        # --------------------------------------------------
        # 4) Reward (manager-level)
        # --------------------------------------------------
        self.success_buffer.append(1.0 if is_goal_success else 0.0)
        success_rate = float(np.mean(self.success_buffer)) if self.success_buffer else 0.0

        reward = self._reward_smoothed(
            is_hit=is_hit,
            is_goal_success=is_goal_success,
            is_success=is_success,
            success_rate=success_rate,
        )

        # --------------------------------------------------
        # 5) Termination / truncation
        # --------------------------------------------------
        terminated = bool(is_goal_success)  # one goal episode
        truncated = bool(self.current_step >= self.max_episode_steps)

        obs = self._build_obs()

        info_out = {
            "is_success": is_success,
            "is_goal_success": is_goal_success,
            "is_paddle_hit": is_hit,
            "success_rate_smooth": success_rate,
            "oracle_goal": oracle_goal.astype(np.float32),
            "goal_norm": goal_norm.astype(np.float32),
        }

        return obs, float(reward), terminated, truncated, info_out

    # --------------------------------------------------
    # Observation
    # --------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        """
        Concatenate:
          - worker_obs (worker_obs_dim,)
          - time fraction (1,)
        """
        worker_obs = np.asarray(self._worker_obs[0], dtype=np.float32).reshape(self.worker_obs_dim)

        t_frac = np.array(
            [self.current_step / max(1, self.max_episode_steps)],
            dtype=np.float32,
        )

        obs = np.hstack([worker_obs, t_frac])
        assert obs.shape == (self.observation_dim,), f"obs.shape={obs.shape}"
        return obs

    # --------------------------------------------------
    # Reward
    # --------------------------------------------------
    def _reward_smoothed(
        self,
        *,
        is_hit: bool,
        is_goal_success: bool,
        is_success: bool,
        success_rate: float,
    ) -> float:
        """
        Manager reward:
          - living cost
          - consistency bonus (smoothed goal success)
          - large bonus for goal success
          - smaller shaping for hit
          - bonus for env-level solved
        """
        r = -0.1
        r += 2.0 * float(success_rate)

        if is_goal_success:
            r += 6.0

        if is_hit:
            r += 1.0

        if is_success:
            r += 3.0

        return float(r)