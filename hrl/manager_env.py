from typing import Tuple, Dict, Any
import numpy as np
from myosuite.utils import gym
from stable_baselines3.common.vec_env import VecEnv

from config import Config
from custom_env import CustomEnv


class TableTennisManager(CustomEnv):
    """
    High-level (manager) policy.

    - Chooses 6D goals for the worker
    - Executes worker for decision_interval steps
    - Rewarded ONLY on real task success (env solved)
    """

    def __init__(
        self,
        worker_env: VecEnv,          # VecNormalize(DummyVecEnv([Worker]))
        worker_model: Any,           # frozen PPO worker
        config: Config,
        decision_interval: int = 10,
        max_episode_steps: int = 800,
    ):
        super().__init__(config)

        self.worker_env = worker_env
        self.worker_model = worker_model
        self.decision_interval = decision_interval
        self.max_episode_steps = max_episode_steps

        # --------------------------------------------------
        # Worker obs = 18 dims
        # Manager obs = worker_obs (18) + time proxy (1) = 19
        # --------------------------------------------------
        self.worker_obs_dim = 18
        self.observation_dim = 19

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Manager action = worker goal (6D)
        # Read bounds from raw worker
        # --------------------------------------------------
        base_worker = self.worker_env.envs[0]
        self.goal_low = np.asarray(base_worker.goal_low, dtype=np.float32)
        self.goal_high = np.asarray(base_worker.goal_high, dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=self.goal_low,
            high=self.goal_high,
            shape=(6,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Runtime state
        # --------------------------------------------------
        self.current_step = 0
        self._worker_obs = None
        self.total_hits = 0

    # ====================================================
    # Reset
    # ====================================================
    def reset(self) -> Tuple[np.ndarray, Dict]:
        self._worker_obs = self.worker_env.reset()
        self.current_step = 0
        self.total_hits = 0

        return self._build_manager_obs(), {"is_success": False}

    # ====================================================
    # Step
    # ====================================================
    def step(self, action: np.ndarray):
        # -----------------------------------------------
        # 1) Set goal on raw worker
        # -----------------------------------------------
        goal = np.clip(action, self.goal_low, self.goal_high).astype(np.float32)
        self.worker_env.env_method("set_goal", goal, indices=0)

        paddle_hit = False
        is_success = False
        terminated = False
        truncated = False

        # -----------------------------------------------
        # 2) Roll worker
        # -----------------------------------------------
        for _ in range(self.decision_interval):
            worker_obs_1d = self._worker_obs[0]
            worker_action, _ = self.worker_model.predict(
                worker_obs_1d,
                deterministic=True,
            )

            obs, _, dones, infos = self.worker_env.step([worker_action])
            self._worker_obs = obs
            self.current_step += 1

            info0 = infos[0]

            # Auxiliary metrics (NOT reward)
            if info0.get("hit", False):
                paddle_hit = True
                self.total_hits += 1

            # TRUE task success (ball landed correctly)
            if info0.get("is_success", False):
                is_success = True

            if bool(dones[0]):
                terminated = True
                break

            if self.current_step >= self.max_episode_steps:
                truncated = True
                break

        # -----------------------------------------------
        # 3) Reward (SUCCESS ONLY)
        # -----------------------------------------------
        reward = self._calculate_reward(is_success)

        obs_out = (
            self._build_manager_obs()
            if not (terminated or truncated)
            else np.zeros(self.observation_dim, dtype=np.float32)
        )

        info = {
            "is_success": bool(is_success),
            "paddle_hit": bool(paddle_hit),
            "total_hits": int(self.total_hits),
        }

        return obs_out, reward, terminated, truncated, info

    # ====================================================
    # Observation
    # ====================================================
    def _build_manager_obs(self) -> np.ndarray:
        """
        Manager sees:
        - worker observation (18)
        - normalized time progress (1)
        """
        w = np.asarray(self._worker_obs[0], dtype=np.float32)

        if w.shape[0] != self.worker_obs_dim:
            ww = np.zeros(self.worker_obs_dim, dtype=np.float32)
            ww[: min(len(w), self.worker_obs_dim)] = w[: min(len(w), self.worker_obs_dim)]
            w = ww

        t = np.array(
            [self.current_step / max(1, self.max_episode_steps)],
            dtype=np.float32,
        )

        obs = np.concatenate([w, t], axis=0)

        assert obs.shape == (self.observation_dim,), f"Manager obs shape mismatch {obs.shape}"
        return obs

    # ====================================================
    # Reward
    # ====================================================
    def _calculate_reward(self, is_success: bool) -> float:
        """
        Manager is rewarded ONLY on real task success.
        """
        return 100.0 if is_success else -1.0