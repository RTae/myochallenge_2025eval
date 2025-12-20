from typing import Tuple, Dict, Optional, Any
import numpy as np
from myosuite.utils import gym
from stable_baselines3.common.vec_env import VecEnv

from config import Config
from custom_env import CustomEnv


class TableTennisManager(CustomEnv):
    """
    Manager that controls a frozen worker policy via goals.

    worker_env: VecEnv (usually VecNormalize(DummyVecEnv([TableTennisWorker])))
    worker_model: frozen SB3 policy
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

        # -------------------------------------------------
        # Manager observation (derived from worker obs)
        # Worker obs = 18 state + 6 goal = 24
        # Manager obs = worker obs + 1 time proxy = 25
        # -------------------------------------------------
        self.observation_dim = 25
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Manager action = goal6
        # Must read bounds from raw worker
        # -------------------------------------------------
        base_worker = self.worker_env.envs[0]
        self.goal_low = np.asarray(base_worker.goal_low, np.float32)
        self.goal_high = np.asarray(base_worker.goal_high, np.float32)

        self.action_space = gym.spaces.Box(
            low=self.goal_low,
            high=self.goal_high,
            shape=(6,),
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Runtime state
        # -------------------------------------------------
        self.current_step = 0
        self._worker_obs = None  # shape (1, obs_dim)
        self.total_hits = 0

    # ============================================================
    # Reset
    # ============================================================
    def reset(self) -> Tuple[np.ndarray, Dict]:
        self._worker_obs = self.worker_env.reset()
        self.current_step = 0
        self.total_hits = 0

        return (
            self._build_manager_obs(self._worker_obs),
            {"is_success": False},
        )

    # ============================================================
    # Step
    # ============================================================
    def step(self, action: np.ndarray):
        # -------------------------------------------------
        # 1) Clip & set goal on RAW worker
        # -------------------------------------------------
        goal = np.clip(action, self.goal_low, self.goal_high).astype(np.float32)
        self.worker_env.env_method("set_goal", goal, indices=0)

        paddle_hit = False
        is_sucess = False
        terminated = False
        truncated = False

        # -------------------------------------------------
        # 2) Roll worker for decision_interval steps
        # -------------------------------------------------
        for _ in range(self.decision_interval):
            worker_obs_1d = self._worker_obs[0]
            worker_action, _ = self.worker_model.predict(
                worker_obs_1d, deterministic=True
            )

            obs, rewards, dones, infos = self.worker_env.step([worker_action])
            self._worker_obs = obs
            self.current_step += 1

            info0 = infos[0]

            if info0.get("hit", False):
                paddle_hit = True
                self.total_hits += 1
                
            if info0.get("is_success", False):
                is_success = True
                
            if bool(dones[0]):
                terminated = True
                break

            if self.current_step >= self.max_episode_steps:
                truncated = True
                break

        # -------------------------------------------------
        # 3) Reward
        # -------------------------------------------------
        reward = self._calculate_reward(paddle_hit)

        obs_out = (
            self._build_manager_obs(self._worker_obs)
            if not (terminated or truncated)
            else np.zeros(self.observation_dim, dtype=np.float32)
        )

        return obs_out, float(reward), terminated, truncated, {"is_paddle_hit": paddle_hit, "is_success": is_success}

    # ============================================================
    # Observation builder
    # ============================================================
    def _build_manager_obs(self, worker_obs_vec: np.ndarray) -> np.ndarray:
        """
        Worker obs is already normalized if VecNormalize is used.
        Shape: (1, 24)
        Manager obs: [worker_obs, time_proxy]
        """
        w = np.asarray(worker_obs_vec[0], np.float32)

        # Safety
        if w.shape[0] != 24:
            ww = np.zeros(24, np.float32)
            ww[: min(24, w.shape[0])] = w[: min(24, w.shape[0])]
            w = ww

        # Simple time proxy (normalized progress)
        t = np.array(
            [self.current_step / max(1, self.max_episode_steps)],
            dtype=np.float32,
        )

        return np.concatenate([w, t], axis=0)

    # ============================================================
    # Reward
    # ============================================================
    def _calculate_reward(self, hit: bool) -> float:
        # Sparse & stable
        return 30.0 if hit else -0.1
