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
    High-level HRL manager (Δ-goal formulation).

    Observation:
        worker_obs (18) + normalized time (1) = 19

    Action:
        action ∈ [-1, 1]^6
        → interpreted as Δgoal (small correction)
    """

    def __init__(
        self,
        worker_env: VecEnv,
        worker_model: Any,
        config: Config,
        decision_interval: int = 10,
        max_episode_steps: int = 800,
        success_buffer_len: int = 20,
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

        # Action: Δgoal (normalized)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.delta_scale = np.array(
            [
                0.15,  # Δx
                0.15,  # Δy
                0.10,  # Δz
                0.10,  # Δnx
                0.10,  # Δny
                0.20,  # Δdt
            ],
            dtype=np.float32,
        )

        self.current_step = 0
        self._worker_obs = None

        # Smoothed success statistics (weak stabilizer only)
        self.success_buffer = deque(maxlen=int(success_buffer_len))

    # --------------------------------------------------
    # Access sim (for video callbacks)
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

        return self._build_obs(), {}

    # --------------------------------------------------
    # Step
    # --------------------------------------------------
    def step(self, action: np.ndarray):
        """
        action ∈ [-1, 1]^6
        Interpreted as scaled Δgoal correction
        """

        # --------------------------------------------------
        # 1) Scale and apply Δgoal
        # --------------------------------------------------
        raw_delta = np.clip(
            np.asarray(action, dtype=np.float32).reshape(6,),
            -1.0,
            1.0,
        )

        goal_delta = raw_delta * self.delta_scale

        worker = self.worker_env.envs[0]
        obs_dict = worker.env.unwrapped.obs_dict

        # Physics-based prediction
        goal_pred = worker.predict_goal_from_state(obs_dict)

        # Final goal sent to worker
        goal_final = np.clip(goal_pred + goal_delta, -1.0, 1.0)
        self.worker_env.env_method("set_goal", goal_final, indices=0)

        # --------------------------------------------------
        # 2) Run frozen worker
        # --------------------------------------------------
        goal_success_any = False
        env_success_any = False

        for _ in range(self.decision_interval):
            obs_1d = np.asarray(self._worker_obs[0], dtype=np.float32)

            worker_action, _ = self.worker_model.predict(
                obs_1d, deterministic=True
            )

            obs, _, dones, infos = self.worker_env.step(worker_action[None])
            self._worker_obs = obs
            self.current_step += 1

            info = infos[0]
            goal_success_any |= bool(info.get("is_goal_success"))
            env_success_any |= bool(info.get("is_success"))

            if dones[0]:
                break

        # --------------------------------------------------
        # 3) Reward
        # --------------------------------------------------
        self.success_buffer.append(1.0 if env_success_any else 0.0)
        success_rate = float(np.mean(self.success_buffer)) if self.success_buffer else 0.0

        delta_norm = float(np.linalg.norm(goal_delta))

        reward = self._reward_manager(
            goal_success=goal_success_any,
            env_success=env_success_any,
            success_rate=success_rate,
            delta_norm=delta_norm,
        )

        # --------------------------------------------------
        # 4) Termination
        # --------------------------------------------------
        terminated = bool(env_success_any)
        truncated = bool(self.current_step >= self.max_episode_steps)

        obs_out = self._build_obs()

        info_out = {
            "goal_delta_norm": delta_norm,
            "goal_delta": goal_delta.copy(),
            **(infos[0] if infos and isinstance(infos[0], dict) else {})
        }

        return obs_out, float(reward), terminated, truncated, info_out

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
        return obs

    # --------------------------------------------------
    # Manager Reward (FINAL, ALIGNED)
    # --------------------------------------------------
    def _reward_manager(
        self,
        *,
        goal_success: bool,
        env_success: bool,
        success_rate: float,
        delta_norm: float,
    ) -> float:
        """
        Manager reward:
        - env_success dominates
        - goal_success is a bridge
        - success_rate stabilizes
        - delta_norm lightly regularizes
        """

        r = -0.05  # living cost

        # --- FINAL TASK ---
        if env_success:
            r += 8.0
            return r  # stop credit leakage

        # --- BRIDGE SIGNAL ---
        if goal_success:
            r += 1.5

        # --- STABILITY (weak) ---
        r += 0.2 * success_rate

        # --- REGULARIZATION ---
        r -= 0.1 * (delta_norm ** 2)

        return float(r)