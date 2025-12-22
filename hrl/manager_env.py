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
        → interpreted as Δgoal (correction) applied to physics prediction

    Final goal:
        goal_final = clip(goal_pred + action, -1, 1)
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

        # Action is Δgoal (correction)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.current_step = 0
        self._worker_obs = None

        # Smoothed success statistics
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
        Interpreted as Δgoal (correction on top of predicted goal)
        """

        # Rename for semantic clarity
        goal_delta = np.clip(
            np.asarray(action, dtype=np.float32).reshape(6,),
            -1.0,
            1.0,
        )

        # --------------------------------------------------
        # 1) Physics-based reference goal
        # --------------------------------------------------
        worker = self.worker_env.envs[0]
        obs_dict = worker.env.unwrapped.obs_dict

        # Normalized predicted goal ∈ [-1,1]^6
        goal_pred = worker.predict_goal_from_state(obs_dict)

        # Apply manager correction
        goal_final = np.clip(goal_pred + goal_delta, -1.0, 1.0)

        # Send to worker
        self.worker_env.env_method("set_goal", goal_final, indices=0)

        # --------------------------------------------------
        # 2) Run frozen worker
        # --------------------------------------------------
        hit_any = False
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
            hit_any |= bool(info.get("worker/is_paddle_hit"))
            goal_success_any |= bool(info.get("worker/is_goal_success"))
            env_success_any |= bool(info.get("is_success"))
            
            if dones[0]:
                break

        # --------------------------------------------------
        # 3) Reward
        # --------------------------------------------------
        self.success_buffer.append(1.0 if goal_success_any else 0.0)
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
        terminated = bool(goal_success_any)
        truncated = bool(self.current_step >= self.max_episode_steps)

        obs_out = self._build_obs()

        info_out = {
            "is_goal_success": bool(goal_success_any),
            "is_success": bool(env_success_any),
            "success_rate_smooth": success_rate,
            "goal_delta_norm": delta_norm,
            "goal_delta": goal_delta.copy(),
            "worker/cos_sim": infos[0].get("worker/cos_sim"),
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
        assert obs.shape == (self.observation_dim,)
        return obs

    # --------------------------------------------------
    # Manager Reward
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
        Reward intuition:
        - succeed often
        - succeed quickly
        - succeed with minimal correction
        """

        r = -0.05                      # living cost
        r += 1.0 * success_rate        # consistency

        if goal_success:
            r += 4.0

        if env_success:
            r += 3.0

        # Penalize large deviation from physics
        r -= 0.3 * delta_norm

        # Encourage small correction when successful
        if goal_success:
            r += 0.5 * np.exp(-delta_norm)

        return float(r)