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
    High-level HRL manager (RAW physical Δ-goal).

    Observation:
        worker_obs (worker_obs_dim) + normalized time (1)

    Action:
        Δgoal_phys ∈ ℝ⁶
        [dx, dy, dz, dnx, dny, ddt]

    The manager directly adjusts the worker's physical goal.
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

        # --------------------------------------------------
        # Safety check: worker must be fully trained
        # --------------------------------------------------
        t_progress = float(np.mean(self.worker_env.env_method("get_progress")))
        t_noise = self.worker_env.env_method("goal_noise_scale")

        assert (
            t_progress >= 1.0 and not any(t_noise)
        ), (
            f"Manager requires fully unlocked worker "
            f"(progress=1, noise=0), got progress={t_progress}, noise={t_noise}"
        )

        self.decision_interval = int(decision_interval)
        self.max_episode_steps = int(max_episode_steps)

        # --------------------------------------------------
        # Observation space
        # --------------------------------------------------
        self.worker_obs_dim = int(self.worker_env.observation_space.shape[0])
        self.observation_dim = self.worker_obs_dim + 1

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Action space: RAW physical Δ-goal
        # --------------------------------------------------
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Runtime
        # --------------------------------------------------
        self.current_step = 0
        self._worker_obs = None
        self.success_buffer = deque(maxlen=int(success_buffer_len))

    # --------------------------------------------------
    # Simulator shortcut
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
        # --------------------------------------------------
        # 1) Predict base goal (PHYSICAL)
        # --------------------------------------------------
        worker0 = self.worker_env.envs[0]
        obs_dict0 = worker0.env.unwrapped.obs_dict

        goal_pred = worker0.predict_goal_from_state(obs_dict0)  # (6,)

        # --------------------------------------------------
        # 2) Apply RAW physical delta
        # --------------------------------------------------
        goal_delta = action.astype(np.float32)

        # Optional SAFETY clamp (not normalization)
        goal_delta = np.clip(goal_delta, -0.3, 0.3)

        goal_final = goal_pred + goal_delta

        self.worker_env.env_method("set_goal", goal_final)

        # --------------------------------------------------
        # 3) Run frozen worker
        # --------------------------------------------------
        goal_success_any = False
        env_success_any = False
        last_infos = None
        last_dones = None

        for _ in range(self.decision_interval):
            obs_batch = np.asarray(self._worker_obs, dtype=np.float32)

            worker_actions, _ = self.worker_model.predict(
                obs_batch, deterministic=True
            )

            obs, _, dones, infos = self.worker_env.step(worker_actions)
            self._worker_obs = obs
            self.current_step += 1

            last_infos = infos
            last_dones = dones

            for info in infos:
                goal_success_any |= bool(info.get("is_goal_success", 0.0))
                env_success_any |= bool(info.get("is_success", 0.0))

            if np.any(dones):
                break

        # --------------------------------------------------
        # 4) Manager reward
        # --------------------------------------------------
        self.success_buffer.append(1.0 if env_success_any else 0.0)
        success_rate = float(np.mean(self.success_buffer))
        delta_norm = float(np.linalg.norm(goal_delta))

        reward = self._compute_reward(
            goal_success=goal_success_any,
            env_success=env_success_any,
            success_rate=success_rate,
            delta_norm=delta_norm,
        )

        # --------------------------------------------------
        # 5) Termination
        # --------------------------------------------------
        terminated = bool(env_success_any)
        truncated = bool(self.current_step >= self.max_episode_steps)

        obs_out = self._build_obs()

        info0 = {}
        if last_infos and isinstance(last_infos, (list, tuple)):
            if isinstance(last_infos[0], dict):
                info0 = last_infos[0]

        info_out = {
            "goal_pred": goal_pred.copy(),
            "goal_delta": goal_delta.copy(),
            "goal_final": goal_final.copy(),
            "goal_delta_norm": delta_norm,
            **info0,
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

        return np.hstack([worker_obs, t_frac])

    # --------------------------------------------------
    # Manager reward
    # --------------------------------------------------
    def _compute_reward(
        self,
        *,
        goal_success: bool,
        env_success: bool,
        success_rate: float,
        delta_norm: float,
    ) -> float:
        r = -0.05                  # step cost
        r -= 0.05 * delta_norm     # penalize large corrections

        if goal_success:
            r += 1.0

        if env_success:
            r += 8.0

        r += 0.4 * success_rate

        return float(r)