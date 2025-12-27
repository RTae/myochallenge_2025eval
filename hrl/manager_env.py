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
    High-level HRL manager (Δ-goal in *normalized goal space*).

    Observation:
        worker_obs (worker_obs_dim) + normalized time (1)

    Action:
        a ∈ [-1, 1]^6
        → interpreted as Δgoal_norm (small correction in normalized space)
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
        
        t_progress = float(np.mean(self.worker_env.env_method("get_progress")))
        t_noice = self.worker_env.env_method("goal_noise_scale")
        assert (
            t_progress >= 1.0
            and not any(t_noice)
        ), f"Manager requires fully unlocked worker (progress=1, noise=0), got progress={t_progress}, noise={t_noice}"

        self.decision_interval = int(decision_interval)
        self.max_episode_steps = int(max_episode_steps)

        self.worker_obs_dim = int(self.worker_env.observation_space.shape[0])
        self.observation_dim = self.worker_obs_dim + 1

        assert self.observation_dim == int(self.worker_env.observation_space.shape[0]) + 1, \
            "Obs dim mismatch"

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # Action: Δgoal_norm
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.delta_min = np.array(
            [0.02, 0.02, 0.015, 0.02, 0.02, 0.04],
            dtype=np.float32,
        )

        self.delta_max = np.array(
            [0.06, 0.06, 0.05, 0.06, 0.06, 0.12],
            dtype=np.float32,
        )

        self.current_step = 0
        self._worker_obs = None
        self.success_buffer = deque(maxlen=int(success_buffer_len))

    @property
    def sim(self):
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
        # --------------------------------------------------
        # 1) Build goal correction (normalized space)
        # --------------------------------------------------
        raw_delta = np.clip(
            np.asarray(action, dtype=np.float32).reshape(6,),
            -1.0,
            1.0,
        )
        progress = float(
            np.mean(
                self.worker_env.env_method("get_progress")
            )
        )

        # quadratic schedule
        scale = self.delta_min + (progress ** 2) * (self.delta_max - self.delta_min)
        scale[2] *= 0.7   # z always stricter
        scale[5] *= 1.3   # dt slightly freer

        goal_delta = raw_delta * scale

        # current obs_dict from env0 is fine (they are independent per env though)
        # For multi-env training you may want per-env goal_pred, but keep it simple first.
        worker0 = self.worker_env.envs[0]
        obs_dict0 = worker0.env.unwrapped.obs_dict

        # Physics-based predicted goal (normalized)
        goal_pred = worker0.predict_goal_from_state(obs_dict0)

        # Final normalized goal
        goal_final = np.clip(goal_pred + goal_delta, -1.0, 1.0)

        self.worker_env.env_method("set_goal", goal_final)

        # --------------------------------------------------
        # 2) Run frozen worker for decision_interval steps
        # --------------------------------------------------
        goal_success_any = False
        env_success_any = False
        last_infos = None
        last_dones = None

        for _ in range(self.decision_interval):
            obs_batch = np.asarray(self._worker_obs, dtype=np.float32)  # (n_envs, obs_dim)

            worker_actions, _ = self.worker_model.predict(
                obs_batch, deterministic=True
            )

            obs, _, dones, infos = self.worker_env.step(worker_actions)
            self._worker_obs = obs
            self.current_step += 1

            last_infos = infos
            last_dones = dones

            # aggregate success across envs
            for info in infos:
                goal_success_any |= bool(info.get("is_goal_success", 0.0))
                env_success_any |= bool(info.get("is_success", 0.0))

            if np.any(dones):
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

        info0 = {}
        if last_infos and isinstance(last_infos, (list, tuple)) and len(last_infos) > 0:
            if isinstance(last_infos[0], dict):
                info0 = last_infos[0]

        info_out = {
            "goal_delta_norm": delta_norm,
            "goal_delta": goal_delta.copy(),
            "goal_final": goal_final.copy(),
            **info0,
        }

        return obs_out, float(reward), terminated, truncated, info_out

    def _build_obs(self) -> np.ndarray:
        worker_obs = np.asarray(self._worker_obs[0], dtype=np.float32).reshape(self.worker_obs_dim)

        t_frac = np.array(
            [self.current_step / max(1, self.max_episode_steps)],
            dtype=np.float32,
        )

        return np.hstack([worker_obs, t_frac])

    def _reward_manager(
        self,
        *,
        goal_success: bool,
        env_success: bool,
        success_rate: float,
        delta_norm: float,
    ) -> float:
        r = -0.05

        if env_success:
            r += 8.0

        r += 1.0 * float(goal_success)
        r += 0.4 * success_rate
        
        delta_norm = min(delta_norm, 0.2)
        r -= 0.2 * np.tanh(delta_norm / 0.1)
        
        return float(r)