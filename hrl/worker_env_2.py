from __future__ import annotations

from typing import Optional
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv

from hrl.utils import (
    safe_unit,
    predict_ball_trajectory,
    quat_to_paddle_normal,
    OWD
)


class TableTennisWorker(CustomEnv):
    """
    Goal-conditioned low-level controller (Phase 2 worker)

    State :
        - ball_vel (3)
        - paddle_normal (3)
        - ball_xy (2)
        - time (1)

    Goal :
        - target_ball_pos (3)
        - target_paddle_normal (2)
        - target_time_to_plane (1)
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # -------------------------------
        # Goal normalization
        # -------------------------------
        self.goal_center = np.array(
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.45], dtype=np.float32
        )
        self.goal_half_range = np.array(
            [0.6, 0.6, 0.5, 0.8, 0.5, 0.35], dtype=np.float32
        )

        self.state_dim = 9
        self.goal_dim = 6
        self.observation_dim = self.state_dim + self.goal_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # -------------------------------
        # Runtime
        # -------------------------------
        self.current_goal: Optional[np.ndarray] = None
        self.goal_start_time: Optional[float] = None
        self.prev_reach_err: Optional[float] = None
        self._prev_paddle_contact = False

        # -------------------------------
        # Success thresholds (RELAXED)
        # -------------------------------
        self.reach_thr = 0.12
        self.time_thr = 0.25
        self.paddle_ori_thr = 0.80
        self.success_bonus = 30.0

        self.max_time = 3.0

        # ==================================================
        # 🔧 NEW: Goal noise (ANNEALED, OFF by default)
        # ==================================================
        self.goal_noise_scale = 0.0  # <-- start deterministic

    # ------------------------------------------------
    # 🔧 NEW: External control (called from callback)
    # ------------------------------------------------
    def set_goal_noise_scale(self, scale: float):
        self.goal_noise_scale = float(np.clip(scale, 0.0, 0.2))

    # ------------------------------------------------
    # Prediction
    # ------------------------------------------------
    def _predict(self, obs_dict):
        pred_pos, paddle_ori_ideal = predict_ball_trajectory(
            ball_pos=obs_dict["ball_pos"],
            ball_vel=obs_dict["ball_vel"],
            paddle_pos=obs_dict["paddle_pos"],
        )
        n_ideal = quat_to_paddle_normal(paddle_ori_ideal)
        return pred_pos, n_ideal

    # ------------------------------------------------
    # Goal API
    # ------------------------------------------------
    def set_goal(self, goal_norm):
        goal_norm = np.clip(goal_norm, -1.0, 1.0)
        self.current_goal = self.goal_center + goal_norm * self.goal_half_range
        self.goal_start_time = float(self.env.unwrapped.obs_dict["time"])

    def _norm_goal(self, g):
        return np.clip((g - self.goal_center) / self.goal_half_range, -1.0, 1.0)

    def predict_goal_from_state(self, obs_dict):
        """
        Deterministic physics-based goal (manager uses this).
        NO noise here.
        """
        pred_pos, n_ideal = self._predict(obs_dict)

        err_x = obs_dict["paddle_pos"][0] - obs_dict["ball_pos"][0]
        vx = max(obs_dict["ball_vel"][0], 0.5)
        dt = np.clip(err_x / vx, 0.05, 1.5)

        nx, ny = self._pack_normal_xy(n_ideal)

        goal_phys = np.array(
            [pred_pos[0], pred_pos[1], pred_pos[2], nx, ny, dt],
            dtype=np.float32,
        )

        # 🔧 NEW: optional noise (annealed)
        if self.goal_noise_scale > 0.0:
            goal_phys[:3] += np.random.normal(
                0.0, self.goal_noise_scale, size=3
            )
            goal_phys[5] += np.random.normal(
                0.0, self.goal_noise_scale * 0.5
            )

        return self._norm_goal(goal_phys)

    # ------------------------------------------------
    # Gym API
    # ------------------------------------------------
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        obs_dict = self.env.unwrapped.obs_dict

        self._prev_paddle_contact = False
        goal = self.predict_goal_from_state(obs_dict)
        self.set_goal(goal)

        paddle_pos = obs_dict["paddle_pos"]
        self.prev_reach_err = np.linalg.norm(
            paddle_pos - self.current_goal[:3]
        )

        return self._build_obs(obs_dict), info

    def step(self, action):
        _, base_reward, terminated, truncated, info = super().step(action)
        obs_dict = info["obs_dict"]

        reward, success, logs = self._compute_reward(obs_dict)
        reward += 0.05 * base_reward

        info.update(logs)
        return self._build_obs(obs_dict), float(reward), terminated, truncated, info

    # ------------------------------------------------
    # Observation
    # ------------------------------------------------
    def _build_obs(self, obs_dict):
        paddle_n = quat_to_paddle_normal(obs_dict["paddle_ori"])
        paddle_n /= np.linalg.norm(paddle_n) + 1e-8

        state = np.hstack([
            obs_dict["ball_vel"] / 5.0,
            paddle_n,
            obs_dict["ball_pos"][:2] / 2.0,
            [obs_dict["time"] / self.max_time],
        ])

        obs = np.hstack([state, self.current_goal])
        return np.clip(obs, -3.0, 3.0)

    # ------------------------------------------------
    # Reward (GOAL-CENTRIC, NOT BALL-CENTRIC)
    # ------------------------------------------------
    def _compute_reward(self, obs_dict):
        paddle_pos = obs_dict["paddle_pos"]
        goal_pos = self.current_goal[:3]

        # --- 1. Progress to goal ---
        reach_err = np.linalg.norm(paddle_pos - goal_pos)
        reach_delta = self.prev_reach_err - reach_err
        self.prev_reach_err = reach_err

        reward = 2.0 * reach_delta

        # --- 2. Orientation ---
        paddle_n = quat_to_paddle_normal(obs_dict["paddle_ori"])
        paddle_n /= np.linalg.norm(paddle_n) + 1e-8
        goal_n = self._unpack_normal_xy(
            self.current_goal[3], self.current_goal[4]
        )

        cos_sim = float(np.dot(paddle_n, goal_n))
        reward += 1.0 * np.clip(cos_sim, 0.0, 1.0)

        # --- 3. Timing ---
        t_now = obs_dict["time"]
        t_target = self.goal_start_time + self.current_goal[5]
        time_err = abs(t_now - t_target)
        reward -= 0.5 * min(time_err, 1.0)

        # --- 4. Contact bonus ---
        touching = obs_dict["touching_info"][0] > 0.5
        if touching and not self._prev_paddle_contact:
            reward += 3.0
        self._prev_paddle_contact = touching

        # --- 5. Success (RELAXED) ---
        success = (
            reach_err < self.reach_thr
            and time_err < self.time_thr
            and cos_sim > self.paddle_ori_thr
        )

        if success:
            reward += self.success_bonus

        logs = {
            "reach_err": reach_err,
            "reach_delta": reach_delta,
            "cos_sim": cos_sim,
            "time_err": time_err,
            "is_goal_success": success,
        }

        return reward, success, logs

    # ------------------------------------------------
    # Helpers
    # ------------------------------------------------
    def _pack_normal_xy(self, n):
        if n[0] > 0:
            n = -n
        return float(n[0]), float(n[1])

    def _unpack_normal_xy(self, nx, ny):
        r2 = nx * nx + ny * ny
        if r2 > 0.999:
            s = 0.999 / np.sqrt(r2)
            nx *= s
            ny *= s
        nz = np.sqrt(max(1.0 - nx * nx - ny * ny, 0.0))
        return safe_unit(
            np.array([nx, ny, nz], dtype=np.float32),
            np.array(OWD),
        )