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
    OWD,
    as_1d
)


class TableTennisWorker(CustomEnv):
    """
    Goal-conditioned low-level controller (Phase 2 worker)

    State (18):
        - ball_vel (3)
        - paddle_normal (3)
        - paddle_vel (3)
        - rel_goal_pos (3)
        - ball_xy (2)
        - time_frac (1)
        - time_to_goal (1)
        - paddle_touch (1)
        - impulse_obs (1)

    Goal (6):
        - target_ball_pos (3)
        - target_paddle_normal (2)
        - target_time_to_plane (1)

    Observation: 18 + 6 = 24
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # ==================================================
        # Goal normalization
        # ==================================================
        self.goal_center = np.array(
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.45], dtype=np.float32
        )
        self.goal_half_range = np.array(
            [0.6, 0.6, 0.5, 0.8, 0.5, 0.35], dtype=np.float32
        )

        self.state_dim = 18
        self.goal_dim = 6
        self.observation_dim = self.state_dim + self.goal_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # ==================================================
        # Runtime
        # ==================================================
        self.current_goal: Optional[np.ndarray] = None
        self.goal_start_time: Optional[float] = None

        self.prev_reach_err: Optional[float] = None
        self.prev_paddle_vel = None
        self.prev_paddle_vel_obs = None
        self._prev_paddle_contact = False
        
        # ==================================================
        # Goal noise 
        # ==================================================
        self.goal_noise_scale = 0.0 
        self.progress = 0.0

        # ==================================================
        # Curriculum state
        # ==================================================
        self.goal_noise_scale = 0.0
        self.progress = 0.0
        self.allow_hard_success = False

        # ==================================================
        # Success thresholds (curriculum-controlled)
        # ==================================================
        self.reach_thr_base = 0.25
        self.time_thr_base = 0.40
        self.paddle_ori_thr_base = 0.65

        self.reach_thr = self.reach_thr_base
        self.time_thr = self.time_thr_base
        self.paddle_ori_thr = self.paddle_ori_thr_base

        self.success_bonus = 10.0
        self.max_time = 3.0

    # ==================================================
    # Curriculum hooks (called by callback)
    # ==================================================
    def set_goal_noise_scale(self, scale: float):
        self.goal_noise_scale = float(np.clip(scale, 0.0, 0.2))

    def set_progress(self, progress: float):
        self.progress = float(np.clip(progress, 0.0, 1.0))
        self.reach_thr = self.reach_thr_base - 0.10 * progress
        self.time_thr = self.time_thr_base - 0.20 * progress
        self.paddle_ori_thr = self.paddle_ori_thr_base + 0.25 * progress

    def set_allow_hard_success(self, flag: bool):
        self.allow_hard_success = bool(flag)

    # ==================================================
    # Prediction
    # ==================================================
    def _predict(self, obs_dict):
        pred_pos, paddle_ori_ideal = predict_ball_trajectory(
            ball_pos=obs_dict["ball_pos"],
            ball_vel=obs_dict["ball_vel"],
            paddle_pos=obs_dict["paddle_pos"],
        )
        n_ideal = quat_to_paddle_normal(paddle_ori_ideal)
        return pred_pos, n_ideal

    # ==================================================
    # Goal API
    # ==================================================
    def set_goal(self, goal_norm):
        goal_norm = np.clip(goal_norm, -1.0, 1.0)
        self.current_goal = self.goal_center + goal_norm * self.goal_half_range
        self.goal_start_time = float(self.env.unwrapped.obs_dict["time"])

    def _norm_goal(self, g):
        return np.clip((g - self.goal_center) / self.goal_half_range, -1.0, 1.0)

    def predict_goal_from_state(self, obs_dict):
        # --------------------------------------------------
        # 1) Predict ball–paddle intersection
        # --------------------------------------------------
        pred_pos, n_ideal = self._predict(obs_dict)
        # pred_pos: (x, y, z) where ball should be hit
        # n_ideal : ideal paddle normal at impact

        # --------------------------------------------------
        # 2) Time-to-contact
        # --------------------------------------------------
        # Distance from current ball position to predicted contact plane
        dx = pred_pos[0] - obs_dict["ball_pos"][0]

        # Robust x-velocity (avoid division explosion)
        vx = obs_dict["ball_vel"][0]
        vx = np.sign(vx) * max(abs(vx), 0.5)

        # Time until ball reaches predicted x-plane
        dt = dx / vx
        dt = np.clip(dt, 0.05, 1.5)

        # --------------------------------------------------
        # 3) Orientation target (packed)
        # --------------------------------------------------
        nx, ny = self._pack_normal_xy(n_ideal)

        # --------------------------------------------------
        # 4) Assemble PHYSICAL goal
        # --------------------------------------------------
        goal_phys = np.array(
            [
                pred_pos[0],  # x_hit
                pred_pos[1],  # y_hit
                pred_pos[2],  # z_hit
                nx,           # paddle normal x
                ny,           # paddle normal y
                dt,           # time-to-plane
            ],
            dtype=np.float32,
        )

        # --------------------------------------------------
        # 5) Curriculum noise (optional, SAFE)
        # --------------------------------------------------
        if self.goal_noise_scale > 0.0:
            goal_phys[:3] += np.random.normal(
                0.0, self.goal_noise_scale, size=3
            )
            goal_phys[5] += np.random.normal(
                0.0, self.goal_noise_scale * 0.5
            )

        # --------------------------------------------------
        # 6) Normalize to [-1, 1] goal space
        # --------------------------------------------------
        return self._norm_goal(goal_phys)
    
    def get_progress(self):
        return float(self.progress)

    # ==================================================
    # Gym API
    # ==================================================
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        obs_dict = self.env.unwrapped.obs_dict

        self._prev_paddle_contact = False
        self.prev_paddle_vel = np.zeros(3, dtype=np.float32)
        self.prev_paddle_vel_obs = None

        goal = self.predict_goal_from_state(obs_dict)
        self.set_goal(goal)

        self.prev_reach_err = np.linalg.norm(
            obs_dict["paddle_pos"] - self.current_goal[:3]
        )

        return self._build_obs(obs_dict), info

    def step(self, action):
        _, base_reward, terminated, truncated, info = super().step(action)
        obs_dict = info["obs_dict"]

        reward, success, logs = self._compute_reward(obs_dict)
        reward += 0.05 * base_reward

        info.update(logs)
        return self._build_obs(obs_dict), float(reward), terminated, truncated, info
    
    def _build_obs(self, obs_dict):
        paddle_n = quat_to_paddle_normal(obs_dict["paddle_ori"])
        paddle_n /= np.linalg.norm(paddle_n) + 1e-8

        ball_vel = obs_dict["ball_vel"] / 5.0
        paddle_vel = obs_dict["paddle_vel"] / 5.0
        ball_xy = obs_dict["ball_pos"][:2] / 2.0

        rel_goal_pos = (self.current_goal[:3] - obs_dict["paddle_pos"]) / 0.5
        rel_goal_pos = np.clip(rel_goal_pos, -1.0, 1.0)

        time_frac = np.array([obs_dict["time"] / self.max_time], dtype=np.float32)

        time_to_goal = (
            (self.goal_start_time + self.current_goal[5]) - obs_dict["time"]
        )
        time_to_goal = np.clip(time_to_goal / self.max_time, -1.0, 1.0)
        time_to_goal = np.asarray(time_to_goal, dtype=np.float32).reshape(1)

        paddle_touch = np.array(
            [float(obs_dict["touching_info"][0])], dtype=np.float32
        )

        vel_delta = paddle_vel - (
            self.prev_paddle_vel_obs if self.prev_paddle_vel_obs is not None else paddle_vel
        )
        impulse_obs = np.array([np.tanh(np.linalg.norm(vel_delta))], dtype=np.float32)
        self.prev_paddle_vel_obs = paddle_vel

        state = np.concatenate(
            [
                as_1d(ball_vel),
                as_1d(paddle_n),
                as_1d(paddle_vel),
                as_1d(rel_goal_pos),
                as_1d(ball_xy),
                as_1d(time_frac),
                as_1d(time_to_goal),
                as_1d(paddle_touch),
                as_1d(impulse_obs),
            ],
            axis=0,
        )

        obs = np.concatenate([state, self.current_goal.astype(np.float32)], axis=0)
        
        return np.clip(obs, -3.0, 3.0)

    # ==================================================
    # Reward
    # ==================================================
    def _compute_reward(self, obs_dict):
        paddle_pos = obs_dict["paddle_pos"]
        goal_pos = self.current_goal[:3]

        # ==================================================
        # Reach error
        # ==================================================
        reach_err = np.linalg.norm(paddle_pos - goal_pos)
        reach_delta = self.prev_reach_err - reach_err
        self.prev_reach_err = reach_err

        lateral_err = np.linalg.norm((paddle_pos - goal_pos)[1:])

        # Base reach shaping (dominant early signal)
        reward = 2.0 * np.clip(reach_delta, -0.05, 0.05)
        reward += 0.6 * np.exp(-3.0 * reach_err) * np.exp(-2.0 * lateral_err)

        reach_w = np.exp(-3.0 * reach_err)

        # ==================================================
        # Velocity & impulse (ANTI-THROW)
        # ==================================================
        paddle_vel = obs_dict["paddle_vel"]
        vel_delta = paddle_vel - self.prev_paddle_vel
        impulse = np.linalg.norm(vel_delta)
        safe_impulse = np.clip(impulse, 0.0, 2.0)
        self.prev_paddle_vel = paddle_vel

        # ==================================================
        # Timing
        # ==================================================
        t_now = obs_dict["time"]
        t_target = self.goal_start_time + self.current_goal[5]
        time_err = abs(t_now - t_target)

        # ==================================================
        # Orientation
        # ==================================================
        paddle_n = quat_to_paddle_normal(obs_dict["paddle_ori"])
        paddle_n /= np.linalg.norm(paddle_n) + 1e-8
        goal_n = self._unpack_normal_xy(
            self.current_goal[3], self.current_goal[4]
        )

        cos_sim = float(np.dot(paddle_n, goal_n))
        ori_pos = np.clip(cos_sim, 0.0, 1.0)

        # ==================================================
        # Damping (distance-aware, orientation-safe)
        # ==================================================
        vel_gate = np.exp(-2.0 * reach_err) * np.exp(-2.0 * time_err)

        reward -= vel_gate * (
            0.06 * np.linalg.norm(paddle_vel) +
            0.15 * safe_impulse
        )

        reward += 0.25 * vel_gate * np.exp(-safe_impulse ** 2)

        # ==================================================
        # Orientation only matters when close
        # ==================================================
        close_gate = np.exp(-6.0 * reach_err)

        # Reward correct alignment near goal
        reward += 1.2 * close_gate * ori_pos

        # Penalize wrong orientation ONLY when very close
        reward -= 0.6 * close_gate * (1.0 - ori_pos) ** 2

        # Gentle late-stage precision push
        if reach_err < 0.12:
            reward += 0.3 * (ori_pos ** 2)

        # ==================================================
        # Timing shaping (smooth, symmetric)
        # ==================================================
        reward += 0.5 * np.exp(-4.0 * time_err)
        reward -= 0.45 * np.tanh(time_err)

        # ==================================================
        # Contact reward
        # ==================================================
        touching = obs_dict["touching_info"][0] > 0.5
        if touching and not self._prev_paddle_contact:
            reward += 3.0 * np.exp(-3.0 * time_err)
        self._prev_paddle_contact = touching

        # ==================================================
        # Success conditions
        # ==================================================
        soft_success = (
            reach_err < 2.5 * self.reach_thr
            and time_err < 2.0 * self.time_thr
            and cos_sim > self.paddle_ori_thr - 0.2
            and safe_impulse < 1.8
        )

        if soft_success:
            reward += 2.0 + np.clip(reach_delta, 0.0, 0.05)

        success = (
            reach_err < self.reach_thr
            and time_err < self.time_thr
            and cos_sim > self.paddle_ori_thr
            and safe_impulse < 1.5
        )

        hard_success = self.allow_hard_success and success
        if hard_success:
            reward += self.success_bonus

        reward = float(np.clip(reward, -5.0, 20.0))

        # ==================================================
        # Logs
        # ==================================================
        logs = {
            "reach_err": reach_err,
            "cos_sim": cos_sim,
            "ori_pos": ori_pos,
            "reach_ori_product": float(close_gate * ori_pos),
            "time_err": time_err,
            "impulse": impulse,
            "is_goal_soft_success": float(soft_success),
            "is_goal_success": float(hard_success),
        }

        return reward, hard_success, logs

    # ==================================================
    # Helpers
    # ==================================================
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