from __future__ import annotations

from typing import Optional
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv

from hrl.utils import (
    safe_unit,
    reflect_normal,
    predict_ball_analytic,
    quat_to_paddle_normal
)

class TableTennisWorker(CustomEnv):
    """
    Low-level worker (muscle controller).

    Observation = state (12) + goal (6) = 18
    
    State: 
        - reach_err (3)
        - ball_vel (3)
        - paddle_normal (3)
        - ball_xy (2)
        - time (1)
        
    Goal:
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
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.475], dtype=np.float32
        )
        self.goal_half_range = np.array(
            [0.6, 0.6, 0.5, 0.8, 0.5, 0.325], dtype=np.float32
        )

        self.state_dim = 12
        self.goal_dim = 6
        self.observation_dim = self.state_dim + self.goal_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # Runtime
        self.current_goal: Optional[np.ndarray] = None
        self.goal_start_time: Optional[float] = None
        self.prev_reach_err: Optional[float] = None
        self._prev_paddle_contact = False

        # Thresholds
        self.reach_thr = 0.25
        self.vel_thr = 1.2
        self.time_thr = 0.35
        self.success_bonus = 40.0

        # Noise
        self.pos_noise_scale = np.array([0.08, 0.08, 0.04], dtype=np.float32)
        self.normal_noise_scale = 0.15
        self.time_noise_scale = 0.08

        self.dt_min = 0.05
        self.dt_max = 1.5

    # ------------------------------------------------
    # Time-to-plane
    # ------------------------------------------------
    def _compute_dt(self, ball_pos, ball_vel, paddle_pos) -> float:
        err_x = paddle_pos[0] - ball_pos[0]
        vx = ball_vel[0]
        if err_x <= 0 or vx <= 1e-3:
            return self.dt_min
        return float(np.clip(err_x / vx, self.dt_min, self.dt_max))

    # ------------------------------------------------
    # Goal helpers
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
        nz = -np.sqrt(max(1.0 - nx * nx - ny * ny, 0.0))
        n = np.array([nx, ny, nz], dtype=np.float32)
        return safe_unit(n, np.array([-1.0, 0.0, 0.0]))

    # ------------------------------------------------
    # Prediction wrapper (FAST & SAFE)
    # ------------------------------------------------
    def _predict(self, obs_dict):
        pred_pos, n_ideal, _ = predict_ball_analytic(
            sim=self.sim,
            id_info=self.id_info,
            ball_pos=obs_dict["ball_pos"],
            ball_vel=obs_dict["ball_vel"],
            paddle_pos=obs_dict["paddle_pos"],
        )
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

    def _sample_goal(self, obs_dict):
        pred_pos, n_ideal = self._predict(obs_dict)
        dt = self._compute_dt(
            obs_dict["ball_pos"],
            obs_dict["ball_vel"],
            obs_dict["paddle_pos"],
        )

        pos_noise = np.random.normal(0, self.pos_noise_scale)
        n_noise = np.random.normal(0, self.normal_noise_scale, size=3)
        time_noise = np.random.normal(0, self.time_noise_scale)

        n = safe_unit(n_ideal + n_noise, np.array([-1.0, 0.0, 0.0]))
        nx, ny = self._pack_normal_xy(n)

        goal_phys = np.array(
            [
                pred_pos[0] + pos_noise[0],
                pred_pos[1] + pos_noise[1],
                pred_pos[2] + pos_noise[2],
                nx,
                ny,
                np.clip(dt + time_noise, self.dt_min, self.dt_max),
            ],
            dtype=np.float32,
        )

        goal_phys = np.clip(
            goal_phys,
            self.goal_center - self.goal_half_range,
            self.goal_center + self.goal_half_range,
        )
        return self._norm_goal(goal_phys)

    def predict_goal_from_state(self, obs_dict):
        pred_pos, n_ideal = self._predict(obs_dict)
        dt = self._compute_dt(
            obs_dict["ball_pos"],
            obs_dict["ball_vel"],
            obs_dict["paddle_pos"],
        )
        nx, ny = self._pack_normal_xy(n_ideal)
        goal_phys = np.array(
            [pred_pos[0], pred_pos[1], pred_pos[2], nx, ny, dt],
            dtype=np.float32,
        )
        return self._norm_goal(goal_phys)

    # ------------------------------------------------
    # Gym API
    # ------------------------------------------------
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        obs_dict = self.env.unwrapped.obs_dict

        self._prev_paddle_contact = False
        self.set_goal(self._sample_goal(obs_dict))
        self.prev_reach_err = float(np.linalg.norm(obs_dict["reach_err"]))

        return self._build_obs(obs_dict), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = super().step(action)
        obs_dict = info["obs_dict"]

        hit = self._detect_paddle_hit(obs_dict)
        reward, success, reach_err, vel_norm, time_err, cos_sim = self._compute_reward(obs_dict, hit)

        reward += 0.05 * base_reward
        self.prev_reach_err = reach_err

        info.update(
            {
                "worker/reach_err": reach_err,
                "worker/vel_norm": vel_norm,
                "worker/time_err": time_err,
                "worker/cos_sim": cos_sim,
                "worker/dt_target": float(self.current_goal[5]),
                "worker/is_goal_success": bool(success),
                "worker/is_paddle_hit": bool(hit),
            }
        )

        return self._build_obs(obs_dict), float(reward), terminated, truncated, info

    # ------------------------------------------------
    # Observation
    # ------------------------------------------------
    def _build_obs(self, obs_dict):
        reach_err = np.asarray(obs_dict["reach_err"], dtype=np.float32)
        ball_vel = np.asarray(obs_dict["ball_vel"], dtype=np.float32)
        paddle_n = quat_to_paddle_normal(obs_dict["paddle_ori"])
        paddle_n /= np.linalg.norm(paddle_n) + 1e-8
        ball_xy = np.asarray(obs_dict["ball_pos"][:2], dtype=np.float32)
        t = np.asarray([float(obs_dict["time"])], dtype=np.float32)

        state = np.hstack([reach_err, ball_vel, paddle_n, ball_xy, t])
        obs = np.hstack([state, self.current_goal])
        return np.clip(obs, -5.0, 5.0)

    # ------------------------------------------------
    # Reward
    # ------------------------------------------------
    def _compute_reward(self, obs_dict, hit):
        reach_err = float(np.linalg.norm(obs_dict["reach_err"]))
        vel_norm = float(np.linalg.norm(obs_dict["paddle_vel"]))
        t_now = float(obs_dict["time"])

        target_time = self.goal_start_time + self.current_goal[5]
        time_err = min(abs(t_now - target_time), 1.0)

        paddle_n = quat_to_paddle_normal(obs_dict["paddle_ori"])
        paddle_n /= np.linalg.norm(paddle_n) + 1e-8

        # Example:
        # Desired paddle normal (goal_n) is facing the ball:
        #     goal_n = [-1, 0, 0]
        #
        # If the paddle is perfectly aligned:
        #     paddle_n = [-1, 0, 0]
        #     cos_sim = dot(paddle_n, goal_n) = 1.0   (perfect alignment)
        #
        # If the paddle is slightly tilted upward:
        #     paddle_n ≈ [-0.97, 0.00, 0.24]
        #     cos_sim ≈ 0.97  (still good)
        #
        # If the paddle is orthogonal:
        #     paddle_n = [0, 1, 0]
        #     cos_sim = 0.0   (bad orientation)
        #
        # If the paddle faces the wrong way:
        #     paddle_n = [1, 0, 0]
        #     cos_sim = -1.0  (completely wrong)
        #
        # We reward cos_sim → 1 when paddle orientation matches the desired reflection normal.
        goal_n = self._unpack_normal_xy(self.current_goal[3], self.current_goal[4])
        cos_sim = float(np.clip(np.dot(paddle_n, goal_n), -1.0, 1.0))

        align_w = 0.6 * np.exp(-3.0 * reach_err)

        reward = (
            1.2 * np.exp(-2.0 * reach_err)
            + 0.8 * (1.0 - np.clip(reach_err, 0, 2))
            + align_w * cos_sim
            - 0.2 * vel_norm
            - 0.3 * time_err
        )

        success = (
            reach_err < self.reach_thr
            and vel_norm < self.vel_thr
            and time_err < self.time_thr
            and cos_sim > 0.9
        )

        if success:
            reward += self.success_bonus
        if hit:
            reward += 0.3

        return reward, success, reach_err, vel_norm, time_err, cos_sim

    # ------------------------------------------------
    # Hit detection
    # ------------------------------------------------
    def _detect_paddle_hit(self, obs_dict):
        touching = np.asarray(obs_dict["touching_info"], dtype=np.float32)
        hit = bool(touching[0] > 0.5 and not self._prev_paddle_contact)
        self._prev_paddle_contact = touching[0] > 0.5
        return hit