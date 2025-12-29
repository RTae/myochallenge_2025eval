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
)


class TableTennisWorker(CustomEnv):
    """
    Goal-conditioned low-level controller for table tennis.

    State (434):
        - time (1)
        - pelvis_pos (3)
        - body_qpos (58)
        - body_qvel (58)
        - ball_pos (3)
        - ball_vel (3)
        - paddle_pos (3)
        - paddle_vel (3)
        - paddle_ori (4)
        - paddle_ori_err (4)
        - reach_err (3)
        - palm_pos (3)
        - palm_err (3)
        - touching_info (6)
        - act (273)

    Goal (6):
        - target_ball_pos (3)
        - target_paddle_normal (2)
        - target_time_to_plane (1)

    Observation: 434 + 6 = 440
    """

    def __init__(self, config: Config):
        super().__init__(config)

        self.state_dim = 434
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
        self._prev_paddle_contact = False
        
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
        
        self.dt_min = 0.05
        self.dt_max = 1.50

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
        
    def get_progress(self):
        return float(self.progress)
        
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
        n = np.array([nx, ny, nz], dtype=np.float32)
        return safe_unit(n, np.array(OWD, dtype=np.float32))

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
    def set_goal(self, goal_phys: np.ndarray):
        self.current_goal = goal_phys
        self.goal_start_time = float(self.env.unwrapped.obs_dict["time"])
        
    def predict_goal_from_state(self, obs_dict):
        # --------------------------------------------------
        # 1) Predict ball–paddle intersection
        # --------------------------------------------------
        pred_pos, n_ideal = self._predict(obs_dict)
        # pred_pos: (x, y, z)
        # n_ideal : ideal paddle normal

        # --------------------------------------------------
        # 2) Time-to-contact
        # --------------------------------------------------
        dx = float(pred_pos[0] - obs_dict["ball_pos"][0])
        vx = float(obs_dict["ball_vel"][0])

        # Only compute time when ball is actually moving toward plane
        if abs(vx) > 1e-3 and (dx * vx) > 0.0:
            dt = abs(dx / vx)
        else:
            # Fallback: "far in time" instead of fake small dt
            dt = 1.5

        dt = float(np.clip(dt, 0.05, 1.5))

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

        return goal_phys
    
    def _build_obs(self, obs_dict):
        # Bound a time in each tral
        time = obs_dict["time"] - self.goal_start_time
        
        obs = np.concatenate([
            np.array([time], dtype=np.float32),

            obs_dict["pelvis_pos"],
            obs_dict["body_qpos"],
            obs_dict["body_qvel"],

            obs_dict["ball_pos"],
            obs_dict["ball_vel"],

            obs_dict["paddle_pos"],
            obs_dict["paddle_vel"],
            obs_dict["paddle_ori"],
            obs_dict["padde_ori_err"],

            obs_dict["reach_err"],
            obs_dict["palm_pos"],
            obs_dict["palm_err"],

            obs_dict["touching_info"],
            obs_dict["act"],

            self.current_goal,
        ], axis=0)

        return obs.astype(np.float32)

    # ==================================================
    # Gym API
    # ==================================================
    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed)
        obs_dict = self.env.unwrapped.obs_dict

        self._prev_paddle_contact = False

        goal = self.predict_goal_from_state(obs_dict)
        self.set_goal(goal)
        self.prev_reach_err = None

        self.prev_reach_err = np.linalg.norm(
            obs_dict["paddle_pos"] - self.current_goal[:3]
        )

        return self._build_obs(obs_dict), info

    def step(self, action):
        _, base_reward, terminated, truncated, info = super().step(action)
        obs_dict = info["obs_dict"]
        rwd_dict = info["rwd_dict"]

        reward, _, logs = self._compute_reward(obs_dict, rwd_dict)
        reward += 0.05 * base_reward

        info.update(logs)
        return self._build_obs(obs_dict), float(reward), terminated, truncated, info
    
    def _compute_reward(self, obs_dict, rwd_dict):
        # --------------------------------------------------
        # PRIMARY GOAL ERRORS
        # --------------------------------------------------
        paddle_pos = obs_dict["paddle_pos"]
        goal_pos = self.current_goal[:3]

        reach_err = float(np.linalg.norm(paddle_pos - goal_pos))
        # delta helps shape progress
        if self.prev_reach_err is None:
            self.prev_reach_err = reach_err
        reach_delta = self.prev_reach_err - reach_err
        self.prev_reach_err = reach_err

        # base shaping: encourage reduction in goal distance
        reward = 1.5 * np.clip(reach_delta, -0.05, 0.05)
        reward += 1.2 * np.exp(-4.0 * reach_err)

        # --------------------------------------------------
        # ORIENTATION TOWARD GOAL
        # --------------------------------------------------
        # (use dot product without env paddle_quat)
        paddle_n = quat_to_paddle_normal(obs_dict["paddle_ori"])
        paddle_n /= np.linalg.norm(paddle_n) + 1e-8
        goal_n = self._unpack_normal_xy(self.current_goal[3], self.current_goal[4])
        cos_sim = float(np.clip(np.dot(paddle_n, goal_n), -1.0, 1.0))
        ori_term = max(cos_sim, 0.0)

        # orientation helps only when somewhat close
        reward += 0.8 * ori_term * np.exp(-2.0 * reach_err)

        # --------------------------------------------------
        # TIMING
        # --------------------------------------------------
        t_now = float(obs_dict["time"])
        t_goal = self.goal_start_time + self.current_goal[5]
        time_err = abs(t_now - t_goal)

        reward += 0.5 * np.exp(-3.0 * time_err)
        reward -= 0.4 * np.tanh(time_err)

        # --------------------------------------------------
        # PALM–PADDLE DISTANCE (ANTI-THROW)
        # --------------------------------------------------
        # palm_dist is distance between palm and paddle (high -> weird posture)
        palm_dist = float(rwd_dict.get("palm_dist", 0.0))
        # penalize large palm-to-paddle gap
        reward -= 0.3 * palm_dist

        # --------------------------------------------------
        # TORSO UPRIGHT ENCOURAGEMENT
        # --------------------------------------------------
        # torso_up is high when torso is upright (0–1)
        # encourage a natural posture
        torso_up = float(rwd_dict.get("torso_up", 0.0))
        reward += 0.4 * torso_up

        # # --------------------------------------------------
        # # ACTION REGULARIZATION
        # # --------------------------------------------------
        # act_reg = float(rwd_dict.get("act_reg", 0.0))
        # reward += 0.6 * act_reg  # env-provided small control penalty

        # --------------------------------------------------
        # MOTION SMOOTHNESS PENALTY
        # --------------------------------------------------
        paddle_vel = obs_dict["paddle_vel"]
        v_norm = float(np.linalg.norm(paddle_vel))
        close_gate = np.exp(-6.0 * reach_err) * np.exp(-3.0 * time_err)
        reward -= close_gate * 0.1 * v_norm

        # --------------------------------------------------
        # STABILITY-GATED CONTACT
        # --------------------------------------------------
        touching = float(obs_dict["touching_info"][0]) > 0.5
        stable_contact = (v_norm < 0.40 and ori_term > 0.6)
        if touching and not self._prev_paddle_contact:
            reward += 2.0 * np.exp(-3.0 * time_err) if stable_contact else -1.0
        self._prev_paddle_contact = touching

        # --------------------------------------------------
        # SOFT SUCCESS SHAPING
        # --------------------------------------------------
        soft_success = (
            reach_err < 2.5 * self.reach_thr
            and time_err < 2.0 * self.time_thr
            and ori_term > self.paddle_ori_thr - 0.2
        )
        if soft_success:
            reward += 1.2

        # --------------------------------------------------
        # HARD SUCCESS
        # --------------------------------------------------
        success = (
            reach_err < self.reach_thr
            and time_err < self.time_thr
            and ori_term > self.paddle_ori_thr
        )
        hard_success = self.allow_hard_success and success
        if hard_success:
            reward += self.success_bonus

        # --------------------------------------------------
        # LOGS
        # --------------------------------------------------
        logs = {
            "reach_err": reach_err,
            "cos_sim": cos_sim,
            "time_err": time_err,
            "palm_dist": palm_dist,
            "torso_up": torso_up,
            "paddle_speed": v_norm,
            "is_goal_soft_success": float(soft_success),
            "is_goal_success": float(hard_success),
        }

        return float(reward), hard_success, logs