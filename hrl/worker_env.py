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

    Observation: 428 + 6 = 434
    """

    def __init__(self, config: Config, debug_draw: bool = False):
        super().__init__(config)
        
        self.debug_draw = debug_draw

        # ==================================================
        # Observation space
        # ==================================================
        self.state_dim = 428
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

        # ==================================================
        # Success thresholds (curriculum-controlled)
        # ==================================================
        self.reach_thr_base = 0.25
        self.reach_max_delta = 0.15
        self.time_thr_base = 0.35
        self.time_max_delta = 0.15
        self.paddle_ori_thr_base = 0.70
        self.paddle_ori_max_delta = 0.25

        self.reach_thr = self.reach_thr_base
        self.time_thr = self.time_thr_base
        self.paddle_ori_thr = self.paddle_ori_thr_base

        self.success_bonus = 10.0
        self.max_time = 3.0
        
        self.dt_min = 0.05
        self.dt_max = 1.50
    
    # =================================================
    # Debug
    # ==================================================
    
    def _draw_prediction_debug(self, pred_pos, n_ideal, dt):

        viewer = getattr(self.env.unwrapped, "viewer", None)
        if viewer is None:
            raise RuntimeError("Viewer not initialized for debug drawing.")

        # --------------------------------------------------
        # COLOR BY TIMING ERROR
        # dt > 0 : early
        # dt < 0 : late
        # --------------------------------------------------
        if dt >= 0.0:
            # green → yellow as it gets very early
            alpha = np.clip(dt / 0.3, 0.0, 1.0)
            color = np.array([
                0.2 + 0.6 * alpha,   # R
                1.0,                 # G
                0.2,                 # B
                0.8
            ])
        else:
            # red → dark red as lateness increases
            alpha = np.clip(-dt / 0.3, 0.0, 1.0)
            color = np.array([
                1.0,                 # R
                0.2 * (1 - alpha),   # G
                0.2 * (1 - alpha),   # B
                0.8
            ])

        # --------------------------------------------------
        # Predicted contact circle
        # --------------------------------------------------
        viewer.add_marker(
            pos=pred_pos,
            size=np.array([0.025, 0.025, 0.001]),
            rgba=color,
            type=viewer.mjviewer.const.GEOM_CYLINDER,
        )

        # --------------------------------------------------
        # Predicted normal direction
        # --------------------------------------------------
        end = pred_pos + 0.15 * n_ideal
        viewer.add_marker(
            pos=(pred_pos + end) / 2,
            size=np.array([0.003, 0.003, 0.15]),
            rgba=color * np.array([0.7, 0.7, 0.7, 1.0]),
            type=viewer.mjviewer.const.GEOM_CAPSULE,
            mat=np.eye(3),
        )

    # ==================================================
    # Curriculum hooks (called by callback)
    # ==================================================
    def set_goal_noise_scale(self, scale: float):
        self.goal_noise_scale = float(np.clip(scale, 0.0, 0.2))

    def set_progress(self, progress: float):
        p = float(np.clip(progress, 0.0, 1.0))

        self.reach_thr = self.reach_thr_base - self.reach_max_delta * p
        self.time_thr  = self.time_thr_base  - self.time_max_delta  * p

        # Angle precision ramps late
        angle_p = p ** 1.5
        self.paddle_ori_thr = (
            self.paddle_ori_thr_base + self.paddle_ori_max_delta * angle_p
        )
        
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
        # Predict ball–paddle intersection
        # --------------------------------------------------
        pred_pos, n_ideal = self._predict(obs_dict)
        # pred_pos: (x, y, z)
        # n_ideal : ideal paddle normal

        # --------------------------------------------------
        # Time-to-contact
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
        # Orientation target (packed)
        # --------------------------------------------------
        nx, ny = self._pack_normal_xy(n_ideal)

        # --------------------------------------------------
        # Assemble PHYSICAL goal
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
        # Curriculum noise
        # --------------------------------------------------
        if self.goal_noise_scale > 0.0:
            goal_phys[:3] += np.random.normal(
                0.0, self.goal_noise_scale, size=3
            )
            goal_phys[5] += np.random.normal(
                0.0, self.goal_noise_scale * 0.5
            )

        return goal_phys
        
    def _flat(self, x):
        return np.asarray(x, dtype=np.float32).reshape(-1)

    def _build_obs(self, obs_dict):
        # time relative to goal start
        time = np.array(
            [obs_dict["time"] - self.goal_start_time],
            dtype=np.float32,
        )

        obs = np.concatenate([
            self._flat(time),

            self._flat(obs_dict["pelvis_pos"]),
            self._flat(obs_dict["body_qpos"]),
            self._flat(obs_dict["body_qvel"]),

            self._flat(obs_dict["ball_pos"]),
            self._flat(obs_dict["ball_vel"]),

            self._flat(obs_dict["paddle_pos"]),
            self._flat(obs_dict["paddle_vel"]),
            self._flat(obs_dict["paddle_ori"]),
            self._flat(obs_dict["padde_ori_err"]),

            self._flat(obs_dict["reach_err"]),
            self._flat(obs_dict["palm_pos"]),
            self._flat(obs_dict["palm_err"]),

            self._flat(obs_dict["touching_info"]),
            self._flat(obs_dict["act"]),

            self._flat(self.current_goal),
        ], axis=0)
        
        assert obs.shape == (self.observation_dim,), f"Invalid observation shape {obs.shape}, expected {self.observation_dim}"

        return obs

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
        
        # --------------------------------------------------
        # 6) Debug drawing
        # --------------------------------------------------
        if self.debug_draw:
            goal_pos = self.current_goal[:3]
            goal_n = self._unpack_normal_xy(
                self.current_goal[3], self.current_goal[4]
            )
            goal_time = self.goal_start_time + self.current_goal[5]
            dt = goal_time - float(obs_dict["time"])
            self._draw_prediction_debug(goal_pos, goal_n, dt)
        
        return self._build_obs(obs_dict), float(reward), terminated, truncated, info
    
    def _compute_reward(self, obs_dict, rwd_dict):
        # ==================================================
        # PRIMARY GOAL ERRORS
        # ==================================================
        paddle_pos = obs_dict["paddle_pos"]
        goal_pos = self.current_goal[:3]

        reach_err = float(np.linalg.norm(paddle_pos - goal_pos))

        if self.prev_reach_err is None:
            self.prev_reach_err = reach_err
        reach_delta = self.prev_reach_err - reach_err
        self.prev_reach_err = reach_err

        reward = 0.0

        # --------------------------------------------------
        # FAST APPROACH SHAPING
        # --------------------------------------------------
        # Encourage reducing distance quickly
        reward += 2.0 * np.clip(reach_delta, -0.1, 0.1)
        reward += 1.2 * np.exp(-4.0 * reach_err)

        # Encourage speed only when far
        if reach_err > 0.25:
            reward += 0.6 * np.linalg.norm(obs_dict["paddle_vel"])

        # ==================================================
        # ORIENTATION TOWARD GOAL
        # ==================================================
        paddle_n = quat_to_paddle_normal(obs_dict["paddle_ori"])
        paddle_n /= np.linalg.norm(paddle_n) + 1e-8

        goal_n = self._unpack_normal_xy(
            self.current_goal[3], self.current_goal[4]
        )

        cos_sim = float(np.clip(np.dot(paddle_n, goal_n), -1.0, 1.0))
        ori_term = max(cos_sim, 0.0)

        # Orientation matters more when close
        reward += 0.8 * ori_term * np.exp(-2.0 * reach_err)

        # ==================================================
        # TIMING
        # ==================================================
        t_now = float(obs_dict["time"])
        t_goal = self.goal_start_time + self.current_goal[5]

        dt = t_goal - t_now   # >0 early, <0 late
        time_err = abs(dt)    # for logging only

        # Sharp peak near correct hit time
        reward += 0.6 * np.exp(-6.0 * abs(dt))

        # Penalize lateness strongly
        if dt < 0.0:
            reward -= 1.5 * (abs(dt) ** 1.5)

        # Mild penalty for being *too* early (hovering)
        elif dt > 0.25:
            reward -= 0.1 * (dt - 0.25)

        # ==================================================
        # PALM–PADDLE DISTANCE (ANTI-THROW)
        # ==================================================
        palm_dist = float(rwd_dict.get("palm_dist", 0.0))
        reward -= 0.3 * palm_dist

        # ==================================================
        # TORSO UPRIGHT ENCOURAGEMENT
        # ==================================================
        torso_up = float(rwd_dict.get("torso_up", 0.0))
        reward += 0.4 * torso_up

        # ==================================================
        # MOTION SMOOTHNESS (ONLY NEAR CONTACT WINDOW)
        # ==================================================
        paddle_vel = obs_dict["paddle_vel"]
        v_norm = float(np.linalg.norm(paddle_vel))

        if abs(dt) < 0.2:
            reward -= 0.1 * v_norm

        # ==================================================
        # CONTACT EVENTS
        # ==================================================
        touching = float(obs_dict["touching_info"][0]) > 0.5

        # -----------------------------
        # Stable contact reward
        # -----------------------------
        stable_contact = (v_norm < 0.6 and ori_term > 0.6)

        if touching and not self._prev_paddle_contact:
            if dt >= -0.05 and stable_contact:
                reward += 2.0 * np.exp(-4.0 * abs(dt))
            else:
                reward -= 1.0

        # -----------------------------
        # Contact quality bonus
        # -----------------------------
        if touching and not self._prev_paddle_contact:
            align_bonus = ori_term ** 2
            time_bonus = np.exp(-4.0 * abs(dt)) if dt >= -0.05 else 0.0

            if v_norm < 0.6 and align_bonus > 0.5:
                reward += 2.5 * align_bonus * time_bonus

        # ==================================================
        # ENV SUCCESS
        # ==================================================
        env_solved = bool(rwd_dict.get("solved", False))

        if (
            env_solved
            and touching
            and not self._prev_paddle_contact
            and 0.0 <= dt <= self.time_thr
            and ori_term > self.paddle_ori_thr
            and v_norm < 0.6
        ):
            reward += 6.0  # sparse table-tennis success bonus

        self._prev_paddle_contact = touching

        # ==================================================
        # GOAL SUCCESS
        # ==================================================
        # Reach within threshold, at correct time, good orientation
        success = (
            reach_err < self.reach_thr
            and 0.0 <= dt <= self.time_thr
            and ori_term > self.paddle_ori_thr
        )

        if success:
            reward += self.success_bonus

        # ==================================================
        # LOGS
        # ==================================================
        logs = {
            "reach_err": reach_err,
            "reach_delta": reach_delta,
            "cos_sim": cos_sim,
            "ori_term": ori_term,
            "time_err": time_err,
            "dt": dt,
            "palm_dist": palm_dist,
            "torso_up": torso_up,
            "paddle_speed": v_norm,
            "env_solved": float(env_solved),
            "is_goal_success": float(success),
        }

        return float(reward), success, logs