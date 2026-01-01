from __future__ import annotations

from typing import Optional, Tuple, Dict
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv

from hrl.utils import predict_ball_trajectory


class TableTennisWorker(CustomEnv):
    """
    Goal-conditioned low-level controller

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
        - target_paddle_ori (4)
        - target_time_to_plane (1)

    Observation: 428 + 6 = 434
    """

    def __init__(self, config: Config):
        super().__init__(config)
        
        # ==================================================
        # Observation space
        # ==================================================
        self.state_dim = 428
        self.goal_dim = 8
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

        self.reach_thr = self.reach_thr_base - self.reach_max_delta * self.progress
        self.time_thr  = self.time_thr_base  - self.time_max_delta  * self.progress

        angle_p = self.progress ** 1.5
        self.paddle_ori_thr = (
            self.paddle_ori_thr_base + self.paddle_ori_max_delta * angle_p
        )

    # ==================================================
    # Curriculum hooks
    # ==================================================
    def set_goal_noise_scale(self, scale: float):
        self.goal_noise_scale = float(np.clip(scale, 0.0, 0.2))

    def set_progress(self, progress: float):
        p = float(np.clip(progress, 0.0, 1.0))
        self.progress = p

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
    # Goal API
    # ==================================================
    def _predict(self, obs_dict):
        pred_pos, paddle_ori_ideal = predict_ball_trajectory(
            ball_pos=obs_dict["ball_pos"],
            ball_vel=obs_dict["ball_vel"],
            paddle_pos=obs_dict["paddle_pos"],
        )
        
        return pred_pos, paddle_ori_ideal

    def set_goal(self, goal_phys: np.ndarray):
        self.current_goal = goal_phys
        self.goal_start_time = float(self.env.unwrapped.obs_dict["time"])
        
    def predict_goal_from_state(self, obs_dict):
        pred_pos, pred_paddle_ori = self._predict(obs_dict)
        dx = float(pred_pos[0] - obs_dict["ball_pos"][0])
        vx = float(obs_dict["ball_vel"][0])

        if abs(vx) > 1e-3 and (dx * vx) > 0.0:
            dt = abs(dx / vx)
        else:
            dt = 1.5

        dt = float(np.clip(dt, 0.05, 1.5))

        goal_phys = np.array(
            [
                pred_pos[0],
                pred_pos[1],
                pred_pos[2],
                pred_paddle_ori[0],
                pred_paddle_ori[1],
                pred_paddle_ori[2],
                pred_paddle_ori[3],
                dt
            ],
            dtype=np.float32,
        )

        if self.goal_noise_scale > 0.0:
            goal_phys[:3] += np.random.normal(0.0, self.goal_noise_scale, size=3)
            goal_phys[5] += np.random.normal(0.0, self.goal_noise_scale * 0.5)

        return goal_phys
        
    def _flat(self, x):
        return np.asarray(x, dtype=np.float32).reshape(-1)

    def _build_obs(self, obs_dict):
        time = np.array(
            [obs_dict["time"] - self.goal_start_time],
            dtype=np.float32,
        )

        # FIXED: "paddle_ori_err" typo fixed
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
        
        assert obs.shape == (self.observation_dim,), f"Invalid shape {obs.shape}"
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
        
        info.update({
            "time_threshold": self.time_thr,
            "reach_threshold": self.reach_thr,
            "paddle_ori_threshold": self.paddle_ori_thr,
        })
        
        return self._build_obs(obs_dict), float(reward), terminated, truncated, info
    
    def _compute_reward(self, obs_dict, rwd_dict) -> Tuple[float, bool, Dict]:
        """
        Refactored Reward: Masked + Component-Wise + Quaternion Error (Ort)
        """
        # ==================================================
        # 1. SETUP & MASKS
        # ==================================================
        # Retrieve the goal we set earlier (Frozen Goal)
        # Goal vector: [x, y, z, nx, ny, dt]
        goal_pos = self.current_goal[:3]
        paddle_pos = obs_dict["paddle_pos"]

        # Calculate Masks (Active only if ball is in front of paddle)
        # Note: We use the goal position as a proxy for the ball's impact point
        err_x = float(goal_pos[0] - paddle_pos[0])
        active_mask = 1.0 if err_x > -0.05 else 0.0

        # Check contact to stop "reaching" reward after the hit
        touch_vec = obs_dict["touching_info"]
        has_hit = float(touch_vec[0]) > 0.5
        
        # Once we hit, we stop enforcing alignment (allow follow-through)
        active_alignment_mask = active_mask * (1.0 - float(self._prev_paddle_contact or has_hit))

        # ==================================================
        # 2. POSITION ALIGNMENT (Component-Wise)
        # ==================================================
        # Split error into Y (width) and Z (height) for precise feedback
        pred_err_y = np.abs(paddle_pos[1] - goal_pos[1])
        pred_err_z = np.abs(paddle_pos[2] - goal_pos[2])

        alignment_y = active_alignment_mask * np.exp(-5.0 * pred_err_y)
        alignment_z = active_alignment_mask * np.exp(-5.0 * pred_err_z)

        # ==================================================
        # 3. ORIENTATION ALIGNMENT
        # ==================================================        
        paddle_ori = obs_dict["paddle_ori"]
        goal_ori = self.current_goal[3:7]
        paddle_ori_err = paddle_ori - goal_ori
        paddle_quat_err_goal = np.linalg.norm(paddle_ori_err)
        paddle_quat_reward = active_alignment_mask * np.exp(-5.0 * paddle_quat_err_goal)
        
        # ==================================================
        # 4. PELVIS ALIGNMENT
        # ==================================================
        # Force the base (pelvis) to move with the hand
        pelvis_pos = obs_dict["pelvis_pos"]
        
        # Logic: Calculate where the pelvis *should* be to support the reach
        paddle_to_pelvis_offset = pelvis_pos[:2] - paddle_pos[:2]
        pelvis_target_pos = goal_pos[:2] + paddle_to_pelvis_offset
        
        pelvis_err = np.linalg.norm(pelvis_pos[:2] - pelvis_target_pos)
        pelvis_alignment = active_alignment_mask * np.exp(-5.0 * pelvis_err)

        # ==================================================
        # 5. AGGREGATE REWARDS
        # ==================================================
        reward = 0.0
        
        # Additive weights (Tune these if needed)
        reward += 1.0 * alignment_y
        reward += 1.0 * alignment_z
        reward += 1.0 * paddle_quat_reward
        reward += 0.5 * pelvis_alignment

        # --------------------------------------------------
        # EXTRAS: Posture, Timing, Success
        # --------------------------------------------------
        # Palm Distance
        palm_closeness = float(rwd_dict.get("palm_dist", 0.0))
        reward -= (1.0 - palm_closeness)
        
        # Torso Up
        reward += 0.5 * float(rwd_dict.get("torso_up", 0.0))

        # Timing Penalty (Only if Late)
        dt = float(self.goal_start_time + self.current_goal[5] - obs_dict["time"])
        if dt < -0.05:
            reward -= 1.0 * abs(dt)

        # Success Bonus
        if bool(rwd_dict.get("solved", False)):
            reward += 25.0
            
        # Contact Bonus
        if has_hit and not self._prev_paddle_contact:
            # Check if aligned during hit
            if alignment_y > 0.5 and alignment_z > 0.5:
                reward += 5.0
            else:
                reward += 1.0 
        
        self._prev_paddle_contact = has_hit

        # ==================================================
        # LOGGING
        # ==================================================
        logs = {
            "alignment_y": alignment_y,
            "alignment_z": alignment_z,
            "quat_reward": paddle_quat_reward,
            "pelvis_reward": pelvis_alignment,
            "palm_dist": palm_closeness,
            "is_contact": float(has_hit),
        }

        return float(reward), False, logs