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

    def __init__(self, config: Config):
        super().__init__(config)
        
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

        self.reach_thr = self.reach_thr_base - self.reach_max_delta * self.progress
        self.time_thr  = self.time_thr_base  - self.time_max_delta  * self.progress

        angle_p = self.progress ** 1.5
        self.paddle_ori_thr = (
            self.paddle_ori_thr_base + self.paddle_ori_max_delta * angle_p
        )

    # ==================================================
    # Curriculum hooks (called by callback)
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
            # 1. Execute low-level environment step
            _, base_reward, terminated, truncated, info = super().step(action)
            
            obs_dict = info["obs_dict"]
            rwd_dict = info["rwd_dict"]

            # 2. Compute our custom goal-conditioned reward
            reward, _, logs = self._compute_reward(obs_dict, rwd_dict)
            
            # 3. Add base reward (small weight to prevent interference, but keep alive)
            reward += 0.05 * base_reward

            # 4. Update info with our custom logs
            info.update(logs)
            
            # 5. Build observation
            obs = self._build_obs(obs_dict)
            
            return obs, float(reward), terminated, truncated, info
    
    def _compute_reward(self, obs_dict, rwd_dict):
            # ==================================================
            # GEOMETRY SETUP
            # ==================================================
            paddle_pos = obs_dict["paddle_pos"]
            goal_pos   = self.current_goal[:3]

            reach_err = float(np.linalg.norm(paddle_pos - goal_pos))

            # Initialize previous error if this is the first step
            if self.prev_reach_err is None:
                self.prev_reach_err = reach_err
            reach_delta = self.prev_reach_err - reach_err
            self.prev_reach_err = reach_err

            reward = 0.0

            # ==================================================
            # 1) APPROACH (Dense Shaping)
            # ==================================================
            # General distance penalty (gradient to goal)
            reward += -0.15 * reach_err

            # Reward for improving reach error (delta)
            reward += 1.2 * np.clip(reach_delta, -0.05, 0.05)

            # Directional velocity bonus (guide velocity vector toward ball when far)
            if reach_err > 0.30:
                v = np.asarray(obs_dict["paddle_vel"], dtype=np.float32)
                to_goal = goal_pos - paddle_pos
                dist = np.linalg.norm(to_goal)
                if dist > 1e-6:
                    to_goal /= dist
                    v_toward = float(np.dot(v, to_goal))
                    reward += 0.25 * np.clip(v_toward, 0.0, 2.0)

            # ==================================================
            # 2) ORIENTATION (Dense Shaping)
            # ==================================================
            paddle_n = quat_to_paddle_normal(obs_dict["paddle_ori"])
            paddle_n /= (np.linalg.norm(paddle_n) + 1e-8)

            goal_n = self._unpack_normal_xy(
                self.current_goal[3], self.current_goal[4]
            )

            cos_sim = float(np.clip(np.dot(paddle_n, goal_n), 0.0, 1.0))

            # Scale orientation reward by distance (only matters when close)
            reward += 0.4 * cos_sim * np.exp(-2.0 * reach_err)

            # ==================================================
            # 3) TIMING (Dense Shaping)
            # ==================================================
            # dt > 0: Ball is approaching. dt < 0: Ball has passed.
            dt = float(self.goal_start_time + self.current_goal[5] - obs_dict["time"])

            # Only apply dense timing signals when physically close
            hit_ready = reach_err < 1.2 * self.reach_thr
            
            if hit_ready:
                # We enforce a "not too late" penalty
                if dt >= -self.time_thr_base:
                    # Guidance toward the predicted ideal time (dt=0)
                    reward += 0.4 * np.exp(-6.0 * max(0.0, dt))
                else:
                    # Linear penalty for being late
                    reward -= 0.8 * (-dt)

            # ==================================================
            # 4) READINESS (Multiplicative Gating)
            # ==================================================
            reach_score = np.exp(-6.0 * reach_err)
            angle_score = cos_sim ** 2
            # Use max(0, dt) to ensure early readiness is rewarded
            time_score  = np.exp(-6.0 * max(dt, 0.0)) 

            readiness = reach_score * angle_score * time_score
            readiness = np.clip(readiness, 0.0, 1.0)
            
            reward += 0.5 * readiness

            # ==================================================
            # 5) POSTURE
            # ==================================================
            palm_closeness = float(rwd_dict.get("palm_dist", 0.0))
            reward -= 0.2 * (1.0 - palm_closeness)

            torso_up = float(rwd_dict.get("torso_up", 0.0))
            reward += 0.25 * torso_up

            # ==================================================
            # 6) CONTACT (One-shot Event)
            # ==================================================
            touch_vec = obs_dict["touching_info"]
            paddle_hit = float(touch_vec[0]) > 0.5

            is_contact = False
            
            # If dt is large positive (early hit), it is still valid.
            goal_aligned = (
                reach_err < self.reach_thr * 1.5
                and dt >= -self.time_thr_base*1.3
                and cos_sim > self.paddle_ori_thr - 0.15
            )
                    
            if paddle_hit and not self._prev_paddle_contact:
                is_contact = True
                if goal_aligned:
                    reward += 5.0
                else:
                    # Small penalty for hitting ball while flailing/uncontrolled
                    reward -= 1.0 
                
            self._prev_paddle_contact = paddle_hit

            # Safety penalty (always active)
            v_norm = float(np.linalg.norm(obs_dict["paddle_vel"]))
            if v_norm > 6.0:
                reward -= 0.05 * (v_norm - 6.0)

            # ==================================================
            # 7) ENV SUCCESS (True Task Solved)
            # ==================================================
            if bool(rwd_dict.get("solved", False)):
                reward += 25.0
                
            # Logging metric for "Perfect" alignment
            hard_aligned = (
                reach_err < self.reach_thr
                and dt >= -self.time_thr_base*1.3:
                and cos_sim > self.paddle_ori_thr
            )

            # ==================================================
            # LOGS
            # ==================================================
            logs = {
                "reach_err": reach_err,
                "reach_delta": reach_delta,
                "cos_sim": cos_sim,
                "time_error": dt,
                "abs_time_error": abs(dt),
                "readiness": readiness,
                "paddle_speed": v_norm,
                "palm_dist": palm_closeness,
                "torso_up": torso_up,
                "is_contact": float(is_contact),
                "is_soft_goal_success": float(goal_aligned),
                "is_goal_success": float(hard_aligned),
            }

            return float(reward), False, logs