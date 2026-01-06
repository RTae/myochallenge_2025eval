from __future__ import annotations

from typing import Optional, Tuple, Dict
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv

from hrl.utils import predict_ball_trajectory, get_z_normal


class TableTennisWorker(CustomEnv):
    """
    Goal-conditioned low-level controller (Normal-Vector Optimized)

    State 432:
        - time (1)
        - pelvis_pos (3)
        - body_qpos (58)
        - body_qvel (58)
        - ball_pos (3)
        - ball_vel (3)
        - paddle_pos (3)
        - paddle_vel (3)
        - paddle_ori (4)
        - reach_err (3)
        - pos_err (3)
        - curr_face_normal (3)
        - face_alignment_dot (1)
        - palm_pos (3)
        - palm_err (3)
        - touching_info (6)
        - act (273)

    Goal 7:
        - target_ball_pos (3)   # x, y, z intercept
        - target_face_normal (3)# nx, ny, nz world-space aim
        - target_time (1)       # dt to intercept

    Observation: 431 + 7 = 438
    """

    def __init__(self, config: Config):
        super().__init__(config)
        
        # ==================================================
        # Observation space
        # ==================================================
        self.state_dim = 431
        self.goal_dim = 7
        self.observation_dim = self.state_dim + self.goal_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        
        self.grip_indices = [242, 243, 244, 245, 246, 247, 248, 249, 258, 260]

        # ==================================================
        # Runtime
        # ==================================================
        self.current_goal: Optional[np.ndarray] = None
        self._track_goal: Optional[np.ndarray] = None
        self.manager_delta: np.ndarray = np.zeros(self.goal_dim, dtype=np.float32)
        
        self.goal_start_time: Optional[float] = None
        self.prev_reach_err: Optional[float] = None
        self._prev_paddle_contact = False
        self.init_paddle_x = None
        
        # ==================================================
        # Curriculum state
        # ==================================================
        self.goal_noise_scale = 0.0
        self.progress = 0.0

        # ==================================================
        # Success thresholds
        # ==================================================
        self.reach_thr_base = 0.25
        self.reach_max_delta = 0.15
        self.time_thr_base = 0.1
        self.time_max_delta = 0.05
        self.paddle_ori_thr_base = 0.4
        self.paddle_ori_max_delta = 0.3

        self.reach_thr = self.reach_thr_base
        self.time_thr = self.time_thr_base
        self.paddle_ori_thr = self.paddle_ori_thr_base
        
        self.set_progress(0.0)

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
        self.paddle_ori_thr = self.paddle_ori_thr_base - self.paddle_ori_max_delta * p
        
    def get_progress(self):
        return float(self.progress)
        
    # ==================================================
    # Goal API
    # ==================================================
    def _predict(self, obs_dict):
        """Helper to get physics prediction"""
        pred_pos, paddle_ori_ideal = predict_ball_trajectory(
            ball_pos=obs_dict["ball_pos"],
            ball_vel=obs_dict["ball_vel"],
            paddle_pos=obs_dict["paddle_pos"],
        )
        return pred_pos, paddle_ori_ideal

    def set_goal(self, goal_phys: np.ndarray, delta: Optional[np.ndarray] = None):
        self.current_goal = goal_phys.astype(np.float32)
        
        if delta is not None:
            self.manager_delta = delta.astype(np.float32)
        else:
            self.manager_delta = np.zeros(self.goal_dim, dtype=np.float32)
            
        self.goal_start_time = float(self.env.unwrapped.obs_dict["time"])
        
    def predict_goal_from_state(self, obs_dict):
        # 1. Physics Prediction
        pred_pos, pred_quat = self._predict(obs_dict)
        
        # 2. Rule-Based Switch
        # relative_y > 0 usually means ball is on the right (Forehand side)
        relative_y = pred_pos[1] - obs_dict["pelvis_pos"][1]
        
        # Get the standard normal from the physics predictor
        base_normal = get_z_normal(pred_quat)
        
        if relative_y > 0.0:
            # Flip to Forehand side (Black color)
            base_normal = -base_normal

        # 3. Calculate DT
        dx = float(pred_pos[0] - obs_dict["ball_pos"][0])
        vx = float(obs_dict["ball_vel"][0])
        dt = float(np.clip(abs(dx / vx) if abs(vx) > 1e-3 else 1.5, 0.05, 1.5))

        # 4. Build 7-Dim Goal [x,y,z, nx,ny,nz, dt]
        goal_phys = np.concatenate([pred_pos, base_normal, [dt]])

        # 5. Apply Noise (Optional)
        if self.goal_noise_scale > 0.0:
            goal_phys[:3] += np.random.normal(0.0, self.goal_noise_scale, size=3)
            goal_phys[3:6] += np.random.normal(0.0, self.goal_noise_scale * 0.1, size=3)
            goal_phys[3:6] /= (np.linalg.norm(goal_phys[3:6]) + 1e-9)
            
        if self.init_paddle_x is None:
            self.init_paddle_x = pred_pos[0]

        return goal_phys.astype(np.float32)
    
    def _update_goal_tracker(self, obs_dict):
        ball_x = obs_dict["ball_pos"][0]
        ball_vel_x = obs_dict["ball_vel"][0]
        paddle_x = obs_dict["paddle_pos"][0]
        touching = obs_dict.get('touching_info')
        
        current_touch = 1.0 if (touching is not None and touching[0] > 0.1) else 0.0
        
        if current_touch > 0:
            self._prev_paddle_contact = True

        is_contact = (current_touch > 0) or self._prev_paddle_contact   # Ball hit a paddle
        is_hitting_zone = (abs(ball_x - paddle_x) < 0.30)               # Ball and Paddle X-distance check
        is_ball_passed = self.init_paddle_x - ball_x < -0.05            # Ball has passed initial paddle X
        is_ball_towards_paddle = (ball_vel_x > 0)                       # Ball moving towards paddle
        
        should_track = (not is_contact) and (not is_hitting_zone) and \
                    (not is_ball_passed) and is_ball_towards_paddle

        if should_track:
            new_pos, new_quat = self._predict(obs_dict)
            
            relative_y = new_pos[1] - obs_dict["pelvis_pos"][1]
            new_normal = get_z_normal(new_quat)
            
            if relative_y > 0.0:
                new_normal = -new_normal
            
            dx = float(new_pos[0] - ball_x)
            new_dt = float(np.clip(abs(dx / ball_vel_x) if abs(ball_vel_x) > 1e-3 else 1.5, 0.05, 1.5))
            
            # Update target (Manager delta is 7-dim now)
            target_goal = np.concatenate([new_pos, new_normal, [new_dt]]) + self.manager_delta[:7]
            self._track_goal = target_goal.copy()

        if self._track_goal is not None:
            # Smooth Position
            curr_pos = self.current_goal[0:3]
            target_pos = self._track_goal[0:3]
            delta_pos = target_pos - curr_pos
            dist_pos = np.linalg.norm(delta_pos)
            
            if dist_pos > 0.05: # MAX_POS_DELTA
                self.current_goal[0:3] = curr_pos + (delta_pos * (0.05 / (dist_pos + 1e-9)))
            else:
                self.current_goal[0:3] = target_pos
            
            # Smooth Normal Vector
            curr_norm_target = self.current_goal[3:6]
            target_norm = self._track_goal[3:6]
            delta_norm = target_norm - curr_norm_target
            dist_norm = np.linalg.norm(delta_norm)
            
            if dist_norm > 0.1: # MAX_ROT_DELTA
                self.current_goal[3:6] += delta_norm * (0.1 / (dist_norm + 1e-9))
                self.current_goal[3:6] /= (np.linalg.norm(self.current_goal[3:6]) + 1e-9)
            else:
                self.current_goal[3:6] = target_norm

            self.current_goal[6] = self._track_goal[6]
    def _build_obs(self, obs_dict):
        # 1. Capture current face direction
        paddle_quat = obs_dict["paddle_ori"]
        curr_normal = get_z_normal(paddle_quat)
        
        # 2. Capture target face direction from our 7-dim goal
        goal_pos = self.current_goal[0:3]
        goal_normal = self.current_goal[3:6]
        
        # 3. Calculate explicit error features for the network
        pos_err = goal_pos - obs_dict["paddle_pos"]
        # How well are we aligned? (1.0 = perfect, 0.0 = edge-on)
        normal_dot = np.dot(curr_normal, goal_normal)

        # 4. Concatenate 
        obs = np.concatenate([
            self._flat(obs_dict["time"]),           # 1
            self._flat(obs_dict["pelvis_pos"]),     # 3
            self._flat(obs_dict["body_qpos"]),      # 58
            self._flat(obs_dict["body_qvel"]),      # 58
            self._flat(obs_dict["ball_pos"]),       # 3
            self._flat(obs_dict["ball_vel"]),       # 3
            self._flat(obs_dict["paddle_pos"]),     # 3
            self._flat(obs_dict["paddle_vel"]),     # 3
            self._flat(paddle_quat),                # 4
            self._flat(curr_normal),                # 3 (Current aim)
            self._flat(obs_dict["reach_err"]),      # 3 (Base env reach err between paddle and ball)
            self._flat(pos_err),                    # 3 (Goal reach err)
            self._flat(normal_dot),                 # 1 (Aim quality)
            self._flat(obs_dict["palm_pos"]),       # 3
            self._flat(obs_dict["palm_err"]),       # 3
            self._flat(obs_dict["touching_info"]),  # 6
            self._flat(obs_dict["act"]),            # 273
            self._flat(self.current_goal),          # 7 (pos, normal, dt)
        ], axis=0)
        
        return obs.astype(np.float32)
    
    def _flat(self, x):
        return np.asarray(x, dtype=np.float32).reshape(-1)

    # ==================================================
    # Gym API
    # ==================================================
    def _force_start_grasp(self):
        """
        Forces the 'Golden Grip' finger values.
        Crucial for Frame 0 stability.
        """
        sim = self.env.unwrapped.sim
        model = sim.model
        data = sim.data

        perfect_fingers = {
            "cmc_abduction": -0.334, "cmc_flexion": 0.0562, "mp_flexion": -0.511, "ip_flexion": -0.881,
            "mcp2_flexion": 1.49, "mcp2_abduction": 0.147, "pm2_flexion": 1.3, "md2_flexion": 1.25,
            "mcp3_flexion": 1.42, "mcp3_abduction": -0.0131, "pm3_flexion": 1.35, "md3_flexion": 1.04,
            "mcp4_flexion": 1.48, "mcp4_abduction": -0.0681, "pm4_flexion": 1.36, "md4_flexion": 1.07,
            "mcp5_flexion": 1.39, "mcp5_abduction": -0.188, "pm5_flexion": 0.872, "md5_flexion": 1.57
        }

        for name, val in perfect_fingers.items():
            try:
                jid = model.joint_name2id(name)
                adr = model.jnt_qposadr[jid]
                data.qpos[adr] = val
            except ValueError:
                pass
        
        sim.forward()
        
    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed)
        
        # 1. INJECT STARTING GRIP (Frame 0)
        self._force_start_grasp()
        
        obs_dict = self.env.unwrapped.obs_dict

        self._prev_paddle_contact = False
        self._track_goal = None
        self.manager_delta = np.zeros(self.goal_dim, dtype=np.float32)
        
        goal = self.predict_goal_from_state(obs_dict)
        self.set_goal(goal, delta=None)
        
        self.prev_reach_err = np.linalg.norm(
            obs_dict["paddle_pos"] - self.current_goal[:3]
        )

        return self._build_obs(obs_dict), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        clean_action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        clamped_action = clean_action.copy()
        clamped_action[self.grip_indices] = 1.0
        clamped_action = np.clip(clamped_action, 0.0, 1.0)

        # Pass the clamped action to the physics engine
        _, base_reward, terminated, truncated, info = super().step(clamped_action)
        
        obs_dict = info["obs_dict"]
        rwd_dict = info["rwd_dict"]
        
        self._update_goal_tracker(obs_dict)

        reward, _, logs = self._compute_reward(obs_dict, rwd_dict)
        if not np.isfinite(reward):
            raise RuntimeError(f"[NaN/Inf] reward={reward}, logs={logs}")
        
        reward += 0.05 * base_reward

        info.update(logs)
        info.update({
            "time_threshold": self.time_thr,
            "reach_threshold": self.reach_thr,
            "paddle_ori_threshold": self.paddle_ori_thr,
        })
        
        obs = self._build_obs(obs_dict)
        
        return obs, float(reward), terminated, truncated, info
    
    def _compute_reward(self, obs_dict, rwd_dict) -> Tuple[float, bool, Dict]:
        # 1. Goal Extraction (7-dim goal: [0,1,2]=Pos, [3,4,5]=Normal, [6]=dt)
        goal_pos = self.current_goal[0:3]
        goal_normal = self.current_goal[3:6]
        
        paddle_pos = obs_dict["paddle_pos"]
        pelvis_pos = obs_dict["pelvis_pos"]
        
        # 2. Status Flags
        is_env_success = bool(rwd_dict.get("solved", False))
        err_x = obs_dict["reach_err"][0] 
        # Only reward alignment if the ball is still in front of the paddle
        active_mask = float(err_x > -0.05)

        touch_vec = obs_dict["touching_info"]
        has_hit = float(touch_vec[0]) > 0.5
        
        raw_palm_dist = np.linalg.norm(obs_dict["palm_err"])
        is_holding = float(raw_palm_dist < 0.20)

        # Mask to reward alignment only BEFORE contact and while holding
        active_alignment_mask = active_mask * \
                                (1.0 - float(self._prev_paddle_contact or has_hit)) * \
                                is_holding

        # 3. Position Alignment (Y and Z height)
        pred_err_y = np.abs(paddle_pos[1] - goal_pos[1])
        pred_err_z = np.abs(paddle_pos[2] - goal_pos[2])
        alignment_y = active_alignment_mask * np.exp(-1.0 * pred_err_y)
        alignment_z = active_alignment_mask * np.exp(-1.0 * pred_err_z)

        # 4. Orientation Alignment
        paddle_quat = obs_dict["paddle_ori"]
        curr_normal = get_z_normal(paddle_quat)
        
        # face (Red or Black) dictated by the goal side-switch logic.
        dot = np.dot(curr_normal, goal_normal)
        dot_clipped = np.clip(dot, -1.0, 1.0)
        
        # We use clip(dot, 0, 1) for the reward so that 'wrong face' hits get 0 reward
        paddle_face_err = np.arccos(np.clip(dot_clipped, 0.0, 1.0))
        paddle_quat_reward = active_alignment_mask * np.exp(-2.0 * paddle_face_err)
        
        # 5. Pelvis Positioning (Encourage staying in hitting pocket)
        paddle_to_pelvis_offset = np.array([-0.2, 0.4])
        pelvis_target_xy = goal_pos[:2] + paddle_to_pelvis_offset
        pelvis_err = np.linalg.norm(pelvis_pos[:2] - pelvis_target_xy)
        pelvis_alignment = active_alignment_mask * np.exp(-5.0 * pelvis_err)

        # 6. Success Threshold Checks
        reach_dist = float(np.linalg.norm(paddle_pos - goal_pos, axis=-1))
        is_reach_good = reach_dist < self.reach_thr
        is_ori_good = paddle_face_err < self.paddle_ori_thr
        
        # Time Logic using Index 6 of Goal
        dt = float(self.goal_start_time + self.current_goal[6] - obs_dict["time"])
        is_time_good = dt > -self.time_thr
        
        is_goal_success = float(is_reach_good and is_ori_good and is_time_good)

        # 7. Reward Aggregation
        reward = 0.0
        reward += 5.0 * alignment_y
        reward += 5.0 * alignment_z
        reward += 10.0 * (1.0 - np.tanh(reach_dist))
        reward += 2.0 * paddle_quat_reward
        reward += 0.5 * pelvis_alignment

        # 8. Penalties
        if not is_holding:
            reward -= 1.0 # Increased penalty for dropping paddle
        
        reward += 0.1 * float(rwd_dict.get("torso_up", 0.0))

        if dt < -self.time_thr:
            reward -= 1.0 * abs(dt) # Late penalty

        # 9. Success Bonus
        if is_env_success:
            reward += 25.0
        
        # Contact Bonus (Bonus for hitting with good alignment)
        is_contact_fresh = False
        if has_hit and not self._prev_paddle_contact:
            if alignment_y > 0.5 and alignment_z > 0.5 and dot > 0.7:
                reward += 10.0 # High bonus for clean, aligned hit
                is_contact_fresh = True
            else:
                reward += 1.0 
        
        # 10. Future Hooks (Placeholders for Penalties 1 & 2)
        # effort_penalty = 0.1 * np.mean(np.square(obs_dict["act"]))
        # reward -= effort_penalty
        
        # elbow_flexion = obs_dict["body_qpos"][18]
        # if elbow_flexion > 1.8:
        #     reward -= 2.0 * (elbow_flexion - 1.8)
        
        self._prev_paddle_contact = has_hit

        logs = {
            "is_goal_success": is_goal_success,            
            "reach_error": reach_dist,
            "reach_y_err": pred_err_y,
            "reach_z_err": pred_err_z,
            "paddle_err": paddle_face_err,
            "time_err": dt,
            "abs_time_err": abs(dt),
            "is_ball_passed": active_mask,
            "is_contact": float(is_contact_fresh),
            "pelvis_err": pelvis_err,
            "alignment_y": alignment_y,
            "alignment_z": alignment_z,
            "quat_reward": paddle_quat_reward,
            "pelvis_reward": pelvis_alignment,
            "palm_dist": raw_palm_dist,
        }

        return float(reward), False, logs