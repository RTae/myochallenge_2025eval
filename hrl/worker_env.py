from typing import Optional
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from hrl.utils import predict_ball_trajectory, get_face_normal, ensure_handle_down, quat_rotate, flip_quat_180_x

class TableTennisWorker(CustomEnv):
    def __init__(self, config: Config):
        super().__init__(config)
        
        # 1(time) + 3(pelvis) + 58(qpos) + 58(qvel) + 3(ball) + 3(ball_v) + 
        # 3(pad) + 3(pad_v) + 4(pad_ori) + 3(reach) + 3(palm) + 3(palm_err) + 
        # 6(touch) + 273(act) = 424
        self.state_dim = 424
        self.goal_dim = 8 
        self.observation_dim = self.state_dim + self.goal_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32,
        )

        # Runtime State
        self.current_goal: Optional[np.ndarray] = None
        self._track_goal: Optional[np.ndarray] = None
        self.manager_delta: np.ndarray = np.zeros(self.goal_dim, dtype=np.float32)
        
        self.goal_start_time: Optional[float] = None
        self.prev_reach_err: Optional[float] = None
        self._prev_paddle_contact = False
        self.init_paddle_x = None
        
        # Bounce Logic State
        self.has_bounced = False
        self.bounce_time = 0.0
        
        # Config
        self.goal_noise_scale = 0.0
        self.progress = 0.0

        # Thresholds
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

    def set_goal_noise_scale(self, scale: float):
        self.goal_noise_scale = float(np.clip(scale, 0.0, 0.2))

    def set_progress(self, progress: float):
        self.progress = float(np.clip(progress, 0.0, 1.0))
        self.reach_thr = self.reach_thr_base - self.reach_max_delta * self.progress
        self.time_thr  = self.time_thr_base  - self.time_max_delta  * self.progress
        self.paddle_ori_thr = self.paddle_ori_thr_base - self.paddle_ori_max_delta * self.progress
        
    def get_progress(self):
        return float(self.progress)
        
    def _predict(self, obs_dict):
        # 1. Use Proxy Position (from Debugger Logic)
        paddle_proxy = obs_dict["paddle_pos"].copy()
        
        if self.init_paddle_x is None:
            self.init_paddle_x = obs_dict["paddle_pos"][0]
        
        if self.init_paddle_x is not None:
            # Use fixed defensive line - 0.2m (as tested in debugger)
            paddle_proxy[0] = self.init_paddle_x - 0.2
            
        return predict_ball_trajectory(
            ball_pos=obs_dict["ball_pos"],
            ball_vel=obs_dict["ball_vel"],
            paddle_pos=paddle_proxy, 
        )

    def set_goal(self, goal_phys: np.ndarray, delta: Optional[np.ndarray] = None):
        self.current_goal = goal_phys.astype(np.float32)
        if delta is not None:
            if delta.shape[0] == self.goal_dim:
                self.manager_delta = delta.astype(np.float32)
            else:
                padded = np.zeros(self.goal_dim)
                padded[:min(len(delta), self.goal_dim)] = delta
                self.manager_delta = padded.astype(np.float32)
        else:
            self.manager_delta = np.zeros(self.goal_dim, dtype=np.float32)
        self.goal_start_time = float(self.env.unwrapped.obs_dict["time"])
        
    def predict_goal_from_state(self, obs_dict):
        pred_pos, pred_quat = self._predict(obs_dict)
                
        target_quat = pred_quat
        is_forehand = pred_pos[1] > 0.0
        if is_forehand:
            target_quat = flip_quat_180_x(pred_quat)
        
        target_quat = ensure_handle_down(target_quat)

        dx = float(pred_pos[0] - obs_dict["ball_pos"][0])
        vx = float(obs_dict["ball_vel"][0])
        dt = float(np.clip(abs(dx / vx) if abs(vx) > 1e-3 else 1.5, 0.05, 1.5))

        goal_phys = np.concatenate([pred_pos, target_quat, [dt]])

        if self.goal_noise_scale > 0.0:
            goal_phys[:3] += np.random.normal(0.0, self.goal_noise_scale, size=3)
            goal_phys[3:7] += np.random.normal(0.0, self.goal_noise_scale * 0.1, size=4)
            goal_phys[3:7] /= (np.linalg.norm(goal_phys[3:7]) + 1e-9)

        return goal_phys.astype(np.float32)
    
    def _update_goal_tracker(self, obs_dict):
        current_time = obs_dict["time"]
        ball_pos = obs_dict["ball_pos"]
        ball_vel = obs_dict["ball_vel"]
        paddle_pos = obs_dict["paddle_pos"]
        
        ball_x = ball_pos[0]
        paddle_x = paddle_pos[0]
        
        # 1. BOUNCE DETECTION LOGIC
        if (not self.has_bounced) and (ball_x > -0.2) and (ball_vel[2] > 0.2):
            self.has_bounced = True
            self.bounce_time = current_time

        # 2. SAFETY CHECKS
        touching = obs_dict.get('touching_info')
        current_touch = 1.0 if (touching is not None and touching[0] > 0.1) else 0.0
        if current_touch > 0: self._prev_paddle_contact = True

        is_contact = (current_touch > 0) or self._prev_paddle_contact
        is_hitting_zone = (abs(ball_x - paddle_x) < 0.30)
        is_ball_passed = (self.init_paddle_x - ball_x) < -0.05
        is_ball_towards_paddle = (ball_vel[0] > 0)
        
        should_track = (not is_contact) and (not is_hitting_zone) and \
                    (not is_ball_passed) and is_ball_towards_paddle

        update_condition = False
        
        if should_track:
            time_since_bounce = current_time - self.bounce_time
            # Update if bounced AND delay > 0.05s
            if self.has_bounced and (time_since_bounce > 0.05):
                 update_condition = True
            # Also allow initial prediction update if goal is empty
            if self._track_goal is None:
                 update_condition = True

        # 4. PREDICT
        if update_condition:
            new_pos, new_quat = self._predict(obs_dict)
                        
            target_quat = new_quat
            is_forehand = new_pos[1] > 0.0
            if is_forehand:
                target_quat = flip_quat_180_x(new_quat)
            else:
                target_quat = new_quat
                
            target_quat = ensure_handle_down(target_quat)
            
            dx = float(new_pos[0] - ball_x)
            new_dt = float(np.clip(abs(dx / ball_vel[0]) if abs(ball_vel[0]) > 1e-3 else 1.5, 0.05, 1.5))
            
            target_goal = np.concatenate([new_pos, target_quat, [new_dt]]) + self.manager_delta[:8]
            self._track_goal = target_goal.copy()

        if self._track_goal is not None:
            # Smooth Position
            curr_pos = self.current_goal[0:3]
            target_pos = self._track_goal[0:3]
            delta_pos = target_pos - curr_pos
            dist_pos = np.linalg.norm(delta_pos)
            
            if dist_pos > 0.05: 
                self.current_goal[0:3] = curr_pos + (delta_pos * (0.05 / (dist_pos + 1e-9)))
            else:
                self.current_goal[0:3] = target_pos
            
            # Smooth Quaternion
            curr_q = self.current_goal[3:7]
            target_q = self._track_goal[3:7]
            
            if np.dot(curr_q, target_q) < 0:
                target_q = -target_q
                
            delta_q = target_q - curr_q
            dist_q = np.linalg.norm(delta_q)
            
            if dist_q > 0.1: 
                new_q = curr_q + (delta_q * (0.1 / (dist_q + 1e-9)))
            else:
                new_q = target_q
            
            new_q /= (np.linalg.norm(new_q) + 1e-9)
            self.current_goal[3:7] = new_q
            self.current_goal[7] = self._track_goal[7]
            
    def _build_obs(self, obs_dict):
        obs = np.concatenate([
            self._flat(obs_dict["time"]),           
            self._flat(obs_dict["pelvis_pos"]),     
            self._flat(obs_dict["body_qpos"]),      
            self._flat(obs_dict["body_qvel"]),      
            self._flat(obs_dict["ball_pos"]),       
            self._flat(obs_dict["ball_vel"]),       
            self._flat(obs_dict["paddle_pos"]),     
            self._flat(obs_dict["paddle_vel"]),     
            self._flat(obs_dict["paddle_ori"]),                
            self._flat(obs_dict["reach_err"]),      
            self._flat(obs_dict["palm_pos"]),       
            self._flat(obs_dict["palm_err"]),       
            self._flat(obs_dict["touching_info"]),  
            self._flat(obs_dict["act"]),            
            self._flat(self.current_goal),
        ], axis=0)
        
        return obs.astype(np.float32)
    
    def _flat(self, x):
        return np.asarray(x, dtype=np.float32).reshape(-1)
        
    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed)
        obs_dict = self.env.unwrapped.obs_dict

        self._prev_paddle_contact = False
        self._track_goal = None
        self.init_paddle_x = obs_dict["paddle_pos"][0]
        
        # Reset Bounce State
        self.has_bounced = False
        self.bounce_time = 0.0
        
        self.manager_delta = np.zeros(self.goal_dim, dtype=np.float32)
        
        goal = self.predict_goal_from_state(obs_dict)
        self.set_goal(goal, delta=None)
        
        self.prev_reach_err = np.linalg.norm(obs_dict["paddle_pos"] - self.current_goal[:3])
        return self._build_obs(obs_dict), info

    def step(self, action: np.ndarray):
        clean_action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        clamped_action = clean_action.copy()
        low  = self.action_space.low
        high = self.action_space.high
        clamped_action = np.clip(clean_action, low, high)

        _, base_reward, terminated, truncated, info = super().step(clamped_action)
        
        obs_dict = info["obs_dict"]
        rwd_dict = info["rwd_dict"]
        self._update_goal_tracker(obs_dict)

        reward, _, logs = self._compute_reward(obs_dict, rwd_dict)
        if not np.isfinite(reward): raise RuntimeError(f"[NaN/Inf] reward={reward}")
        
        reward += 0.05 * base_reward
        info.update(logs)
        info.update({
            "time_threshold": self.time_thr,
            "reach_threshold": self.reach_thr,
            "paddle_ori_threshold": self.paddle_ori_thr,
        })
        return self._build_obs(obs_dict), float(reward), terminated, truncated, info
    
    def _compute_reward(self, obs_dict, rwd_dict):
        goal_pos = self.current_goal[0:3]
        goal_quat = self.current_goal[3:7]
        
        paddle_pos = obs_dict["paddle_pos"]
        pelvis_pos = obs_dict["pelvis_pos"]
        ball_x = obs_dict["ball_pos"][0]
        touch_vec = obs_dict["touching_info"]
        
        is_env_success = bool(rwd_dict.get("solved", False))
        is_ball_towards_paddle = (obs_dict["ball_vel"][0] > 0)
        is_ball_passed = (self.init_paddle_x - ball_x) < -0.05
        active_mask = float(is_ball_towards_paddle and not is_ball_passed)
        has_hit = float(touch_vec[0]) > 0.1
        raw_palm_dist = np.linalg.norm(obs_dict["palm_err"])
        is_holding = float(raw_palm_dist < 0.20)

        active_alignment_mask = active_mask * (1.0 - float(self._prev_paddle_contact or has_hit)) * is_holding

        # Position
        pred_err_y = np.abs(paddle_pos[1] - goal_pos[1])
        pred_err_z = np.abs(paddle_pos[2] - goal_pos[2])
        alignment_y = active_alignment_mask * np.exp(-1.0 * pred_err_y)
        alignment_z = active_alignment_mask * np.exp(-1.0 * pred_err_z)

        # Orientation
        paddle_quat = obs_dict["paddle_ori"]
        curr_n = get_face_normal(paddle_quat)
        goal_n = get_face_normal(goal_quat)

        dot = np.dot(curr_n, goal_n)
        paddle_face_err = np.arccos(np.clip(dot, -1.0, 1.0))
        
        handle_world = quat_rotate(paddle_quat, np.array([1.0, 0.0, 0.0]))
        # penalize if handle points upward in world z
        handle_up_pen = max(0.0, handle_world[2])   # >0 means pointing up

        # only reward if it's the correct face (not backside)
        paddle_quat_reward = active_alignment_mask * np.exp(-2.0 * paddle_face_err)
        
        # Pelvis
        paddle_to_pelvis_offset = np.array([-0.2, 0.4])
        pelvis_target_xy = goal_pos[:2] + paddle_to_pelvis_offset
        pelvis_err = np.linalg.norm(pelvis_pos[:2] - pelvis_target_xy)
        pelvis_alignment = active_alignment_mask * np.exp(-5.0 * pelvis_err)

        reach_dist = float(np.linalg.norm(paddle_pos - goal_pos, axis=-1))
        is_reach_good = reach_dist < self.reach_thr
        dt = float(self.goal_start_time + self.current_goal[7] - obs_dict["time"])
        is_time_good = dt > -self.time_thr
        is_ori_good = paddle_face_err < self.paddle_ori_thr
        is_goal_success = float(is_reach_good and is_ori_good and is_time_good)

        reward = 0.0
        reward += 2.0 * alignment_y
        reward += 2.0 * alignment_z
        reward += active_mask * 2.0 * (1.0 - np.tanh(reach_dist))
        reward += 2.0 * paddle_quat_reward
        reward += 0.5 * pelvis_alignment
        reward -= 0.2 * handle_up_pen

        if not is_holding: reward -= 1.0 
        reward += 0.1 * float(rwd_dict.get("torso_up", 0.0))
        
        if dt < -self.time_thr:
            reward -= 0.5 * np.tanh(max(0.0, -dt))

        if is_env_success:
            reward += 25.0
        
        is_contact_fresh = False
        if has_hit and not self._prev_paddle_contact:
            if alignment_y > 0.3 and alignment_z > 0.3 and dot > 0.3:
                reward += 10.0 
                is_contact_fresh = True
            else:
                reward += 1.0 
        
        self._prev_paddle_contact = has_hit
        
        # elbow_flexion = obs_dict["body_qpos"][18] 
        # if elbow_flexion > 2.0:
        #     reward -= 2.0 * (elbow_flexion - 2.0)
        

        logs = {
            "is_goal_success": is_goal_success,            
            "reach_error": reach_dist,
            "reach_y_err": pred_err_y,
            "reach_z_err": pred_err_z,
            "paddle_degree_err": np.degrees(paddle_face_err),
            "handle_up_pen": handle_up_pen,
            "time_err": dt,
            "abs_time_err": abs(dt),
            "is_ball_passed": float(is_ball_passed),
            "is_contact": float(is_contact_fresh),
            "pelvis_err": pelvis_err,
            "alignment_y": alignment_y,
            "alignment_z": alignment_z,
            "quat_reward": paddle_quat_reward,
            "pelvis_reward": pelvis_alignment,
            "palm_dist": raw_palm_dist,
        }

        return float(reward), False, logs