import gymnasium as gym
import numpy as np
from myosuite.utils import gym as myo_gym
from collections import deque
from config import Config


class MultiHitRallyEnv(gym.Env):
    """
    Multi-hit rally environment for MyoChallenge 2025.
    Key changes:
    - Tracks multiple hits per episode
    - Rewards each successful return
    - Penalizes missed returns
    - Maintains rally length counter
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.env = myo_gym.make(cfg.env_id)

        # Episode bookkeeping
        self.max_steps = cfg.episode_len  # Typically 300 timesteps (3 seconds)
        self.step_count = 0
        self.total_steps = 0

        # Multi-hit tracking
        self.hit_count = 0
        self.rally_lengths = deque(maxlen=100)  # Track recent rally lengths
        self.current_rally_active = False
        self.last_hit_timestep = 0
        
        # Dynamic curriculum
        self.current_phase = 1
        self.phase_success_buffer = deque(maxlen=50)

        # Shaping memory
        self.prev_reach_dist = None

        # Spaces
        obs, _ = self.env.reset()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    # --------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed)
        obs_dict = self.env.obs_dict

        self.step_count = 0
        self.hit_count = 0
        self.current_rally_active = False
        self.last_hit_timestep = 0

        # Initialize distance tracking
        ball_pos = obs_dict["ball_pos"]
        paddle_pos = obs_dict["paddle_pos"]
        self.prev_reach_dist = float(np.linalg.norm(ball_pos - paddle_pos))

        return obs, {}

    # --------------------------------------------------
    def _check_opponent_landing(self, obs_dict):
        """Check if ball landed on opponent's side (x > 0) and on table"""
        ball_pos = obs_dict["ball_pos"]
        ball_vel = obs_dict["ball_vel"]
        
        # Ball on opponent's side, descending, near table height
        if (ball_pos[0] > 0.5 and ball_vel[2] < -0.2):
            TABLE_HEIGHT = 0.76  # ITTF table height
            if abs(ball_pos[2] - TABLE_HEIGHT) < 0.15:
                return True
        return False

    # --------------------------------------------------
    def _check_missed_ball(self, obs_dict):
        """Check if ball passed paddle without being hit"""
        ball_pos = obs_dict["ball_pos"]
        ball_vel = obs_dict["ball_vel"]
        
        # Ball is on agent's side but moving away (negative X) and below net height
        if ball_pos[0] < -0.5 and ball_vel[0] < -0.5 and ball_vel[2] < 0:
            return True
        return False

    # --------------------------------------------------
    def _update_curriculum(self):
        """Dynamic phase transitions based on rally lengths"""
        avg_rally = np.mean(self.rally_lengths) if self.rally_lengths else 0.0
        
        if self.current_phase == 1 and avg_rally >= 1.5:
            self.current_phase = 2
            print(f"🏆 Phase 2: Sustained Rally (avg hits: {avg_rally:.1f})")
            
        elif self.current_phase == 2 and avg_rally >= 3.0:
            self.current_phase = 3
            print(f"🏆 Phase 3: Extended Rally (avg hits: {avg_rally:.1f})")

    # --------------------------------------------------
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs_dict = self.env.obs_dict

        self.step_count += 1
        self.total_steps += 1

        # ==================================================
        # Dynamic Curriculum
        # ==================================================
        if self.total_steps % 500 == 0:
            self._update_curriculum()

        # ==================================================
        # Geometry
        # ==================================================
        ball_pos = obs_dict["ball_pos"]
        ball_vel = obs_dict["ball_vel"]
        paddle_pos = obs_dict["paddle_pos"]
        paddle_vel = obs_dict["paddle_vel"]
        touching_info = obs_dict["touching_info"]

        reach_dist = float(np.linalg.norm(ball_pos - paddle_pos))
        ball_to_paddle = paddle_pos - ball_pos
        incoming = float(np.dot(ball_vel, ball_to_paddle) < -0.1)
        ball_speed = float(np.linalg.norm(ball_vel))
        swing_speed = float(np.linalg.norm(paddle_vel))

        # ==================================================
        # Reward Components
        # ==================================================
        rwd = info.get("rwd_dict", {})
        act_reg = float(rwd.get("act_reg", 0.0))
        dense = float(rwd.get("dense", 0.0))

        # ==================================================
        # Custom Shaping (Multi-Hit Optimized)
        # ==================================================
        custom = 0.0

        # ---- (1) Reach progress ----
        if self.current_phase <= 2 and incoming:
            progress = self.prev_reach_dist - reach_dist
            custom += 1.5 * np.clip(progress, -0.05, 0.05)

        # ---- (2) Commit velocity ----
        if incoming and reach_dist < 0.18:
            approach_vel = np.dot(paddle_vel, ball_to_paddle) / (reach_dist + 1e-6)
            custom += 0.5 * np.clip(approach_vel, 0.0, 3.0)

        # ---- (3) Anti-hover ----
        if self.current_phase >= 2 and incoming and reach_dist < 0.12:
            custom -= 0.15  # Stronger penalty

        # ---- (4) Muscle efficiency ----
        custom -= 0.02 * np.sum(np.square(action))

        # ==================================================
        # Contact & Rally Logic (MULTI-HIT SUPPORT)
        # ==================================================
        
        # --- Invalid contacts ---
        ground_contact = touching_info[3] > 0.5
        net_contact = touching_info[4] > 0.5
        
        if ground_contact or net_contact:
            custom -= 30.0
            truncated = True
            if self.current_rally_active:
                self.rally_lengths.append(self.hit_count)
                self.phase_success_buffer.append(0.0)

        # --- Paddle contact (HIT) ---
        paddle_contact = touching_info[0] > 0.5
        if paddle_contact:
            # Start new rally on first hit
            if not self.current_rally_active:
                self.current_rally_active = True
                self.hit_count = 1
                self.last_hit_timestep = self.step_count
                custom += 10.0  # First hit bonus
            else:
                self.hit_count += 1
                custom += 15.0  # Subsequent hit bonus (higher!)

            # Direction reward (hit toward opponent)
            hit_dir = ball_vel[0] / (ball_speed + 1e-6)
            custom += 10.0 * np.clip(hit_dir, 0.0, 1.0)

            # Speed reward
            custom += 3.0 * np.tanh(ball_speed / 4.0)

        # --- Check for successful landing ---
        if self.current_rally_active:
            if self._check_opponent_landing(obs_dict):
                custom += 50.0 * self.hit_count  # Scale with rally length
                self.current_rally_active = False
                self.rally_lengths.append(self.hit_count)
                self.phase_success_buffer.append(1.0)
                
            elif self._check_missed_ball(obs_dict):
                # Ball passed paddle without hit
                custom -= 10.0
                self.current_rally_active = False
                self.rally_lengths.append(self.hit_count)
                self.phase_success_buffer.append(0.0)

        # ==================================================
        # FINAL REWARD (Phase-Dependent)
        # ==================================================
        
        if self.current_phase == 1:
            reward = (
                custom * 1.0
                + 2.0 * np.exp(-4.0 * reach_dist)
                + 0.5 * dense
                + act_reg
            )
        elif self.current_phase == 2:
            reward = (
                custom * 1.5
                + 1.0 * dense
                + act_reg
            )
        else:  # Phase 3
            reward = (
                custom * 2.0
                + 0.5 * dense
                + act_reg
            )

        # ==================================================
        # Logging
        # ==================================================
        info.update({
            "phase": self.current_phase,
            "hit_count": self.hit_count,
            "rally_active": int(self.current_rally_active),
            "avg_rally_length": np.mean(self.rally_lengths) if self.rally_lengths else 0.0,
            "success_rate": np.mean(self.phase_success_buffer) if self.phase_success_buffer else 0.0,
            "custom_reward": float(custom),
        })

        return obs, reward, terminated, truncated, info