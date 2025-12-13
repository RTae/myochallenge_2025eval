import gymnasium as gym
import numpy as np
from myosuite.utils import gym as myo_gym
from collections import deque
from config import Config


class CustomEnv(gym.Env):
    """
    Multi-hit rally environment for MyoChallenge 2025.

    Changes in this version:
    - Fix incoming detection
    - Add anti-undercut + block shaping
    - Add close-range miss penalty (forces actual contact attempt)
    - Add contact cooldown (prevents multi-counting one touch)
    - Slightly safer termination / rally bookkeeping
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.env = myo_gym.make(cfg.env_id)

        # Episode bookkeeping
        self.max_steps = cfg.episode_len
        self.step_count = 0
        self.total_steps = 0

        # Multi-hit tracking
        self.hit_count = 0
        self.rally_lengths = deque(maxlen=100)
        self.current_rally_active = False
        self.last_hit_timestep = 0

        # Contact cooldown (avoid double-counting a single contact)
        self.contact_cooldown = 0
        self.contact_cooldown_steps = 6  # ~6 frames: tune 4-10

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
        self.contact_cooldown = 0

        ball_pos = np.asarray(obs_dict["ball_pos"], dtype=np.float32)
        paddle_pos = np.asarray(obs_dict["paddle_pos"], dtype=np.float32)
        self.prev_reach_dist = float(np.linalg.norm(ball_pos - paddle_pos))

        return obs, {}

    # --------------------------------------------------
    def _check_missed_ball(self, obs_dict):
        """
        More reliable miss:
        - ball was incoming (toward paddle) recently, now it has passed behind paddle (x much smaller than paddle x),
          and it's moving away from paddle.
        """
        ball_pos = np.asarray(obs_dict["ball_pos"], dtype=np.float32)
        ball_vel = np.asarray(obs_dict["ball_vel"], dtype=np.float32)
        paddle_pos = np.asarray(obs_dict["paddle_pos"], dtype=np.float32)

        rel = ball_pos - paddle_pos
        # passed behind paddle in x and moving further away (rel dot vel > 0)
        if rel[0] < -0.25 and np.dot(ball_vel, rel) > 0.0 and ball_pos[2] < 1.0:
            return True
        return False

    # --------------------------------------------------
    def _update_curriculum(self):
        """Dynamic phase transitions based on avg rally length."""
        avg_rally = float(np.mean(self.rally_lengths)) if self.rally_lengths else 0.0

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

        # update curriculum sometimes
        if self.total_steps % 500 == 0:
            self._update_curriculum()

        # ==================================================
        # Geometry
        # ==================================================
        ball_pos = np.asarray(obs_dict["ball_pos"], dtype=np.float32)
        ball_vel = np.asarray(obs_dict["ball_vel"], dtype=np.float32)
        paddle_pos = np.asarray(obs_dict["paddle_pos"], dtype=np.float32)
        paddle_vel = np.asarray(obs_dict["paddle_vel"], dtype=np.float32)
        touching_info = np.asarray(obs_dict["touching_info"], dtype=np.float32)

        rel = ball_pos - paddle_pos               # paddle -> ball vector
        reach_dist = float(np.linalg.norm(rel))
        ball_speed = float(np.linalg.norm(ball_vel))
        swing_speed = float(np.linalg.norm(paddle_vel))

        # ✅ Correct incoming: ball moving toward paddle means ball_vel points opposite rel => dot(ball_vel, rel) < 0
        incoming = float(np.dot(ball_vel, rel) < -0.05)

        # Height relation (positive => paddle below ball)
        vertical_error = float(ball_pos[2] - paddle_pos[2])
        horiz_dist = float(np.linalg.norm(rel[:2]))

        # ==================================================
        # Reward Components from MyoSuite
        # ==================================================
        rwd = (info or {}).get("rwd_dict", {})
        act_reg = float(rwd.get("act_reg", 0.0))
        dense = float(rwd.get("dense", 0.0))

        # ==================================================
        # Custom shaping (Multi-hit optimized + anti-undercut)
        # ==================================================
        custom = 0.0

        # (1) Reach progress (only when incoming)
        if self.current_phase <= 2 and incoming:
            if self.prev_reach_dist is not None:
                progress = self.prev_reach_dist - reach_dist
                custom += 1.5 * np.clip(progress, -0.05, 0.05)

        self.prev_reach_dist = reach_dist

        # (2) Commit velocity (move paddle toward ball)
        if incoming and reach_dist < 0.20:
            # Move paddle in direction of rel (toward ball) => dot(paddle_vel, rel) > 0
            approach_vel = float(np.dot(paddle_vel, rel) / (reach_dist + 1e-6))
            custom += 0.4 * np.clip(approach_vel, 0.0, 3.0)

        # (3) ✅ Anti-undercut: when close & incoming, punish being under the ball
        if incoming and reach_dist < 0.22:
            if vertical_error > 0.02:
                custom -= 6.0 * np.clip(vertical_error, 0.0, 0.12)

        # (4) ✅ Block bonus: encourage paddle to be at/above ball height when close
        if incoming and reach_dist < 0.18:
            if paddle_pos[2] > ball_pos[2] - 0.01:
                custom += 1.0

        # (5) Anti-hover (but not too strong; don’t kill exploration)
        if self.current_phase >= 2 and incoming and reach_dist < 0.12:
            custom -= 0.08

        # (6) Energy regularization (keep small)
        custom -= 0.005 * float(np.sum(np.square(action)))

        # ==================================================
        # Contact & Rally Logic
        # ==================================================
        if self.contact_cooldown > 0:
            self.contact_cooldown -= 1

        ground_contact = touching_info[3] > 0.5
        net_contact = touching_info[4] > 0.5
        env_contact = touching_info[5] > 0.5

        # Terminate harsh failures only if rally was active or phase>=2 (avoid early random death spirals)
        if (ground_contact or net_contact) and (self.current_phase >= 2 or self.current_rally_active):
            custom -= 20.0
            truncated = True
            if self.current_rally_active:
                self.rally_lengths.append(self.hit_count)
                self.phase_success_buffer.append(0.0)
                self.current_rally_active = False

        # Paddle contact (HIT) with cooldown gate
        paddle_contact = (touching_info[0] > 0.5) and (self.contact_cooldown == 0)

        if paddle_contact:
            self.contact_cooldown = self.contact_cooldown_steps

            if not self.current_rally_active:
                self.current_rally_active = True
                self.hit_count = 1
                self.last_hit_timestep = self.step_count
                custom += 12.0
            else:
                self.hit_count += 1
                custom += 10.0

            # Reward sending ball toward opponent (+x)
            if ball_speed > 1e-6:
                hit_dir = float(ball_vel[0] / (ball_speed + 1e-6))
                custom += 6.0 * np.clip(hit_dir, 0.0, 1.0)

            # Reward speed after contact (but saturate)
            custom += 2.0 * np.tanh(ball_speed / 4.0)

        # ✅ Close-range miss penalty (forces real interception attempts)
        # If ball is incoming and close but we fail to contact for a while, punish it.
        if incoming and reach_dist < 0.18 and not paddle_contact:
            custom -= 0.6

        # Successful opponent landing ends rally
        if self.current_rally_active:

            if info.get("rwd_sparse", False):
                # TRUE successful return
                custom += 40.0 * self.hit_count
                self.current_rally_active = False
                self.rally_lengths.append(self.hit_count)
                self.phase_success_buffer.append(1.0)

            elif info.get("done", False):
                # Rally ended but failed
                custom -= 10.0
                self.current_rally_active = False
                self.rally_lengths.append(self.hit_count)
                self.phase_success_buffer.append(0.0)


        # ==================================================
        # Final reward
        # ==================================================
        if self.current_phase == 1:
            reward = (
                1.0 * custom
                + 0.5 * dense
                + 0.2 * act_reg
            )
        elif self.current_phase == 2:
            reward = (
                1.3 * custom
                + 0.8 * dense
                + 0.2 * act_reg
            )
        else:
            reward = (
                1.6 * custom
                + 1.0 * dense
                + 0.2 * act_reg
            )

        # ==================================================
        # Episode cutoff
        # ==================================================
        if self.step_count >= self.max_steps:
            truncated = True

        # ==================================================
        # Logging
        # ==================================================
        info = info or {}
        info.update({
            "phase": self.current_phase,
            "hit_count": int(self.hit_count),
            "hit": int(paddle_contact),
            "rally_active": int(self.current_rally_active),
            "avg_rally_length": float(np.mean(self.rally_lengths)) if self.rally_lengths else 0.0,
            "success_rate": float(np.mean(self.phase_success_buffer)) if self.phase_success_buffer else 0.0,
            "incoming": incoming,
            "reach_dist": reach_dist,
            "vertical_error": vertical_error,
            "horiz_dist": horiz_dist,
            "ball_speed": ball_speed,
            "custom_reward": float(custom),
            "rwd_dense": float(dense),
            "act_reg": float(act_reg),
        })

        return obs, float(reward), terminated, truncated, info
