import gymnasium as gym
import numpy as np
from loguru import logger
from myosuite.utils import gym as myo_gym
from collections import deque
from config import Config


class CustomEnv(gym.Env):
    """
    MyoChallenge Table Tennis – pelvis-aware, anti-undercut environment.

    Fixes:
    - Forces pelvis x/y usage
    - Prevents bending-only strategies
    - Encourages true interception + blocking
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.env = myo_gym.make(cfg.env_id)

        # Episode control
        self.max_steps = cfg.episode_len
        self.step_count = 0
        self.total_steps = 0

        # Rally tracking
        self.hit_count = 0
        self.rally_lengths = deque(maxlen=100)
        self.current_rally_active = False

        # Contact cooldown
        self.contact_cooldown = 0
        self.contact_cooldown_steps = 6

        # Curriculum
        self.current_phase = 1
        self.phase_success_buffer = deque(maxlen=50)

        # State memory
        self.prev_reach_dist = None
        self.prev_pelvis_pos = None

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
        self.contact_cooldown = 0

        ball_pos = np.asarray(obs_dict["ball_pos"], dtype=np.float32)
        paddle_pos = np.asarray(obs_dict["paddle_pos"], dtype=np.float32)
        pelvis_pos = np.asarray(obs_dict["pelvis_pos"], dtype=np.float32)

        self.prev_reach_dist = float(np.linalg.norm(ball_pos - paddle_pos))
        self.prev_pelvis_pos = pelvis_pos.copy()

        return obs, {}

    # --------------------------------------------------
    def _update_curriculum(self):
        avg_rally = float(np.mean(self.rally_lengths)) if self.rally_lengths else 0.0
        if self.current_phase == 1 and avg_rally >= 1.5:
            self.current_phase = 2
            logger.info("🏆 Phase 2 unlocked")
        elif self.current_phase == 2 and avg_rally >= 3.0:
            self.current_phase = 3
            logger.info("🏆 Phase 3 unlocked")

    # --------------------------------------------------
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs_dict = self.env.obs_dict

        self.step_count += 1
        self.total_steps += 1
        if self.total_steps % 500 == 0:
            self._update_curriculum()

        # ==================================================
        # Geometry
        # ==================================================
        ball_pos = np.asarray(obs_dict["ball_pos"], dtype=np.float32)
        ball_vel = np.asarray(obs_dict["ball_vel"], dtype=np.float32)
        paddle_pos = np.asarray(obs_dict["paddle_pos"], dtype=np.float32)
        paddle_vel = np.asarray(obs_dict["paddle_vel"], dtype=np.float32)
        pelvis_pos = np.asarray(obs_dict["pelvis_pos"], dtype=np.float32)
        touching_info = np.asarray(obs_dict["touching_info"], dtype=np.float32)

        rel = ball_pos - paddle_pos
        reach_dist = float(np.linalg.norm(rel))
        lateral_dist = float(np.linalg.norm(rel[:2]))
        vertical_error = float(ball_pos[2] - paddle_pos[2])

        ball_speed = float(np.linalg.norm(ball_vel))

        incoming = float(np.dot(ball_vel, rel) < -0.05)

        pelvis_delta = pelvis_pos[:2] - self.prev_pelvis_pos[:2]
        pelvis_speed_xy = float(np.linalg.norm(pelvis_delta))
        self.prev_pelvis_pos = pelvis_pos.copy()

        # Reward components
        rwd = (info or {}).get("rwd_dict", {})
        dense = float(rwd.get("dense", 0.0))
        act_reg = float(rwd.get("act_reg", 0.0))

        custom = 0.0

        # (1) Reward pelvis motion toward ball (far range)
        if incoming and lateral_dist > 0.25:
            dir_xy = rel[:2] / (np.linalg.norm(rel[:2]) + 1e-6)
            pelvis_toward = float(np.dot(pelvis_delta, dir_xy))
            custom += 2.5 * np.clip(pelvis_toward, 0.0, 0.03)

        # (2) Penalize bending near ball without pelvis motion
        if incoming and lateral_dist < 0.25 and pelvis_speed_xy < 0.006:
            custom -= 2.0

        # (3) Anti-undercut
        if incoming and lateral_dist < 0.22 and vertical_error > 0.02:
            custom -= 6.0 * np.clip(vertical_error, 0.0, 0.12)

        # (4) Block bonus
        if incoming and lateral_dist < 0.18 and paddle_pos[2] >= ball_pos[2] - 0.01:
            custom += 1.2

        # (5) Reach progress (secondary)
        if incoming and self.prev_reach_dist is not None:
            progress = self.prev_reach_dist - reach_dist
            custom += 1.0 * np.clip(progress, -0.04, 0.04)

        self.prev_reach_dist = reach_dist

        # (6) Energy regularization
        custom -= 0.005 * float(np.sum(np.square(action)))

        # ==================================================
        # Contact logic
        # ==================================================
        if self.contact_cooldown > 0:
            self.contact_cooldown -= 1

        paddle_contact = (touching_info[0] > 0.5) and (self.contact_cooldown == 0)

        if paddle_contact:
            self.contact_cooldown = self.contact_cooldown_steps
            self.hit_count += 1
            self.current_rally_active = True
            custom += 12.0

            if ball_speed > 1e-6:
                custom += 6.0 * np.clip(ball_vel[0] / ball_speed, 0.0, 1.0)

            custom += 2.0 * np.tanh(ball_speed / 4.0)

        # Successful return (official signal)
        if self.current_rally_active and info.get("rwd_sparse", False):
            custom += 40.0 * self.hit_count
            self.rally_lengths.append(self.hit_count)
            self.phase_success_buffer.append(1.0)
            self.current_rally_active = False

        # Failed rally
        if self.current_rally_active and info.get("done", False):
            custom -= 10.0
            self.rally_lengths.append(self.hit_count)
            self.phase_success_buffer.append(0.0)
            self.current_rally_active = False

        # ==================================================
        # Final reward
        # ==================================================
        phase_scale = {1: 1.0, 2: 1.3, 3: 1.6}[self.current_phase]
        reward = (
            phase_scale * custom
            + 0.8 * dense
            + 0.2 * act_reg
        )

        if self.step_count >= self.max_steps:
            truncated = True

        # ==================================================
        # Logging
        # ==================================================
        info = info or {}
        info.update({
            "phase": self.current_phase,
            "hit": int(paddle_contact),
            "hit_count": self.hit_count,
            "incoming": incoming,
            "pelvis_speed_xy": pelvis_speed_xy,
            "lateral_dist": lateral_dist,
            "vertical_error": vertical_error,
            "ball_speed": ball_speed,
            "custom_reward": float(custom),
        })

        return obs, float(reward), terminated, truncated, info
