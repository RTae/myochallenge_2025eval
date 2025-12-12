import gymnasium as gym
import numpy as np
from gymnasium import spaces

from myosuite.utils import gym as myo_gym
from config import Config


class CustomEnv(gym.Env):
    """
    Custom Environment Wrapper for MyoSuite (MyoChallenge Table Tennis).

    - Policy observes raw Gym obs (flat vector)
    - Reward uses obs_dict (semantic, stable)
    - Contact via touching_info (not heuristics)
    - Curriculum-ready reward structure
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

        # For approach shaping
        self.prev_dist = None
        self.hit_once = False

        # ---- Spaces (IMPORTANT: use env spaces directly) ----
        obs, _ = self.env.reset()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Curriculum phase
        self.phase = 1

    # --------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)

        obs_dict = self.env.obs_dict

        self.step_count = 0
        self.hit_once = False

        ball_pos = obs_dict["ball_pos"]
        paddle_pos = obs_dict["paddle_pos"]
        self.prev_dist = np.linalg.norm(ball_pos - paddle_pos)

        return obs, {}

    # --------------------------------------------------
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs_dict = self.env.obs_dict

        self.step_count += 1
        self.total_steps += 1

        # -----------------------------
        # Curriculum scheduling
        # -----------------------------
        if self.total_steps < 2_000_000:
            self.phase = 1
        elif self.total_steps < 6_000_000:
            self.phase = 2
        else:
            self.phase = 3

        # -----------------------------
        # Extract from obs_dict
        # -----------------------------
        ball_pos = obs_dict["ball_pos"]
        ball_vel = obs_dict["ball_vel"]
        paddle_pos = obs_dict["paddle_pos"]
        paddle_vel = obs_dict["paddle_vel"]
        reach_err = obs_dict["reach_err"]
        muscle_act = obs_dict["muscle_activations"]
        touch = obs_dict["touching_info"]

        # Touch indices (from spec)
        TOUCH_PADDLE = 0
        TOUCH_OWN = 1
        TOUCH_OPP = 2
        TOUCH_GROUND = 3
        TOUCH_NET = 4
        TOUCH_ENV = 5

        # -----------------------------
        # Geometry
        # -----------------------------
        rel = ball_pos - paddle_pos
        dist = np.linalg.norm(rel)
        ball_speed = np.linalg.norm(ball_vel)
        incoming = float(np.dot(ball_vel, rel) < 0.0)

        # -----------------------------
        # Reward
        # -----------------------------
        reward = 0.0

        # (1) Pre-contact shaping (Phase 1–2 only)
        if self.phase <= 2 and not self.hit_once and incoming and self.prev_dist is not None:
            progress = self.prev_dist - dist
            reward += 2.0 * np.clip(progress, -0.05, 0.05)

        self.prev_dist = dist

        # (2) Paddle-ball contact
        hit = touch[TOUCH_PADDLE] > 0.5

        if hit:
            self.hit_once = True

            # Base hit reward
            reward += 20.0 if self.phase == 1 else 15.0 if self.phase == 2 else 10.0

            # Post-hit quality (Phase ≥2)
            if self.phase >= 2:
                reward += 1.0 * np.tanh(ball_speed / 3.0)
                reward += 1.0 * np.tanh(ball_vel[0] / 2.0)

        # (3) Failure penalties (Phase ≥2)
        if self.phase >= 2:
            if touch[TOUCH_NET] > 0.5:
                reward -= 3.0
            if touch[TOUCH_GROUND] > 0.5:
                reward -= 3.0
            if touch[TOUCH_ENV] > 0.5:
                reward -= 1.5

        # (4) Energy regularization (very small, always on)
        reward -= 5e-5 * np.sum(muscle_act ** 2)

        # -----------------------------
        # Episode cutoff
        # -----------------------------
        if self.step_count >= self.max_steps:
            truncated = True

        # -----------------------------
        # Info (for logging)
        # -----------------------------
        info = info or {}
        info.update({
            "phase": self.phase,
            "hit": int(hit),
            "dist": float(dist),
            "incoming": incoming,
            "ball_speed": float(ball_speed),
        })

        return obs, reward, terminated, truncated, info
