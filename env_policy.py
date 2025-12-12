import gymnasium as gym
import numpy as np
from myosuite.utils import gym as myo_gym
from config import Config


class CustomEnv(gym.Env):
    """
    Custom Environment Wrapper for MyoSuite (MyoChallenge Table Tennis).

    - Policy observes raw Gym obs (flat vector)
    - Reward uses obs_dict (semantic, stable)
    - No opponent / no rally logic (single-agent focus)
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

        # Shaping state
        self.prev_reach_dist = None
        self.hit_once = False

        # ---- Spaces ----
        obs, _ = self.env.reset()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Curriculum phase
        self.phase = 1

    # --------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed)
        obs_dict = self.env.obs_dict

        self.step_count = 0
        self.hit_once = False

        reach_err = obs_dict["reach_err"]
        self.prev_reach_dist = float(np.linalg.norm(reach_err))

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
        if self.total_steps < 10_000_000:
            self.phase = 1
        elif self.total_steps < 20_000_000:
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
        muscle_act = np.asarray(obs_dict["act"], dtype=np.float32)
        touch = obs_dict["touching_info"]

        # Touch indices
        TOUCH_PADDLE = 0
        TOUCH_GROUND = 3
        TOUCH_NET = 4
        TOUCH_ENV = 5

        # -----------------------------
        # Geometry
        # -----------------------------
        reach_dist = float(np.linalg.norm(reach_err))
        ball_speed = float(np.linalg.norm(ball_vel))

        rel = ball_pos - paddle_pos
        incoming = float(np.dot(ball_vel, rel) < 0.0)

        # -----------------------------
        # Reward
        # -----------------------------
        reward = 0.0
        truncated = False

        # ========== (1) PRE-CONTACT SHAPING ==========
        if self.phase <= 2 and not self.hit_once and incoming:

            # Reach-error progress
            if self.prev_reach_dist is not None:
                progress = self.prev_reach_dist - reach_dist
                reward += 2.0 * np.clip(progress, -0.05, 0.05)

            # Paddle velocity toward ball (only when close)
            if reach_dist < 0.30:
                approach_speed = np.dot(paddle_vel, reach_err) / (reach_dist + 1e-6)
                reward += 0.1 * np.clip(approach_speed, 0.0, 1.0)

        self.prev_reach_dist = reach_dist

        # ========== (2) CONTACT ==========
        hit = touch[TOUCH_PADDLE] > 0.5

        if hit:
            self.hit_once = True

            # Base hit reward
            reward += 20.0 if self.phase == 1 else 15.0 if self.phase == 2 else 10.0

            # Post-hit quality
            if self.phase >= 2:
                reward += 1.0 * np.tanh(ball_speed / 3.0)
                reward += 1.0 * np.tanh(ball_vel[0] / 2.0)

            # Penalize weak taps
            if ball_speed < 0.5:
                reward -= 2.0

        # ========== (3) FAILURE PENALTIES ==========
        if self.phase >= 2:
            if touch[TOUCH_NET] > 0.5:
                reward -= 3.0
            if touch[TOUCH_GROUND] > 0.5:
                reward -= 3.0
            if touch[TOUCH_ENV] > 0.5:
                reward -= 1.5

        # ========== (4) ENERGY REGULARIZATION ==========
        reward -= 5e-5 * np.mean(muscle_act ** 2)

        # -----------------------------
        # Episode cutoff
        # -----------------------------
        if self.step_count >= self.max_steps:
            truncated = True

        # -----------------------------
        # Info
        # -----------------------------
        info = info or {}
        info.update({
            "phase": self.phase,
            "hit": int(hit),
            "reach_dist": float(reach_dist),
            "incoming": incoming,
            "ball_speed": float(ball_speed),
            "energy": float(np.mean(muscle_act ** 2)),
        })

        return obs, reward, terminated, truncated, info
