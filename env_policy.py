import gymnasium as gym
import numpy as np
from gymnasium import spaces

from myosuite.utils import gym as myo_gym
from utils import flatten_obs, HitDetector
from config import Config


class CustomEnv(gym.Env):
    """
    Custom Environment Wrapper for MyoSuite environment with tailored reward structure.
    1. Progress-based shaping reward to encourage ball approach.
    2. Sparse hit reward upon successful paddle-ball contact.
    3. Energy regularization to penalize excessive actions.
    4. Instrumentation for detailed episode metrics.
    5. Episode truncation based on maximum step count.
    6. Observation flattening for compatibility with RL algorithms.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.env = myo_gym.make(cfg.env_id)

        self.hit_detector = HitDetector(
            dv_thr=0.25,
            ball_mass=cfg.BALL_MASS,
            paddle_face_radius=cfg.PADDLE_FACE_RADIUS,
        )

        self.max_steps = cfg.episode_len
        self.step_count = 0
        self.prev_dist = None

        # Infer obs shape
        self.env.reset()
        obs = flatten_obs(self.env.obs_dict)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )
        self.action_space = self.env.action_space

    # --------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        obs_dict = info.get("obs_dict", {}) if info else self.env.obs_dict

        self.hit_detector.reset(obs_dict)
        self.step_count = 0

        ball_pos = obs_dict["ball_pos"]
        paddle_pos = obs_dict["paddle_pos"]
        self.prev_dist = np.linalg.norm(ball_pos - paddle_pos)

        return flatten_obs(obs_dict), {}

    # --------------------------------------------------
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs_dict = info.get("obs_dict", {}) if info else self.env.obs_dict
        self.step_count += 1

        hit, contact_force, dv = self.hit_detector.step(obs_dict)

        ball_pos = obs_dict["ball_pos"]
        paddle_pos = obs_dict["paddle_pos"]
        ball_vel = obs_dict["ball_vel"]

        rel = ball_pos - paddle_pos
        dist = np.linalg.norm(rel)
        ball_speed = np.linalg.norm(ball_vel)

        incoming = float(np.dot(ball_vel, rel) < 0.0)

        # --------------------------------------------------
        # PROGRESS-BASED SHAPING (core fix)
        # --------------------------------------------------
        reward = 0.0
        if incoming and self.prev_dist is not None:
            progress = self.prev_dist - dist
            reward += 2.0 * np.clip(progress, -0.05, 0.05)

        self.prev_dist = dist

        # --------------------------------------------------
        # HIT REWARD (dominant, sparse)
        # --------------------------------------------------
        if hit:
            reward += 20.0
            reward += np.tanh(dv / 3.0)
            reward += np.tanh(contact_force / 20.0)

        # --------------------------------------------------
        # ENERGY REGULARIZATION
        # --------------------------------------------------
        act = np.array(obs_dict.get("act", []), dtype=np.float32)
        reward -= 0.001 * np.sum(act ** 2)

        # --------------------------------------------------
        # Episode cutoff
        # --------------------------------------------------
        if self.step_count >= self.max_steps:
            truncated = True

        # --------------------------------------------------
        # Instrumentation (Method 1)
        # --------------------------------------------------
        info = info or {}
        info.update({
            "hit": int(hit),
            "dv": float(dv),
            "contact_force": float(contact_force),
            "dist": float(dist),
            "incoming": incoming,
            "ball_speed": float(ball_speed),
        })

        return flatten_obs(obs_dict), reward, terminated, truncated, info
