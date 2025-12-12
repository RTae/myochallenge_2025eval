# single_policy_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from myosuite.utils import gym as myo_gym
from utils import flatten_myo_obs_worker, HitDetector
from config import Config


class CustomEnv(gym.Env):
    """
    Single-policy PPO environment for MyoSuite table tennis.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.env = myo_gym.make(cfg.env_id)

        self.hit_detector = HitDetector(dv_thr=0.25, ball_mass=cfg.BALL_MASS, paddle_face_radius=cfg.PADDLE_FACE_RADIUS)
        self.max_steps = cfg.worker_episode_len
        self.step_count = 0

        # Infer obs shape
        self.env.reset()
        obs_dict = self.env.obs_dict
        obs = flatten_myo_obs_worker(obs_dict)

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

        return flatten_myo_obs_worker(obs_dict), {}

    # --------------------------------------------------
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs_dict = info.get("obs_dict", {}) if info else self.env.obs_dict

        self.step_count += 1

        # -----------------------------
        # Hit detection
        # -----------------------------
        hit, contact_force, dv = self.hit_detector.step(obs_dict)

        ball_pos = obs_dict["ball_pos"]
        paddle_pos = obs_dict["paddle_pos"]
        paddle_vel = obs_dict["paddle_vel"]
        ball_vel = obs_dict["ball_vel"]

        dist = np.linalg.norm(ball_pos - paddle_pos)

        reward = 0.0

        # -----------------------------
        # Phase-based shaping (implicit)
        # -----------------------------
        if dist > 0.6:
            # approach
            rel = ball_pos - paddle_pos
            if np.dot(ball_vel, rel) < 0:
                reward += 0.1 * np.dot(paddle_vel, rel) / (np.linalg.norm(rel) + 1e-6)

        elif dist > 0.25:
            # preparation
            reward += 0.05 * np.linalg.norm(paddle_vel)

        else:
            # strike zone
            reward += -0.05
            if hit:
                reward += 15.0
                reward += np.tanh(dv / 3.0)
                reward += np.tanh(contact_force / 20.0)

        # energy penalty
        act = np.array(obs_dict.get("act", []), dtype=np.float32)
        reward += -0.001 * np.sum(act ** 2)

        # episode cutoff
        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "hit": hit,
            "dv": dv,
            "contact_force": contact_force,
            "dist": dist,
        }

        return flatten_myo_obs_worker(obs_dict), reward, terminated, truncated, info
