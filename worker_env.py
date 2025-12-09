# worker_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config
from hrl_utils import flatten_myo_obs_worker, build_worker_obs, intrinsic_reward


class WorkerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Config):
        super().__init__()
        from myosuite.utils import gym as myo_gym

        self.config = config
        self.base_env = myo_gym.make(config.env_id, obs_keys="dict")

        obs_dict, _ = self.base_env.reset()
        base_vec = flatten_myo_obs_worker(obs_dict)
        self.base_dim = base_vec.shape[0]

        low = np.concatenate([
            -np.inf * np.ones(self.base_dim, dtype=np.float32),
            -np.inf * np.ones(config.goal_dim, dtype=np.float32),
            np.array([0.0])
        ])
        high = np.concatenate([
            np.inf * np.ones(self.base_dim, dtype=np.float32),
            np.inf * np.ones(config.goal_dim, dtype=np.float32),
            np.array([1.0])
        ])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = self.base_env.action_space

        self.goal = np.zeros(config.goal_dim, dtype=np.float32)
        self.t = 0

    def _sample_goal(self, obs_dict):
        return np.random.normal(0, self.config.goal_std, size=self.config.goal_dim).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs_dict, info = self.base_env.reset()
        self.goal = self._sample_goal(obs_dict)
        self.t = 0
        return build_worker_obs(obs_dict, self.goal, self.t, self.config), info

    def step(self, action):
        obs_dict, _, terminated, truncated, info = self.base_env.step(action)
        self.t = (self.t + 1) % self.config.high_level_period
        r_int = intrinsic_reward(obs_dict, self.goal)

        obs = build_worker_obs(obs_dict, self.goal, self.t, self.config)
        done = terminated or truncated
        return obs, r_int, done, False, info
