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
        self.base_env = myo_gym.make(config.env_id)

        _, _ = self.base_env.reset()
        obs_dict = self.base_env.obs_dict
        
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

    def reset(self, **kwargs):
        obs_vec, info = self.base_env.reset(**kwargs)
        obs_dict = self.base_env.obs_dict
        flat = flatten_myo_obs_worker(obs_dict)
        return flat, info

    def step(self, action):
        obs_vec, reward, terminated, truncated, info = self.base_env.step(action)
        obs_dict = self.base_env.obs_dict
        flat = flatten_myo_obs_worker(obs_dict)
        return flat, reward, terminated, truncated, info
