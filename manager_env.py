# manager_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from config import Config
from hrl_utils import flatten_myo_obs_manager, build_worker_obs


class ManagerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Config, worker_model_path="worker.zip"):
        super().__init__()
        from myosuite.utils import gym as myo_gym

        self.config = config
        self.base_env = myo_gym.make(config.env_id)
        self.worker = PPO.load(worker_model_path)

        obs_dict, _ = self.base_env.reset()
        self.last_obs = obs_dict

        vec = flatten_myo_obs_manager(obs_dict)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=vec.shape, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-config.goal_bound,
            high=config.goal_bound,
            shape=(config.goal_dim,),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        obs_dict, info = self.base_env.reset()
        self.last_obs = obs_dict
        return flatten_myo_obs_manager(obs_dict), info

    def step(self, goal):
        goal = np.array(goal, dtype=np.float32)
        total_reward = 0.0
        terminated = truncated = False

        obs_dict = self.last_obs
        for t in range(self.config.high_level_period):
            if terminated or truncated:
                break
            worker_obs = build_worker_obs(obs_dict, goal, t, self.config).reshape(1, -1)
            a_low, _ = self.worker.predict(worker_obs, deterministic=True)

            obs_dict, r_env, terminated, truncated, info = self.base_env.step(a_low)
            total_reward += r_env

        self.last_obs = obs_dict
        done = terminated or truncated
        return flatten_myo_obs_manager(obs_dict), total_reward, done, False, info
