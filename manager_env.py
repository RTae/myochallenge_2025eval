# manager_env.py
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from config import Config
from hrl_utils import flatten_myo_obs_manager, build_worker_obs


class ManagerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Config, worker_model_path: str):
        super().__init__()
        from myosuite.utils import gym as myo_gym

        self.config = config
        self.base_env = myo_gym.make(config.env_id)

        # ------------------------------
        # Load trained worker (goal-conditioned, obs_dim=428)
        # ------------------------------
        worker_model_path = os.path.abspath(worker_model_path)
        if not worker_model_path.endswith(".zip"):
            worker_model_path += ".zip"
        if not os.path.exists(worker_model_path):
            raise FileNotFoundError(f"Worker model not found at: {worker_model_path}")

        self.worker = PPO.load(worker_model_path)

        # ------------------------------
        # Init manager obs shape
        # ------------------------------
        obs_vec, info = self.base_env.reset()
        obs_dict = self.base_env.obs_dict
        self.last_obs = obs_dict

        mgr_flat = flatten_myo_obs_manager(obs_dict)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=mgr_flat.shape,
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-config.goal_bound,
            high=config.goal_bound,
            shape=(config.goal_dim,),
            dtype=np.float32,
        )

    # ============================================================
    # RESET
    # ============================================================
    def reset(self, *, seed=None, options=None):
        obs_vec, info = self.base_env.reset(seed=seed)
        obs_dict = self.base_env.obs_dict
        self.last_obs = obs_dict

        mgr_flat = flatten_myo_obs_manager(obs_dict)
        return mgr_flat, info

    # ============================================================
    # STEP
    # ============================================================
    def step(self, goal):
        goal = np.asarray(goal, dtype=np.float32)

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        obs_dict = self.last_obs

        # Manager goal is fixed for high_level_period worker steps
        for t in range(self.config.high_level_period):
            if terminated or truncated:
                break

            worker_obs = build_worker_obs(
                obs_dict=obs_dict,
                goal=goal,
                t=t,
                cfg=self.config,
            ).reshape(1, -1)

            action_low, _ = self.worker.predict(worker_obs, deterministic=True)

            obs_vec, r_env, terminated, truncated, info = self.base_env.step(action_low)
            obs_dict = self.base_env.obs_dict

            total_reward += r_env

        self.last_obs = obs_dict
        done = terminated or truncated

        mgr_flat = flatten_myo_obs_manager(obs_dict)
        return mgr_flat, total_reward, done, False, info
