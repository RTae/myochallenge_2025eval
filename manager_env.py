# manager_env.py
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from config import Config
from hrl_utils import flatten_myo_obs_manager, build_worker_obs


class ManagerEnv(gym.Env):
    """
    High-level HRL manager.
    Action = 3D goal vector
    Obs = compact manager state (~16D)
    Reward = true MyoSuite environment reward
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config, worker_model_path: str):
        super().__init__()
        from myosuite.utils import gym as myo_gym

        self.cfg = config
        self.base_env = myo_gym.make(config.env_id)

        worker_model_path = os.path.abspath(worker_model_path)
        if not os.path.exists(worker_model_path):
            raise FileNotFoundError(f"Cannot load worker model: {worker_model_path}")

        print(f"[ManagerEnv] Loading worker model: {worker_model_path}")
        self.worker = PPO.load(worker_model_path)

        # Infer obs shape
        self.base_env.reset()
        obs_dict = self.base_env.obs_dict
        self.last_obs = obs_dict

        mgr_flat = flatten_myo_obs_manager(obs_dict)
        print(f"[ManagerEnv] manager_obs_dim = {mgr_flat.shape[0]}")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=mgr_flat.shape,
            dtype=np.float32,
        )

        # Manager outputs a goal
        self.action_space = spaces.Box(
            low=-config.goal_bound,
            high=config.goal_bound,
            shape=(config.goal_dim,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self.base_env.reset(seed=seed)
        obs_dict = self.base_env.obs_dict
        self.last_obs = obs_dict
        return flatten_myo_obs_manager(obs_dict), {}

    def step(self, goal):
        # Clip manager goal
        goal = np.clip(
            np.asarray(goal, dtype=np.float32),
            -self.cfg.goal_bound,
            self.cfg.goal_bound,
        )

        terminated = False
        truncated = False
        total_reward = 0.0

        obs_dict = self.last_obs

        # Manager goal is executed for K low-level steps
        for t in range(self.cfg.high_level_period):

            worker_obs = build_worker_obs(
                obs_dict=obs_dict,
                goal=goal,
                t_in_macro=t,
                cfg=self.cfg,
            ).reshape(1, -1)

            action_low, _ = self.worker.predict(worker_obs, deterministic=True)
            action_low = action_low.reshape(-1)

            _, r_env, terminated, truncated, info = self.base_env.step(action_low)
            obs_dict = self.base_env.obs_dict
            total_reward += r_env

            if terminated or truncated:
                break

        self.last_obs = obs_dict
        mgr_obs = flatten_myo_obs_manager(obs_dict)
        done = terminated or truncated

        return mgr_obs, total_reward, done, False, info
