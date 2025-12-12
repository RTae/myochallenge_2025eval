import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from loguru import logger

from config import Config
from hrl_utils import flatten_myo_obs_manager, build_worker_obs


class ManagerEnv(gym.Env):
    """
    High-level HRL manager.
    Action = 3D goal vector (goal_dim)
    Obs    = compact manager state (16D)
    Reward = env reward + dense shaping
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

        logger.info(f"[ManagerEnv] Loading worker model: {worker_model_path}")
        self.worker = PPO.load(worker_model_path)

        # freeze worker parameters
        for param in self.worker.policy.parameters():
            param.requires_grad = False

        assert not any(p.requires_grad for p in self.worker.policy.parameters()), \
            "Worker model parameters are not frozen!"

        # infer manager obs shape
        self.base_env.reset()
        obs_dict = self.base_env.obs_dict
        self.last_obs = obs_dict

        mgr_flat = flatten_myo_obs_manager(obs_dict)
        logger.info(f"[ManagerEnv] manager_obs_dim = {mgr_flat.shape[0]} (expected ~16)")

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

    def reset(self, *, seed=None, options=None):
        self.base_env.reset(seed=seed)
        obs_dict = self.base_env.obs_dict
        self.last_obs = obs_dict
        return flatten_myo_obs_manager(obs_dict), {}

    def step(self, goal):
        goal = np.clip(goal, -0.15, 0.15).astype(np.float32)

        total_reward = 0.0
        obs_dict = self.last_obs

        for t in range(self.cfg.high_level_period):
            worker_obs = build_worker_obs(
                obs_dict=obs_dict,
                goal=goal,
                t_in_macro=t,
                cfg=self.cfg,
            ).reshape(1, -1)

            action_low, _ = self.worker.predict(worker_obs, deterministic=True)
            action_low = action_low.reshape(-1)

            _, _, terminated, truncated, _ = self.base_env.step(action_low)
            obs_dict = self.base_env.obs_dict

            rel = obs_dict["paddle_pos"] - obs_dict["ball_pos"]
            goal_err = np.linalg.norm(rel - goal)

            total_reward += -goal_err   # ← pure shaping

        self.last_obs = obs_dict
        mgr_obs = flatten_myo_obs_manager(obs_dict)

        return mgr_obs, total_reward, False, False, {}