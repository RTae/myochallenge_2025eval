import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from loguru import logger

from config import Config
from hrl_utils import flatten_myo_obs_manager, build_worker_obs, HitDetector


class ManagerEnv(gym.Env):
    """
    High-level HRL manager.
    Action = 3D goal vector (desired paddle velocity)
    Obs    = compact manager state (16D)
    Reward = velocity alignment + sparse hit success
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

        # Freeze worker parameters
        for param in self.worker.policy.parameters():
            param.requires_grad = False

        assert not any(p.requires_grad for p in self.worker.policy.parameters()), \
            "Worker model parameters are not frozen!"

        # Hit detector (manager-side, NOT from worker info)
        self.hit_detector = HitDetector(dv_thr=1.5)

        # Infer manager obs shape
        self.base_env.reset()
        obs_dict = self.base_env.obs_dict
        self.last_obs = obs_dict

        self.hit_detector.reset(obs_dict)

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
        self.hit_detector.reset(obs_dict)
        return flatten_myo_obs_manager(obs_dict), {}

    def step(self, goal):
        goal = np.clip(goal, -self.cfg.goal_bound, self.cfg.goal_bound).astype(np.float32)

        total_reward = 0.0
        obs_dict = self.last_obs
        terminated_any = False
        truncated_any = False
        hit_occurred = False

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

            # --------------------------------------------------
            # 1) Velocity-consistent shaping (normalized)
            # --------------------------------------------------
            paddle_vel = obs_dict["paddle_vel"]
            vel_err = np.linalg.norm(paddle_vel - goal)
            total_reward += -vel_err / self.cfg.high_level_period

            # --------------------------------------------------
            # 2) Sparse hit success (manager-side detection)
            # --------------------------------------------------
            hit, _, _ = self.hit_detector.step(obs_dict)
            if hit:
                # Early-hit bonus (encourages timing)
                bonus = 5.0 * (1.0 - t / self.cfg.high_level_period)
                total_reward += bonus
                hit_occurred = True

            terminated_any |= terminated
            truncated_any |= truncated
            if terminated or truncated:
                break

        # --------------------------------------------------
        # 3) Small regularization to avoid trivial zero goals
        # --------------------------------------------------
        total_reward -= 0.01 * np.linalg.norm(goal)

        self.last_obs = obs_dict
        mgr_obs = flatten_myo_obs_manager(obs_dict)

        info = {
            "hit_occurred": hit_occurred
        }

        return mgr_obs, total_reward, terminated_any, truncated_any, info