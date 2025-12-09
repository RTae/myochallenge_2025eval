# manager_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
import os

from config import Config
from hrl_utils import flatten_myo_obs_manager, build_worker_obs


class ManagerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Config, worker_model_path="worker.zip"):
        super().__init__()
        from myosuite.utils import gym as myo_gym

        self.config = config

        # ------------------------------
        # Base MyoSuite environment
        # ------------------------------
        self.base_env = myo_gym.make(config.env_id)
        worker_model_path = os.path.abspath(worker_model_path)

        # ------------------------------
        # Load trained worker
        # ------------------------------
        self.worker = PPO.load(worker_model_path)

        # ------------------------------
        # Initialize environment
        # ------------------------------
        obs_vec, info = self.base_env.reset()

        # IMPORTANT: pull the real dict
        obs_dict = self.base_env.obs_dict

        self.last_obs = obs_dict

        # Flatten manager observation
        flat_obs = flatten_myo_obs_manager(obs_dict)

        # ------------------------------
        # Observation + Action space
        # ------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-config.goal_bound,
            high=config.goal_bound,
            shape=(config.goal_dim,),
            dtype=np.float32
        )

    # ============================================================
    # RESET
    # ============================================================
    def reset(self, *, seed=None, options=None):
        obs_vec, info = self.base_env.reset()

        # Pull dict
        obs_dict = self.base_env.obs_dict
        self.last_obs = obs_dict

        flat = flatten_myo_obs_manager(obs_dict)
        return flat, info

    # ============================================================
    # STEP
    # ============================================================
    def step(self, goal):
        goal = np.array(goal, dtype=np.float32)

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        obs_dict = self.last_obs

        # Manager acts every K steps
        for t in range(self.config.high_level_period):

            if terminated or truncated:
                break

            # --------------------------
            # Build WORKER OBSERVATION
            # --------------------------
            worker_obs = build_worker_obs(obs_dict, goal, t, self.config)
            worker_obs = worker_obs.reshape(1, -1)

            # Low-level action
            action_low, _ = self.worker.predict(worker_obs, deterministic=True)

            # Env step (returns array)
            obs_vec, r_env, terminated, truncated, info = self.base_env.step(action_low)

            # Pull dict again
            obs_dict = self.base_env.obs_dict

            total_reward += r_env

        # Save last obs for next HRL step
        self.last_obs = obs_dict

        done = terminated or truncated

        # Manager observes FLATTENED dict features
        flat_obs = flatten_myo_obs_manager(obs_dict)

        return flat_obs, total_reward, done, False, info
