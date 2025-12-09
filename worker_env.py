# worker_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config
from hrl_utils import build_worker_obs


class WorkerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Config):
        super().__init__()
        from myosuite.utils import gym as myo_gym

        self.config = config

        # Base MyoSuite environment
        self.base_env = myo_gym.make(config.env_id)

        # Reset once to infer obs shape
        _, _ = self.base_env.reset()
        obs_dict = self.base_env.obs_dict

        zero_goal = np.zeros(config.goal_dim, dtype=np.float32)
        example_obs = build_worker_obs(
            obs_dict=obs_dict,
            goal=zero_goal,
            t_in_macro=0,
            cfg=config,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=example_obs.shape,
            dtype=np.float32,
        )

        self.action_space = self.base_env.action_space

    # ============================================================
    # INTERNAL: construct worker observation
    # ============================================================
    def _build_obs(self, obs_dict):
        zero_goal = np.zeros(self.config.goal_dim, dtype=np.float32)
        return build_worker_obs(
            obs_dict=obs_dict,
            goal=zero_goal,
            t_in_macro=0,
            cfg=self.config,
        )

    # ============================================================
    # RESET
    # ============================================================
    def reset(self, *, seed=None, options=None):
        obs_vec, info = self.base_env.reset(seed=seed)
        obs_dict = self.base_env.obs_dict
        worker_obs = self._build_obs(obs_dict)
        return worker_obs, info

    # ============================================================
    # STEP
    # ============================================================
    def step(self, action):
        obs_vec, reward, terminated, truncated, info = self.base_env.step(action)
        obs_dict = self.base_env.obs_dict
        worker_obs = self._build_obs(obs_dict)
        return worker_obs, reward, terminated, truncated, info
