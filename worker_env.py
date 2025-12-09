# worker_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config
from hrl_utils import flatten_myo_obs_worker


class WorkerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Config):
        super().__init__()
        from myosuite.utils import gym as myo_gym

        self.config = config

        # ------------------------------------------
        # Base MyoSuite env
        # ------------------------------------------
        self.base_env = myo_gym.make(config.env_id)

        # Reset once and capture obs_dict
        _, _ = self.base_env.reset()
        obs_dict = self.base_env.obs_dict

        # ------------------------------------------
        # Build observation + action spaces
        # ------------------------------------------
        base = flatten_myo_obs_worker(obs_dict)       # (429,)
        goal_dummy = np.zeros(config.goal_dim, dtype=np.float32)   # (3,)
        phase_dummy = np.array([0.0], dtype=np.float32)            # (1,)

        example_obs = np.concatenate([base, goal_dummy, phase_dummy], axis=-1)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=example_obs.shape, dtype=np.float32
        )

        self.action_space = self.base_env.action_space

    # ============================================================
    # INTERNAL: Construct worker observation
    # ============================================================
    def _build_obs(self, obs_dict):
        """
        Worker training is *not HRL*. So worker should NOT receive:
            - goal
            - phase
        But we add zero placeholders so ManagerEnv can reuse worker policy later.

        Worker observation = [base_obs, goal=0, phase=0]
        """
        base = flatten_myo_obs_worker(obs_dict)         # (429,)
        goal = np.zeros(self.config.goal_dim, dtype=np.float32)  # (3,)
        phase = np.array([0.0], dtype=np.float32)       # (1,)

        return np.concatenate([base, goal, phase], axis=-1).astype(np.float32)

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
