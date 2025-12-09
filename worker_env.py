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

        # ------------------------------
        # Base MyoSuite environment
        # ------------------------------
        self.base_env = myo_gym.make(config.env_id)

        # Reset once to initialize dict
        obs_vec, info = self.base_env.reset()
        obs_dict = self.base_env.obs_dict

        # ------------------------------
        # Worker observation space
        # (Only flattened true MyoSuite obs)
        # ------------------------------
        flat = flatten_myo_obs_worker(obs_dict)
        self.obs_dim = flat.shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Action space = raw MuJoCo motor activations
        self.action_space = self.base_env.action_space


    # ============================================================
    # RESET
    # ============================================================
    def reset(self, *, seed=None, options=None):
        obs_vec, info = self.base_env.reset(seed=seed)
        obs_dict = self.base_env.obs_dict

        flat = flatten_myo_obs_worker(obs_dict)
        return flat, info


    # ============================================================
    # STEP
    # ============================================================
    def step(self, action):
        obs_vec, r_env, terminated, truncated, info = self.base_env.step(action)
        obs_dict = self.base_env.obs_dict

        flat = flatten_myo_obs_worker(obs_dict)
        return flat, r_env, terminated, truncated, info
