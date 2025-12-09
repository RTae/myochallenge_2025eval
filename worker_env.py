# worker_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config
from hrl_utils import build_worker_obs, intrinsic_reward

from loguru import logger


class WorkerEnv(gym.Env):
    """
    Low-level goal-conditioned worker.
    Obs = [base_obs (424), goal (3), phase (1)] = (428,)
    Act = MyoSuite muscle activations
    Reward = intrinsic reward (goal tracking)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config):
        super().__init__()
        from myosuite.utils import gym as myo_gym

        self.cfg = config
        self.base_env = myo_gym.make(config.env_id)

        # HRL state
        self.goal = np.zeros(self.cfg.goal_dim, dtype=np.float32)
        self.t_in_macro = 0

        # Infer obs space
        self.base_env.reset()
        obs_dict = self.base_env.obs_dict

        # First sampled goal
        self.goal = self._sample_goal()
        self.t_in_macro = 0

        worker_obs = self._build_obs(obs_dict)
        logger.info(f"[WorkerEnv] worker_obs_dim = {worker_obs.shape[0]} (expected 428)")

        # Observation space (428,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=worker_obs.shape,
            dtype=np.float32,
        )

        self.action_space = self.base_env.action_space

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _sample_goal(self):
        """Sample goal offset near zero for training."""
        return np.random.normal(
            loc=0.0,
            scale=self.cfg.goal_std,
            size=(self.cfg.goal_dim,),
        ).astype(np.float32)

    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self.base_env.reset(seed=seed)
        obs_dict = self.base_env.obs_dict

        self.goal = self._sample_goal()
        self.t_in_macro = 0

        worker_obs = build_worker_obs(
            obs_dict=obs_dict,
            goal=self.goal,
            t_in_macro=self.t_in_macro,
            cfg=self.cfg,
        )
        return worker_obs, {}

    def step(self, action):
        obs_vec, env_reward, terminated, truncated, info = self.base_env.step(action)
        obs_dict = self.base_env.obs_dict

        # Update phase in macro goal window
        self.t_in_macro += 1

        # Intrinsic goal tracking reward
        r_int = intrinsic_reward(obs_dict, self.goal)

        # Reset goal at macro boundary or episode end
        if self.t_in_macro >= self.cfg.high_level_period or terminated or truncated:
            self.goal = self._sample_goal()
            self.t_in_macro = 0

        worker_obs = self._build_obs(obs_dict)

        info = info or {}
        info["env_reward"] = env_reward
        info["intrinsic_reward"] = r_int

        return worker_obs, r_int, terminated, truncated, info
