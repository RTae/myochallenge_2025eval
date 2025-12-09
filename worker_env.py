# worker_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config
from hrl_utils import build_worker_obs, intrinsic_reward


class WorkerEnv(gym.Env):
    """
    Low-level goal-conditioned policy.

    Obs:  [base_obs (428), goal (3), phase (1)]  → (432,)
    Act:  same as underlying MyoSuite env (muscle activations)
    Reward: intrinsic (how well paddle matches goal offset)
    """
    metadata = {"render_modes": []}

    def __init__(self, config: Config):
        super().__init__()
        from myosuite.utils import gym as myo_gym

        self.cfg = config
        self.base_env = myo_gym.make(config.env_id)

        # Internal HRL state
        self.goal = np.zeros(self.cfg.goal_dim, dtype=np.float32)
        self.t_in_macro = 0

        # Initialize once to infer obs shape
        self.base_env.reset()
        obs_dict = self.base_env.obs_dict

        # Sample an initial random goal
        self.goal = self._sample_goal(obs_dict)
        self.t_in_macro = 0

        worker_obs = self._build_obs(obs_dict)

        # Observation: worker obs (432,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=worker_obs.shape,
            dtype=np.float32,
        )

        # Action: same as base MyoSuite env
        self.action_space = self.base_env.action_space

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _sample_goal(self, obs_dict):
        """
        Sample a 3D desired (paddle - ball) offset around zero.
        You can make this smarter later.
        """
        return np.random.normal(
            loc=0.0,
            scale=self.cfg.goal_std,
            size=(self.cfg.goal_dim,),
        ).astype(np.float32)

    def _build_obs(self, obs_dict):
        return build_worker_obs(
            obs_dict=obs_dict,
            goal=self.goal,
            t_in_macro=self.t_in_macro,
            cfg=self.cfg,
        )

    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs_vec, info = self.base_env.reset(seed=seed)
        obs_dict = self.base_env.obs_dict

        # New macro, new goal
        self.goal = self._sample_goal(obs_dict)
        self.t_in_macro = 0

        worker_obs = self._build_obs(obs_dict)
        return worker_obs, info

    def step(self, action):
        # Step underlying env
        obs_vec, env_reward, terminated, truncated, info = self.base_env.step(action)
        obs_dict = self.base_env.obs_dict

        # Update macro-step phase
        self.t_in_macro += 1

        # Intrinsic reward based on goal-tracking
        r_int = intrinsic_reward(obs_dict, self.goal)

        # When macro ends or episode ends, start a new macro goal
        if self.t_in_macro >= self.cfg.high_level_period or terminated or truncated:
            self.goal = self._sample_goal(obs_dict)
            self.t_in_macro = 0

        worker_obs = self._build_obs(obs_dict)

        # Log both env reward and intrinsic reward
        info = info or {}
        info["env_reward"] = env_reward
        info["intrinsic_reward"] = r_int

        return worker_obs, r_int, terminated, truncated, info
