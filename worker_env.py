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

        self.cfg = config

        # --------------------------------------------------------
        # Load underlying MyoSuite environment
        # --------------------------------------------------------
        self.base_env = myo_gym.make(config.env_id)

        # Reset once to inspect shapes
        self.base_env.reset()
        obs_dict = self.base_env.obs_dict

        # --------------------------------------------------------
        # Base flatten size
        # --------------------------------------------------------
        base = flatten_myo_obs_worker(obs_dict)
        self.base_dim = base.shape[0]

        # Final worker obs = [base, goal(3), phase(1)]
        self.obs_dim = self.base_dim + config.goal_dim + 1

        # --------------------------------------------------------
        # Gym spaces
        # --------------------------------------------------------
        low = -np.inf * np.ones(self.obs_dim, dtype=np.float32)
        high = np.inf * np.ones(self.obs_dim, dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = self.base_env.action_space

        # Goal + phase tracking
        self.goal = None
        self.t_in_macro = 0

    # ============================================================
    # Helper: sample a random high-level goal
    # ============================================================
    def _sample_goal(self):
        return np.random.normal(
            loc=0.0,
            scale=self.cfg.goal_std,
            size=self.cfg.goal_dim
        ).astype(np.float32)

    # ============================================================
    # Construct worker observation explicitly
    # ============================================================
    def _build_obs(self, obs_dict):
        base = flatten_myo_obs_worker(obs_dict)              # (base_dim,)
        goal = self.goal                                     # (goal_dim,)
        phase = np.array([self.t_in_macro /
                          (self.cfg.high_level_period - 1)],
                          dtype=np.float32)                  # (1,)

        return np.concatenate([base, goal, phase],
                              axis=-1).astype(np.float32)

    # ============================================================
    # RESET
    # ============================================================
    def reset(self, *, seed=None, options=None):
        obs_vec, info = self.base_env.reset(seed=seed)
        obs_dict = self.base_env.obs_dict

        # New goal each episode
        self.goal = self._sample_goal()
        self.t_in_macro = 0

        worker_obs = self._build_obs(obs_dict)
        return worker_obs, info

    # ============================================================
    # STEP
    # ============================================================
    def step(self, action):

        obs_vec, reward_env, terminated, truncated, info = self.base_env.step(action)
        obs_dict = self.base_env.obs_dict

        # Update macro-step time
        self.t_in_macro += 1

        # If new macro-step → resample goal
        if self.t_in_macro >= self.cfg.high_level_period:
            self.goal = self._sample_goal()
            self.t_in_macro = 0

        worker_obs = self._build_obs(obs_dict)

        return worker_obs, reward_env, terminated, truncated, info
