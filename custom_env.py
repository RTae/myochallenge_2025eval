from typing import Tuple, Dict, Optional
import numpy as np
from myosuite.utils import gym
from config import Config


class CustomEnv(gym.Env):
    """
    Thin wrapper around MyoSuite env:
    - Standardizes reset/step signature
    - Injects `is_success`
    - Forwards everything else
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.env = gym.make(config.env_id)

        # expose spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        info["is_success"] = bool(info.get("solved", False))

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env

    def __getattr__(self, name):
        return getattr(self.env, name)
