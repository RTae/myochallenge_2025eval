from typing import Tuple, Dict, Optional
import numpy as np

from myosuite.utils import gym as myo_gym
from config import Config


class CustomEnv(myo_gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, config: Config, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device
        self.env = myo_gym.make(config.env_id)
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        info.update({
            "is_success": bool(info.get("solved", False)),
        })

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
