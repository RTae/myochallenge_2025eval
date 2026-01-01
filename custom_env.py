from typing import Tuple, Dict, Optional
import numpy as np
from myosuite.utils import gym
from config import Config


class CustomEnv(gym.Env):

    def __init__(self, config: Config, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        self.env = gym.make(
            config.env_id,
            max_episode_steps=config.episode_len,
            render_mode=render_mode,
        )

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        act_reg = info['rwd_dict']['act_reg']
        effort = -1.0 * np.mean(act_reg)
        info["is_success"] = bool(info.get("solved", False))
        info['effort'] = effort
        
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)
