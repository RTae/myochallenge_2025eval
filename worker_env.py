# worker_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config
from hrl_utils import build_worker_obs, HitDetector, worker_reward
from loguru import logger


class WorkerEnv(gym.Env):
    """
    Low-level worker (contact-conditioned).

    Obs    = [base (424), paddle_vel (3), goal_vel (3), phase (1)]
    Action = muscle activations
    Reward = contact-conditioned (hit-driven)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config):
        super().__init__()
        from myosuite.utils import gym as myo_gym

        self.cfg = config
        self.base_env = myo_gym.make(config.env_id)

        # --- hit detector ---
        self.hit_detector = HitDetector(force_thr=1e-3, dv_thr=0.8)

        self.goal = np.zeros(3, dtype=np.float32)
        self.t_in_macro = 0
        self.hit_count = 0
        self.step_count = 0

        # Infer obs shape
        self.base_env.reset()
        obs_dict = self.base_env.obs_dict
        self.goal = self._sample_goal()

        worker_obs = build_worker_obs(
            obs_dict, self.goal, self.t_in_macro, self.cfg
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=worker_obs.shape,
            dtype=np.float32,
        )

        self.action_space = self.base_env.action_space

    # --------------------------------------------------
    def _sample_goal(self):
        """
        Desired paddle velocity (small magnitude).
        """
        return np.random.normal(
            loc=0.0,
            scale=self.cfg.goal_std,  # e.g. 0.2–0.4
            size=(3,),
        ).astype(np.float32)

    # --------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self.base_env.reset(seed=seed)
        obs_dict = self.base_env.obs_dict

        self.hit_detector.reset(obs_dict)
        self.goal = self._sample_goal()
        self.t_in_macro = 0

        return build_worker_obs(
            obs_dict, self.goal, self.t_in_macro, self.cfg
        ), {}

    # --------------------------------------------------
    def step(self, action):
        _, _, terminated, truncated, info = self.base_env.step(action)
        obs_dict = self.base_env.obs_dict

        self.t_in_macro += 1

        hit, contact_force, dv = self.hit_detector.step(obs_dict)
        
        self.hit_count += int(hit)
        self.step_count += 1
        
        # ---- INTRINSIC REWARD ----
        r_int = worker_reward(
            obs_dict=obs_dict,
            hit=hit,
            contact_force=contact_force,
            dv=dv,
        )

        # ---- GOAL RESET ----
        if self.t_in_macro >= self.cfg.high_level_period or terminated or truncated:
            self.goal = self._sample_goal()
            self.t_in_macro = 0

        obs = build_worker_obs(
            obs_dict, self.goal, self.t_in_macro, self.cfg
        )

        info = info or {}
        info["hit"] = hit
        info["hit_rate"] = self.hit_count / max(1, self.step_count)
        info["contact_force"] = contact_force
        info["dv"] = dv
        info["intrinsic_reward"] = r_int

        return obs, r_int, terminated, truncated, info
