from typing import Tuple, Dict, Optional
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal


class TableTennisWorker(CustomEnv):
    def __init__(self, config: Config, training_stage: int = 0):
        super().__init__(config)

        self.goal_low = np.array([-1.2, -0.6, -0.4, -0.8, -0.5, 0.15], np.float32)
        self.goal_high = np.array([0.6, 0.6, 0.6, 0.8, 0.5, 1.0], np.float32)

        self.goal_dim = 6
        self.observation_dim = 18

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        self.training_stage = training_stage
        self.recent_successes = []

        self.success_pos_thr = 0.12
        self.success_vel_thr = 1.2
        self.success_time_thr = 0.35
        self.success_bonus = 15.0

        self.stage_cfg = [
            dict(W_POS=2.5, W_PELV=1.5, W_TIME=0.5, W_VEL=0.1),
            dict(W_POS=2.5, W_PELV=1.5, W_TIME=1.2, W_VEL=0.25),
            dict(W_POS=3.0, W_PELV=2.0, W_TIME=2.0, W_VEL=0.45),
        ]

        self.reset_hrl_state()

    # ---------------------------
    # HRL helpers
    # ---------------------------
    def reset_hrl_state(self):
        self.current_goal = None
        self.goal_start_time = None
        self.goal_start_ball_pos = None
        self._episode_goal_achieved = False

    def set_goal(self, goal6: np.ndarray):
        goal6 = np.asarray(goal6, dtype=np.float32)
        assert goal6.shape == (6,)

        obs = self.env.obs_dict
        self.current_goal = goal6
        self.goal_start_time = float(obs["time"])
        self.goal_start_ball_pos = np.asarray(obs["ball_pos"], np.float32).copy()

    # ---------------------------
    # Gym API
    # ---------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        _, info = super().reset(seed)
        self.reset_hrl_state()
        return self._augment_observation(), info

    def step(self, action: np.ndarray):
        _, base_reward, terminated, truncated, info = super().step(action)

        reward, rinfo = self._compute_reward()
        total_reward = reward + 0.05 * base_reward

        if rinfo["goal_achieved"]:
            self._episode_goal_achieved = True

        if terminated or truncated:
            self._update_curriculum()
            self.reset_hrl_state()

        info.update({
            **rinfo,
            "episode_goal_achieved": self._episode_goal_achieved,
            "is_success": rinfo["goal_achieved"],
        })

        return self._augment_observation(), float(total_reward), terminated, truncated, info

    # ---------------------------
    # Observation / Reward
    # ---------------------------
    def _augment_observation(self) -> np.ndarray:
        obs = self.env.obs_dict

        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        ball = np.asarray(obs["ball_pos"], np.float32)
        paddle = np.asarray(obs["paddle_pos"], np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], np.float32)
        paddle_ori = quat_to_paddle_normal(np.asarray(obs["paddle_ori"], np.float32))
        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], np.float32)
        t = np.array([float(obs["time"])], np.float32)

        state = np.hstack([
            paddle - ball,
            paddle_vel,
            paddle_ori,
            pelvis_xy - ball[:2],
            t,
        ])
        return np.hstack([state, self.current_goal]).astype(np.float32)

    def _sample_goal(self):
        scale = [0.4, 0.7, 1.0][self.training_stage]
        return self.goal_low + scale * (self.goal_high - self.goal_low) * np.random.rand(self.goal_dim)

    def _compute_reward(self):
        obs = self.env.obs_dict
        cfg = self.stage_cfg[self.training_stage]

        paddle = np.asarray(obs["paddle_pos"], np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], np.float32)
        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], np.float32)
        t = float(obs["time"])

        dx, dy, dz, dpx, dpy, dt = self.current_goal
        target_time = self.goal_start_time + dt

        ball0 = self.goal_start_ball_pos
        target_paddle = ball0 + np.array([dx, dy, dz])
        target_pelvis = ball0[:2] + np.array([dpx, dpy])

        pos_err = np.linalg.norm(paddle - target_paddle)
