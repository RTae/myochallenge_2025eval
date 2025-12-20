from typing import Tuple, Dict, Optional
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal


class TableTennisWorker(CustomEnv):
    """
    Low-level controller.
    Observation: 18 (state) + 6 (goal) = 24
    Action: original MyoSuite action space
    """

    def __init__(self, config: Config, training_stage: int = 0):
        super().__init__(config)

        # Goal bounds
        self.goal_low = np.array([-1.2, -0.6, -0.4, -0.8, -0.5, 0.15], np.float32)
        self.goal_high = np.array([0.6, 0.6, 0.6, 0.8, 0.5, 1.0], np.float32)

        self.goal_dim = 6
        self.observation_dim = 18

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )

        # HRL state
        self.current_goal = None
        self.goal_start_time = None
        self.goal_start_ball_pos = None

        self._prev_paddle_contact = False

    # -----------------------------
    # HRL helpers
    # -----------------------------
    def reset_hrl_state(self):
        self.current_goal = None
        self.goal_start_time = None
        self.goal_start_ball_pos = None
        self._prev_paddle_contact = False

    def set_goal(self, goal6: np.ndarray):
        goal6 = np.asarray(goal6, np.float32)
        assert goal6.shape == (6,)
        obs = self.env.obs_dict
        self.current_goal = goal6
        self.goal_start_time = float(obs["time"])
        self.goal_start_ball_pos = np.asarray(obs["ball_pos"], np.float32).copy()

    # -----------------------------
    # Gym API
    # -----------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        _, info = super().reset(seed)
        self.reset_hrl_state()
        return self._augment_observation(), info

    def step(self, action: np.ndarray):
        _, base_reward, terminated, truncated, info = super().step(action)

        reward, rinfo = self._compute_reward()

        touching = np.asarray(
            self.env.unwrapped.obs_dict.get("touching_info", []),
            dtype=np.float32,
        )

        # MyoSuite convention:
        ball_table  = touching[0] if touching.size > 0 else 0.0
        ball_net    = touching[1] if touching.size > 1 else 0.0
        ball_paddle = touching[2] if touching.size > 2 else 0.0

        paddle_contact = (
            ball_paddle > 0.5 and
            ball_table  < 0.1 and
            ball_net    < 0.1
        )

        # Rising edge = actual hit
        hit = paddle_contact and not self._prev_paddle_contact
        self._prev_paddle_contact = paddle_contact

        info.update({
            # --- HRL signals ---
            "hit": bool(hit),
            "paddle_contact": bool(paddle_contact),
            "goal_achieved": bool(rinfo.get("goal_achieved", False)),
        })

        if terminated or truncated:
            self.reset_hrl_state()

        return (
            self._augment_observation(),
            float(reward + 0.05 * base_reward),
            terminated,
            truncated,
            info,
        )


    # -----------------------------
    # Observation & Reward
    # -----------------------------
    def _augment_observation(self) -> np.ndarray:
        obs = self.env.unwrapped.obs_dict

        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        ball = np.asarray(obs["ball_pos"], np.float32)
        paddle = np.asarray(obs["paddle_pos"], np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], np.float32)
        paddle_ori = quat_to_paddle_normal(obs["paddle_ori"])
        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], np.float32)
        t = np.array([float(obs["time"])], np.float32)

        state = np.hstack([
            paddle - ball,           # 3
            paddle_vel,              # 3
            paddle_ori,              # 3
            pelvis_xy - ball[:2],    # 2
            t,                       # 1
        ])  # = 12

        return np.hstack([state, self.current_goal]).astype(np.float32)  # 24

    def _sample_goal(self):
        return self.goal_low + (self.goal_high - self.goal_low) * np.random.rand(6)

    def _compute_reward(self):
        return 0.0, {"goal_achieved": False}
