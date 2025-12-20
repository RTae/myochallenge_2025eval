from typing import Tuple, Dict, Optional, Any
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal


class TableTennisWorker(CustomEnv):
    """
    Low-level (worker) controller for HRL.

    Observation:
        state (15) + goal (6) = 21 dims

    state = [
        paddle - ball (3),
        paddle_vel (3),
        ball_vel (3),          # NEW
        paddle_normal (3),
        pelvis_xy - ball_xy (2),
        time (1)
    ]
    """

    def __init__(self, config: Config, training_stage: int = 0):
        super().__init__(config)

        # ------------------------------
        # Goal bounds
        # ------------------------------
        self.goal_low = np.array([-1.2, -0.6, -0.4, -0.8, -0.5, 0.15], np.float32)
        self.goal_high = np.array([0.6,  0.6,  0.6,  0.8,  0.5,  1.0], np.float32)
        self.goal_dim = 6

        # ------------------------------
        # Observation dims
        # ------------------------------
        self.state_dim = 15
        self.observation_dim = self.state_dim + self.goal_dim  # 21

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # ------------------------------
        # HRL state
        # ------------------------------
        self.training_stage = training_stage
        self.current_goal = None
        self.goal_start_time = None
        self.goal_start_ball_pos = None
        self._episode_goal_achieved = False
        self._prev_paddle_contact = False

        # success thresholds
        self.success_pos_thr = 0.12
        self.success_vel_thr = 1.2
        self.success_time_thr = 0.35
        self.success_bonus = 15.0

        self.reset_hrl_state()

    # ============================================================
    # HRL helpers
    # ============================================================
    def reset_hrl_state(self):
        self.current_goal = None
        self.goal_start_time = None
        self.goal_start_ball_pos = None
        self._episode_goal_achieved = False
        self._prev_paddle_contact = False

    def set_goal(self, goal6: np.ndarray):
        goal6 = np.asarray(goal6, np.float32)
        obs = self.env.unwrapped.obs_dict
        self.current_goal = goal6
        self.goal_start_time = float(obs["time"])
        self.goal_start_ball_pos = np.asarray(obs["ball_pos"], np.float32).copy()

    def _sample_goal(self):
        return self.goal_low + (self.goal_high - self.goal_low) * np.random.rand(6)

    # ============================================================
    # Gym API
    # ============================================================
    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.reset_hrl_state()
        if self.current_goal is None:
            self.set_goal(self._sample_goal())
        return self._augment_observation(), {}

    def step(self, action: np.ndarray):
        _, base_reward, terminated, truncated, info = super().step(action)

        shaped_reward, rinfo = self._compute_reward()
        hit = self._detect_paddle_hit()

        reward = shaped_reward + 0.05 * base_reward

        info.update({
            "hit": hit,
            "goal_achieved": rinfo["goal_achieved"],
        })

        if terminated or truncated:
            self.reset_hrl_state()

        return self._augment_observation(), reward, terminated, truncated, info

    # ============================================================
    # Observation
    # ============================================================
    def _augment_observation(self):
        obs = self.env.unwrapped.obs_dict

        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        ball = np.asarray(obs["ball_pos"], np.float32)
        ball_vel = np.asarray(obs["ball_vel"], np.float32)
        paddle = np.asarray(obs["paddle_pos"], np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], np.float32)
        paddle_n = quat_to_paddle_normal(obs["paddle_ori"])
        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], np.float32)
        t = np.array([float(obs["time"])], np.float32)

        state = np.hstack([
            paddle - ball,
            paddle_vel,
            ball_vel,             # NEW
            paddle_n,
            pelvis_xy - ball[:2],
            t,
        ])

        return np.hstack([state, self.current_goal]).astype(np.float32)

    # ============================================================
    # Hit detection
    # ============================================================
    def _detect_paddle_hit(self) -> bool:
        touching = np.asarray(self.env.unwrapped.obs_dict.get("touching_info", []))
        ball_paddle = touching[2] if touching.size > 2 else 0.0
        hit = ball_paddle > 0.5 and not self._prev_paddle_contact
        self._prev_paddle_contact = ball_paddle > 0.5
        return hit

    # ============================================================
    # Reward
    # ============================================================
    def _compute_reward(self):
        obs = self.env.unwrapped.obs_dict
        if self.current_goal is None:
            return 0.0, {"goal_achieved": False}

        paddle = np.asarray(obs["paddle_pos"], np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], np.float32)
        t = float(obs["time"])

        dx, dy, dz, _, _, dt = self.current_goal
        target_time = self.goal_start_time + dt
        target = self.goal_start_ball_pos + np.array([dx, dy, dz])

        pos_err = np.linalg.norm(paddle - target)
        vel_norm = np.linalg.norm(paddle_vel)
        time_err = abs(t - target_time)

        reward = np.exp(-6 * pos_err) - 0.1 * vel_norm
        success = (
            pos_err < self.success_pos_thr
            and vel_norm < self.success_vel_thr
            and time_err < self.success_time_thr
        )

        if success:
            reward += self.success_bonus

        return reward, {"goal_achieved": success}