from typing import Tuple, Dict, Optional, Any
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal


class TableTennisWorker(CustomEnv):
    """
    Low-level controller (Worker).

    Observation (18):
      - state (12)
      - goal  (6)

    state =
      paddle - ball        (3)
      paddle velocity      (3)
      paddle normal        (3)
      pelvis_xy - ball_xy  (2)
      time                 (1)

    goal =
      [dx, dy, dz, dpx, dpy, dt]
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # -----------------------------
        # Goal bounds (RELATIVE TO BALL)
        # -----------------------------
        self.goal_low = np.array(
            [-0.6, -0.6, -0.4, -0.8, -0.5, 0.15], np.float32
        )
        self.goal_high = np.array(
            [ 0.6,  0.6,  0.6,  0.8,  0.5, 0.8], np.float32
        )

        self.goal_dim = 6
        self.state_dim = 12
        self.obs_dim = 18

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # HRL state
        self.current_goal = None
        self.goal_start_time = None
        self.goal_start_ball = None
        self._prev_paddle_contact = False

        # Success thresholds
        self.pos_thr = 0.12
        self.vel_thr = 1.2
        self.time_thr = 0.35

    # =====================================================
    # HRL API
    # =====================================================
    def set_goal(self, goal: np.ndarray):
        goal = np.asarray(goal, np.float32)
        self.current_goal = goal
        obs = self.env.unwrapped.obs_dict
        self.goal_start_time = float(obs["time"])
        self.goal_start_ball = np.asarray(obs["ball_pos"], np.float32)

    # =====================================================
    # Gym API
    # =====================================================
    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed)

        self._prev_paddle_contact = False

        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        return self._build_obs(), info

    def step(self, action):
        _, base_reward, terminated, truncated, info = super().step(action)

        reward, success = self._compute_reward()
        hit = self._detect_paddle_hit()

        total_reward = reward + 0.05 * base_reward

        info.update({
            "is_success": success,
            "is_paddle_hit": hit,
        })

        if terminated or truncated:
            self.current_goal = None

        return self._build_obs(), total_reward, terminated, truncated, info

    # =====================================================
    # Observation
    # =====================================================
    def _build_obs(self):
        obs = self.env.unwrapped.obs_dict

        ball = np.asarray(obs["ball_pos"], np.float32).reshape(3)
        paddle = np.asarray(obs["paddle_pos"], np.float32).reshape(3)
        paddle_vel = np.asarray(obs["paddle_vel"], np.float32).reshape(3)
        paddle_n = quat_to_paddle_normal(
            np.asarray(obs["paddle_ori"], np.float32)
        ).reshape(3)

        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], np.float32).reshape(2)
        t = np.array([float(obs["time"])], np.float32)

        state = np.concatenate([
            paddle - ball,
            paddle_vel,
            paddle_n,
            pelvis_xy - ball[:2],
            t,
        ])

        return np.concatenate([state, self.current_goal])

    # =====================================================
    # Reward
    # =====================================================
    def _compute_reward(self):
        obs = self.env.unwrapped.obs_dict

        paddle = np.asarray(obs["paddle_pos"], np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], np.float32)
        ball = np.asarray(obs["ball_pos"], np.float32)
        t = float(obs["time"])

        dx, dy, dz, dpx, dpy, dt = self.current_goal
        target_time = self.goal_start_time + dt

        target_paddle = self.goal_start_ball + np.array([dx, dy, dz])
        pos_err = np.linalg.norm(paddle - target_paddle)

        # Explicit Myo reach error (low weight)
        reach_err = np.linalg.norm(paddle - ball)
        vel_norm = np.linalg.norm(paddle_vel)
        time_err = abs(t - target_time)

        reward = (
            2.5 * np.exp(-4.0 * pos_err)
            - 0.2 * reach_err
            - 0.1 * vel_norm
        )

        success = (
            pos_err < self.pos_thr
            and vel_norm < self.vel_thr
            and time_err < self.time_thr
        )

        if success:
            reward += 15.0

        return float(reward), bool(success)

    # =====================================================
    # Paddle hit detection
    # =====================================================
    def _detect_paddle_hit(self):
        touching = np.asarray(
            self.env.unwrapped.obs_dict.get("touching_info", []),
            np.float32,
        )

        ball_paddle = touching[2] if touching.size > 2 else 0.0
        hit = ball_paddle > 0.5 and not self._prev_paddle_contact
        self._prev_paddle_contact = ball_paddle > 0.5
        return bool(hit)

    def _sample_goal(self):
        return self.goal_low + np.random.rand(6) * (self.goal_high - self.goal_low)