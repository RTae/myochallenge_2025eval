from typing import Tuple, Dict, Any
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal


class TableTennisWorker(CustomEnv):
    """
    Low-level controller.
    Learns HOW to reach a goal, not WHICH goal to choose.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # ===============================
        # Goal space (manager controls)
        # ===============================
        self.goal_low = np.array(
            [-0.20, -0.20, -0.15, -0.60, -0.40, 0.20],
            dtype=np.float32
        )
        self.goal_high = np.array(
            [ 0.20,  0.20,  0.25,  0.60,  0.40, 0.80],
            dtype=np.float32
        )
        self.goal_dim = 6

        # ===============================
        # Observation: 12 state + 6 goal
        # ===============================
        self.state_dim = 12
        self.observation_dim = self.state_dim + self.goal_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # ===============================
        # HRL internal state
        # ===============================
        self.current_goal = None
        self.goal_start_time = None
        self.goal_start_ball_pos = None
        self._prev_paddle_contact = False

        # Success thresholds
        self.success_pos_thr = 0.10
        self.success_vel_thr = 1.2
        self.success_time_thr = 0.25
        self.success_bonus = 15.0

    # ------------------------------------------------
    # HRL helpers
    # ------------------------------------------------
    def set_goal(self, goal: np.ndarray):
        goal = np.asarray(goal, np.float32)
        assert goal.shape == (6,)

        obs = self.env.unwrapped.obs_dict
        self.current_goal = goal
        self.goal_start_time = float(obs["time"])
        self.goal_start_ball_pos = np.asarray(obs["ball_pos"], np.float32).copy()

    def _sample_goal(self):
        return (
            self.goal_low
            + (self.goal_high - self.goal_low) * np.random.rand(6)
        ).astype(np.float32)

    # ------------------------------------------------
    # Gym API
    # ------------------------------------------------
    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed)

        self.current_goal = None
        self._prev_paddle_contact = False

        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        return self._build_obs(), info

    def step(self, action):
        _, base_reward, terminated, truncated, info = super().step(action)

        shaped_reward, rinfo = self._compute_reward()
        hit = self._detect_paddle_hit()

        total_reward = shaped_reward + 0.05 * base_reward

        info.update({
            "hit": bool(hit),
            "goal_achieved": bool(rinfo["goal_achieved"]),
        })

        if terminated or truncated:
            self.current_goal = None

        return self._build_obs(), float(total_reward), terminated, truncated, info

    # ------------------------------------------------
    # Observation
    # ------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        obs = self.env.unwrapped.obs_dict

        ball = np.asarray(obs["ball_pos"], np.float32)
        paddle = np.asarray(obs["paddle_pos"], np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], np.float32)
        paddle_n = quat_to_paddle_normal(obs["paddle_ori"])
        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], np.float32)
        t = np.array([obs["time"]], np.float32)

        state = np.hstack([
            paddle - ball,         # reach_err (3)
            paddle_vel,            # (3)
            paddle_n,              # (3)
            pelvis_xy - ball[:2],  # (2)
            t,                     # (1)
        ])

        return np.hstack([state, self.current_goal]).astype(np.float32)

    # ------------------------------------------------
    # Hit detection
    # ------------------------------------------------
    def _detect_paddle_hit(self) -> bool:
        touching = np.asarray(
            self.env.unwrapped.obs_dict.get("touching_info", []),
            np.float32
        )

        ball_paddle = touching[2] if touching.size > 2 else 0.0
        hit = (ball_paddle > 0.5) and not self._prev_paddle_contact
        self._prev_paddle_contact = ball_paddle > 0.5
        return hit

    # ------------------------------------------------
    # Reward
    # ------------------------------------------------
    def _compute_reward(self) -> Tuple[float, Dict[str, Any]]:
        obs = self.env.unwrapped.obs_dict

        paddle = np.asarray(obs["paddle_pos"], np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], np.float32)
        t = float(obs["time"])

        dx, dy, dz, dpx, dpy, dt = self.current_goal
        target_time = self.goal_start_time + dt

        target_paddle = self.goal_start_ball_pos + np.array([dx, dy, dz])
        pos_err = np.linalg.norm(paddle - target_paddle)
        vel_norm = np.linalg.norm(paddle_vel)
        time_err = abs(t - target_time)

        reward = np.exp(-6.0 * pos_err) - 0.05 * vel_norm

        goal_achieved = (
            pos_err < self.success_pos_thr
            and vel_norm < self.success_vel_thr
            and time_err < self.success_time_thr
        )

        if goal_achieved:
            reward += self.success_bonus

        return reward, {"goal_achieved": goal_achieved}