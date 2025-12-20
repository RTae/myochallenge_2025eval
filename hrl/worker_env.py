# hrl/worker_env.py
from typing import Tuple, Dict, Optional, Any
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal


class TableTennisWorker(CustomEnv):
    """
    Low-level worker (muscle controller).

    Observation = state (15) + goal (6) = 21
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # ------------------------------------------------
        # Goal bounds (RELATIVE offsets, physically sane)
        # ------------------------------------------------
        self.goal_low = np.array(
            [-0.6, -0.6, -0.4, -0.8, -0.5, 0.15],
            dtype=np.float32,
        )
        self.goal_high = np.array(
            [ 0.6,  0.6,  0.6,  0.8,  0.5, 0.8],
            dtype=np.float32,
        )

        self.goal_dim = 6
        self.state_dim = 15
        self.observation_dim = 21

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        self.current_goal: Optional[np.ndarray] = None

        # Success thresholds
        self.reach_thr = 0.12
        self.vel_thr = 1.2
        self.time_thr = 0.35
        self.success_bonus = 15.0

    # ------------------------------------------------
    # Goal API (called by manager)
    # ------------------------------------------------
    def set_goal(self, goal: np.ndarray):
        goal = np.asarray(goal, dtype=np.float32)
        assert goal.shape == (6,)
        self.current_goal = goal

    def _sample_goal(self):
        return np.random.uniform(self.goal_low, self.goal_high)

    # ------------------------------------------------
    # Gym API
    # ------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed)

        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        return self._build_obs(), info

    def step(self, action: np.ndarray):
        obs, base_reward, terminated, truncated, info = super().step(action)

        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        shaped_reward, goal_success = self._compute_reward()

        # Paddle hit signal (rising edge)
        hit = self._detect_paddle_hit()

        total_reward = shaped_reward + 0.05 * float(base_reward)

        info.update({
            "is_goal_success": bool(goal_success),
            "is_paddle_hit": bool(hit),
        })

        return self._build_obs(), total_reward, terminated, truncated, info

    # ------------------------------------------------
    # Observation builder (STRICT SHAPES)
    # ------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        obs = self.env.unwrapped.obs_dict

        reach_err = np.asarray(obs["reach_err"], dtype=np.float32)      # (3,)
        ball_vel  = np.asarray(obs["ball_vel"], dtype=np.float32)       # (3,)
        paddle_n  = quat_to_paddle_normal(
            np.asarray(obs["paddle_ori"], dtype=np.float32)
        )                                                                # (3,)
        ball_xy   = np.asarray(obs["ball_pos"][:2], dtype=np.float32)   # (2,)
        t         = np.asarray([obs["time"]], dtype=np.float32)         # (1,)

        # ---- HARD ASSERTS (catch bugs early) ----
        assert reach_err.shape == (3,)
        assert ball_vel.shape == (3,)
        assert paddle_n.shape == (3,)
        assert ball_xy.shape == (2,)
        assert t.shape == (1,)
        assert self.current_goal.shape == (6,)

        state = np.hstack([
            reach_err,    # 3
            ball_vel,     # 3
            paddle_n,     # 3
            ball_xy,      # 2
            t,            # 1
        ])                 # = 15

        obs_out = np.hstack([state, self.current_goal])  # 21

        assert obs_out.shape == (21,)
        return obs_out

    # ------------------------------------------------
    # Reward + success logic
    # ------------------------------------------------
    def _compute_reward(self) -> Tuple[float, bool]:
        obs = self.env.unwrapped.obs_dict

        reach_err = np.linalg.norm(obs["reach_err"])
        vel_norm  = np.linalg.norm(obs["paddle_vel"])
        time_err  = abs(obs["time"] - self.current_goal[5])

        reward = (
            2.5 * np.exp(-4.0 * reach_err)
            - 0.1 * vel_norm
        )

        success = (
            reach_err < self.reach_thr
            and vel_norm < self.vel_thr
            and time_err < self.time_thr
        )

        if success:
            reward += self.success_bonus

        return float(reward), bool(success)

    # ------------------------------------------------
    # Paddle hit detection
    # ------------------------------------------------
    def _detect_paddle_hit(self) -> bool:
        touching = np.asarray(
            self.env.unwrapped.obs_dict.get("touching_info", []),
            dtype=np.float32,
        )

        if touching.size < 3:
            return False

        # Common MyoSuite convention
        ball_table = touching[0]
        ball_net   = touching[1]
        ball_paddle = touching[2]

        return bool(
            ball_paddle > 0.5 and
            ball_table < 0.1 and
            ball_net < 0.1
        )