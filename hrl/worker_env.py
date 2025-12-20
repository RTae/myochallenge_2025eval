# hrl/worker_env.py
from typing import Tuple, Dict, Optional
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal


class TableTennisWorker(CustomEnv):
    """
    Low-level worker (muscle controller).

    Observation = state (15) + goal (6) = 21

    state = [
        reach_err (3),
        ball_vel  (3),
        paddle_normal (3),
        ball_xy (2),
        time (1),
    ]
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # ------------------------------------------------
        # Goal bounds (RELATIVE offsets, sane)
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
        self._prev_paddle_contact = False  # for rising-edge hit

        # Success thresholds for GOAL success (not episode success)
        self.reach_thr = 0.12
        self.vel_thr = 1.2
        self.time_thr = 0.35
        self.success_bonus = 15.0

    # ------------------------------------------------
    # Goal API (called by manager)
    # ------------------------------------------------
    def set_goal(self, goal: np.ndarray):
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)
        assert goal.shape == (6,), f"goal.shape={goal.shape}"
        self.current_goal = goal

    def _sample_goal(self) -> np.ndarray:
        return np.random.uniform(self.goal_low, self.goal_high).astype(np.float32)

    # ------------------------------------------------
    # Gym API
    # ------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        _, info = super().reset(seed=seed)

        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        self._prev_paddle_contact = False
        return self._build_obs(), info

    def step(self, action: np.ndarray):
        _, base_reward, terminated, truncated, info = super().step(action)

        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        shaped_reward, goal_success = self._compute_reward()

        # rising-edge paddle hit
        hit = self._detect_paddle_hit()

        total_reward = float(shaped_reward + 0.05 * float(base_reward))

        # IMPORTANT:
        # - keep info["is_success"] from CustomEnv (solved) as-is
        # - add is_goal_success separately
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

        reach_err = np.asarray(obs["reach_err"], dtype=np.float32).reshape(3,)    # (3,)
        ball_vel  = np.asarray(obs["ball_vel"], dtype=np.float32).reshape(3,)     # (3,)
        paddle_n  = quat_to_paddle_normal(
            np.asarray(obs["paddle_ori"], dtype=np.float32).reshape(4,)
        ).reshape(3,)                                                             # (3,)

        ball_pos = np.asarray(obs["ball_pos"], dtype=np.float32).reshape(3,)
        ball_xy  = ball_pos[:2].reshape(2,)                                       # (2,)

        # FIX: obs["time"] might be array(0.01) or array([0.01]) -> always make (1,)
        t_scalar = float(np.asarray(obs["time"]).reshape(-1)[0])
        t = np.array([t_scalar], dtype=np.float32).reshape(1,)                    # (1,)

        assert self.current_goal is not None
        goal = np.asarray(self.current_goal, dtype=np.float32).reshape(6,)        # (6,)

        state = np.hstack([reach_err, ball_vel, paddle_n, ball_xy, t]).astype(np.float32)
        assert state.shape == (15,), f"state.shape={state.shape}"

        obs_out = np.hstack([state, goal]).astype(np.float32)
        assert obs_out.shape == (21,), f"obs_out.shape={obs_out.shape}"
        return obs_out

    # ------------------------------------------------
    # Reward + goal success logic (based on reach_err)
    # ------------------------------------------------
    def _compute_reward(self) -> Tuple[float, bool]:
        obs = self.env.unwrapped.obs_dict

        reach_err = float(np.linalg.norm(np.asarray(obs["reach_err"], dtype=np.float32)))
        vel_norm  = float(np.linalg.norm(np.asarray(obs["paddle_vel"], dtype=np.float32)))

        t_scalar = float(np.asarray(obs["time"]).reshape(-1)[0])
        dt_goal = float(np.asarray(self.current_goal, dtype=np.float32).reshape(-1)[5])
        time_err = abs(t_scalar - dt_goal)

        # shaping
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
    # Paddle hit detection (RISING EDGE)
    # ------------------------------------------------
    def _detect_paddle_hit(self) -> bool:
        touching = np.asarray(
            self.env.unwrapped.obs_dict.get("touching_info", []),
            dtype=np.float32,
        ).reshape(-1)

        if touching.size < 3:
            return False

        ball_table  = float(touching[0])
        ball_net    = float(touching[1])
        ball_paddle = float(touching[2])

        paddle_contact = (ball_paddle > 0.5) and (ball_table < 0.1) and (ball_net < 0.1)

        hit = bool(paddle_contact and (not self._prev_paddle_contact))
        self._prev_paddle_contact = bool(paddle_contact)
        return hit