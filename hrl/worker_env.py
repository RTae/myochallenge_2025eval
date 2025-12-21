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
        self.state_dim = 12
        self.observation_dim = self.goal_dim + self.state_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        self.current_goal: Optional[np.ndarray] = None
        self.prev_reach_err: Optional[float] = None
        self.goal_start_time: Optional[float] = None
        self._prev_paddle_contact = False

        # Success thresholds for goal success, not episode success
        self.reach_thr = 0.25
        self.vel_thr = 1.2
        self.time_thr = 0.35
        self.success_bonus = 40

    # ------------------------------------------------
    # Goal API
    # ------------------------------------------------
    def set_goal(self, goal: np.ndarray):
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)
        self.current_goal = goal
        self.goal_start_time = float(
            np.asarray(self.env.unwrapped.obs_dict["time"]).reshape(-1)[0]
        )

    def _sample_goal(self) -> np.ndarray:
        return np.random.uniform(self.goal_low, self.goal_high).astype(np.float32)

    # ------------------------------------------------
    # Gym API
    # ------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        _, info = super().reset(seed=seed)
        obs_dict = info['obs_dict']
            
        if self.current_goal is None:
            self.set_goal(self._sample_goal())
        else:
            self.goal_start_time = float(np.asarray(obs_dict["time"]).reshape(-1)[0])

        self._prev_paddle_contact = False
        self.prev_reach_err = float(np.linalg.norm(np.asarray(obs_dict["reach_err"], dtype=np.float32)))
        return self._build_obs(obs_dict), info

    def step(self, action: np.ndarray):
        _, base_reward, terminated, truncated, info = super().step(action)
        
        obs_dict = info['obs_dict']

        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        shaped_reward, goal_success, reach_err, vel_norm, time_err = self._compute_reward(obs_dict)

        hit = self._detect_paddle_hit()
        total_reward = float(shaped_reward + 0.05 * float(base_reward))
        reach_err_delta = 0.0 if self.prev_reach_err is None else (self.prev_reach_err - reach_err)
        self.prev_reach_err = reach_err

        info.update({
            "base_reward": float(base_reward),
            "shaped_reward": float(shaped_reward),
            "reach_err_delta": float(reach_err_delta),
            "reach_err": float(reach_err),
            "paddle_vel_norm": float(vel_norm),
            "goal_time_err": float(time_err),
            "is_goal_success": bool(goal_success),
            "is_paddle_hit": bool(hit),
        })
        return self._build_obs(obs_dict), total_reward, terminated, truncated, info

    # ------------------------------------------------
    # Observation builder
    # ------------------------------------------------
    def _build_obs(self, obs_dict) -> np.ndarray:

        reach_err = np.asarray(obs_dict["reach_err"], dtype=np.float32).reshape(3,)
        ball_vel  = np.asarray(obs_dict["ball_vel"], dtype=np.float32).reshape(3,)
        paddle_n  = quat_to_paddle_normal(
            np.asarray(obs_dict["paddle_ori"], dtype=np.float32).reshape(4,)
        ).reshape(3,)

        ball_xy = np.asarray(obs_dict["ball_pos"][:2], dtype=np.float32).reshape(2,)
        t = np.array([float(obs_dict["time"])], dtype=np.float32).reshape(1,)

        goal = np.asarray(self.current_goal, dtype=np.float32).reshape(6,)

        state = np.hstack([reach_err, ball_vel, paddle_n, ball_xy, t])
        assert state.shape == (self.state_dim,), f"state.shape={state.shape}"

        obs_out = np.hstack([state, goal])
        assert obs_out.shape == (self.observation_dim,), f"obs_out.shape={obs_out.shape}"

        return obs_out

    # ------------------------------------------------
    # Reward + goal success logic
    # ------------------------------------------------
    def _compute_reward(self, obs_dict) -> Tuple[float, bool, float, float, float]:
        
        reach_err = float(np.linalg.norm(np.asarray(obs_dict["reach_err"], dtype=np.float32)))
        vel_norm  = float(np.linalg.norm(np.asarray(obs_dict["paddle_vel"], dtype=np.float32)))
        t_now     = float(np.asarray(obs_dict["time"]).reshape(-1)[0])
        time_err = 0.0

        if getattr(self, "goal_start_time", None) is not None:
            target_time = float(self.goal_start_time) + float(self.current_goal[5])
            time_err = abs(t_now - target_time)

        reward = (
            1.2 * np.exp(-2.0 * reach_err)
            + 0.8 * (1.0 - np.clip(reach_err, 0, 2))
            - 0.05 * vel_norm
            - 0.3 * time_err
        )

        success = (
            reach_err < self.reach_thr
            and vel_norm < self.vel_thr
            and time_err < self.time_thr
        )

        if success:
            reward += self.success_bonus

        return float(reward), bool(success), reach_err, vel_norm, time_err

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