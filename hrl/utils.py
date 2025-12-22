from __future__ import annotations

from typing import Tuple, Dict, Optional
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal

from hrl.utils import calculate_prediction


class TableTennisWorker(CustomEnv):
    """
    Low-level worker (muscle controller).

    Obs = state (12) + goal_phys (6) = 18
    state = [
        reach_err (3),
        ball_vel (3),
        paddle_normal (3),
        ball_xy (2),
        time (1),
    ]

    Goal (physical) = [
        target_x,
        target_y,
        target_z,
        normal_x,
        normal_y,
        dt  (time offset)
    ]

    NOTE:
    - set_goal() accepts goal in NORMALIZED space [-1,1]^6.
    - internally we store goal in PHYSICAL units in self.current_goal.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # ------------------------------------------------
        # Goal normalization (phys <-> norm)
        # ------------------------------------------------
        self.goal_center = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.475], dtype=np.float32)
        self.goal_half_range = np.array([0.6, 0.6, 0.5, 0.8, 0.5, 0.325], dtype=np.float32)

        self.goal_dim = 6
        self.state_dim = 12
        self.observation_dim = self.goal_dim + self.state_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        self.current_goal: Optional[np.ndarray] = None  # PHYSICAL 6D goal
        self.prev_reach_err: Optional[float] = None
        self.goal_start_time: Optional[float] = None
        self._prev_paddle_contact = False

        # Goal success thresholds (worker-level)
        self.reach_thr = 0.25
        self.vel_thr = 1.2
        self.time_thr = 0.35
        self.success_bonus = 40.0

    # ------------------------------------------------
    # Goal helpers
    # ------------------------------------------------
    def _denorm_goal(self, goal_norm: np.ndarray) -> np.ndarray:
        goal_norm = np.clip(goal_norm, -1.0, 1.0).astype(np.float32)
        return self.goal_center + goal_norm * self.goal_half_range

    def _norm_goal(self, goal_phys: np.ndarray) -> np.ndarray:
        goal_phys = np.asarray(goal_phys, dtype=np.float32).reshape(6,)
        goal_norm = (goal_phys - self.goal_center) / self.goal_half_range
        return np.clip(goal_norm, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------
    # External API: manager calls this
    # ------------------------------------------------
    def set_goal(self, goal: np.ndarray):
        """
        goal: normalized [-1,1]^6
        """
        goal_norm = np.asarray(goal, dtype=np.float32).reshape(6,)
        goal_norm = np.clip(goal_norm, -1.0, 1.0)

        # Store PHYSICAL goal
        self.current_goal = self._denorm_goal(goal_norm)

        # Reset timing reference PER GOAL (critical for manager)
        t_now = float(np.asarray(self.env.unwrapped.obs_dict["time"]).reshape(-1)[0])
        self.goal_start_time = t_now

    # ------------------------------------------------
    # (Optional) Predict a good goal from current state
    # ------------------------------------------------
    def predict_goal_from_state(self, obs_dict: dict) -> np.ndarray:
        """
        Returns: normalized goal [-1,1]^6 constructed from physics prediction.

        Requires obs_dict to contain:
        - "ball_pos" (3,)
        - "ball_vel" (3,)
        - "paddle_pos" (3,)  (or you can derive from your env if named differently)
        """
        ball_pos = np.asarray(obs_dict["ball_pos"], dtype=np.float32).reshape(3,)
        ball_vel = np.asarray(obs_dict["ball_vel"], dtype=np.float32).reshape(3,)
        paddle_pos = np.asarray(obs_dict["paddle_pos"], dtype=np.float32).reshape(3,)

        # calculate_prediction returns (pred_ball_pos, n_ideal, paddle_ori_ideal)
        pred_ball_pos, n_ideal, _ = calculate_prediction(ball_pos, ball_vel, paddle_pos)

        pred_ball_pos = np.asarray(pred_ball_pos, dtype=np.float32).reshape(3,)
        n_ideal = np.asarray(n_ideal, dtype=np.float32).reshape(3,)

        # Use predicted time-to-plane as dt if your function returns it.
        # If not returned, we estimate dt using vx.
        # NOTE: this is just a fallback; better is to return dt from calculate_prediction.
        vx = float(ball_vel[0])
        err_x = float(paddle_pos[0] - ball_pos[0])
        if err_x > 0.0 and vx > 1e-3:
            dt = float(np.clip(err_x / vx, 0.0, 2.0))
        else:
            dt = 0.35  # safe default (inside your goal range)

        goal_phys = np.array(
            [pred_ball_pos[0], pred_ball_pos[1], pred_ball_pos[2], n_ideal[0], n_ideal[1], dt],
            dtype=np.float32,
        )

        return self._norm_goal(goal_phys)

    # ------------------------------------------------
    # Random/ball-conditioned sampling (for worker pretrain)
    # ------------------------------------------------
    def _sample_goal(self, obs_dict: dict) -> np.ndarray:
        """
        Sample a ball-conditioned goal in NORMALIZED space [-1,1]^6.
        """
        ball_xy = np.asarray(obs_dict["ball_pos"][:2], dtype=np.float32)

        noise_xy = np.random.normal(scale=0.15, size=2).astype(np.float32)
        target_xy = ball_xy + noise_xy

        dz = np.random.uniform(0.0, 0.3)
        nxny = np.random.uniform(-0.5, 0.5, size=2)
        dt = np.random.uniform(0.2, 0.5)

        goal_phys = np.array([target_xy[0], target_xy[1], dz, nxny[0], nxny[1], dt], dtype=np.float32)
        return self._norm_goal(goal_phys)

    # ------------------------------------------------
    # Gym API
    # ------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        _, info = super().reset(seed=seed)
        obs_dict = self.env.unwrapped.obs_dict

        # New episode -> reset internal trackers
        self._prev_paddle_contact = False
        self.prev_reach_err = float(np.linalg.norm(np.asarray(obs_dict["reach_err"], dtype=np.float32)))

        # Set an initial goal (for standalone worker training)
        goal_norm = self._sample_goal(obs_dict)
        self.set_goal(goal_norm)

        return self._build_obs(obs_dict), info

    def step(self, action: np.ndarray):
        _, base_reward, terminated, truncated, info = super().step(action)
        obs_dict = info["obs_dict"]

        assert self.current_goal is not None, "Worker goal missing during step()"

        hit = self._detect_paddle_hit(obs_dict)
        shaped_reward, goal_success, reach_err, vel_norm, time_err = self._compute_reward(obs_dict, hit)

        total_reward = float(shaped_reward + 0.05 * float(base_reward))

        reach_err_delta = 0.0 if self.prev_reach_err is None else (self.prev_reach_err - reach_err)
        self.prev_reach_err = reach_err

        info.update(
            {
                "base_reward": float(base_reward),
                "shaped_reward": float(shaped_reward),
                "reach_err_delta": float(reach_err_delta),
                "reach_err": float(reach_err),
                "paddle_vel_norm": float(vel_norm),
                "goal_time_err": float(time_err),
                "is_goal_success": bool(goal_success),
                "is_paddle_hit": bool(hit),
            }
        )

        return self._build_obs(obs_dict), total_reward, terminated, truncated, info

    # ------------------------------------------------
    # Observation builder
    # ------------------------------------------------
    def _build_obs(self, obs_dict) -> np.ndarray:
        reach_err = np.asarray(obs_dict["reach_err"], dtype=np.float32).reshape(3,)
        ball_vel = np.asarray(obs_dict["ball_vel"], dtype=np.float32).reshape(3,)

        paddle_n = quat_to_paddle_normal(
            np.asarray(obs_dict["paddle_ori"], dtype=np.float32).reshape(4,)
        ).reshape(3,)

        ball_xy = np.asarray(obs_dict["ball_pos"][:2], dtype=np.float32).reshape(2,)
        t = np.array([float(obs_dict["time"])], dtype=np.float32).reshape(1,)

        goal = np.asarray(self.current_goal, dtype=np.float32).reshape(6,)  # PHYSICAL

        state = np.hstack([reach_err, ball_vel, paddle_n, ball_xy, t])
        obs_out = np.hstack([state, goal])

        assert state.shape == (self.state_dim,), f"state.shape={state.shape}"
        assert obs_out.shape == (self.observation_dim,), f"obs_out.shape={obs_out.shape}"

        return obs_out

    # ------------------------------------------------
    # Reward + goal success
    # ------------------------------------------------
    def _compute_reward(self, obs_dict: dict, hit: bool) -> Tuple[float, bool, float, float, float]:
        reach_err = float(np.linalg.norm(np.asarray(obs_dict["reach_err"], dtype=np.float32)))
        vel_norm = float(np.linalg.norm(np.asarray(obs_dict["paddle_vel"], dtype=np.float32)))
        t_now = float(np.asarray(obs_dict["time"]).reshape(-1)[0])

        # Timing error relative to when this goal was issued
        time_err = 0.0
        if self.goal_start_time is not None:
            target_time = float(self.goal_start_time) + float(self.current_goal[5])
            time_err = abs(t_now - target_time)
        time_err = min(time_err, 1.0)

        # Dense shaping
        reward = (
            1.2 * np.exp(-2.0 * reach_err)
            + 0.8 * (1.0 - np.clip(reach_err, 0, 2))
            - 0.2 * vel_norm
            - 0.3 * time_err
        )

        # Penalize moving away from target
        if self.prev_reach_err is not None:
            reach_delta = self.prev_reach_err - reach_err
            if reach_delta < 0.0:
                reward += 0.1 * reach_delta  # negative penalty

        # Extra velocity penalty when still far
        reward -= 0.2 * vel_norm * (reach_err > 0.2)

        success = (reach_err < self.reach_thr) and (vel_norm < self.vel_thr) and (time_err < self.time_thr)

        if success:
            reward += self.success_bonus

        if hit:
            reward += 0.3

        return float(reward), bool(success), reach_err, vel_norm, time_err

    # ------------------------------------------------
    # Hit detection
    # ------------------------------------------------
    def _detect_paddle_hit(self, obs_dict) -> bool:
        touching = np.asarray(obs_dict.get("touching_info", []), dtype=np.float32).reshape(-1)
        if touching.size < 1:
            return False

        # touching[0] = ball-paddle contact in your env comment
        ball_paddle = float(touching[0])
        paddle_contact = (ball_paddle > 0.5)

        hit = bool(paddle_contact and (not self._prev_paddle_contact))
        self._prev_paddle_contact = bool(paddle_contact)
        return hit