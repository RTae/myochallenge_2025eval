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

    Observation = state (12) + goal (6) = 18

    state = [
        reach_err (3),
        ball_vel  (3),
        paddle_normal (3),
        ball_xy (2),
        time (1),
    ]

    goal = [
        x_hit, y_hit, z_hit, n_x, n_y, dt
    ]
    where (x_hit,y_hit,z_hit) is predicted ball intersection at paddle x-plane,
    n_ideal is an "ideal" paddle normal for reflection,
    dt is time-to-plane estimate.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # -------------------------------
        # Goal normalization parameters
        # -------------------------------
        self.goal_center = np.array(
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.475],
            dtype=np.float32,
        )
        self.goal_half_range = np.array(
            [0.6, 0.6, 0.5, 0.8, 0.5, 0.325],
            dtype=np.float32,
        )

        self.goal_dim = 6
        self.state_dim = 12
        self.observation_dim = self.goal_dim + self.state_dim  # 18

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # Runtime state
        self.current_goal: Optional[np.ndarray] = None  # stored in PHYSICAL units
        self.prev_reach_err: Optional[float] = None
        self.goal_start_time: Optional[float] = None
        self._prev_paddle_contact = False

        # Goal success thresholds (goal success != env solved)
        self.reach_thr = 0.25
        self.vel_thr = 1.2
        self.time_thr = 0.35
        self.success_bonus = 40.0

        # Goal sampling noise (you can anneal these later)
        self.pos_noise_scale = np.array([0.08, 0.08, 0.04], dtype=np.float32)
        self.normal_noise_scale = 0.15
        self.time_noise_scale = 0.08

        # Safety bounds for dt
        self.dt_min = 0.05
        self.dt_max = 1.5

    # ------------------------------------------------
    # Goal API
    # ------------------------------------------------
    def set_goal(self, goal_norm: np.ndarray):
        """
        Set a new goal (goal_norm in [-1,1]^6).
        Stores current_goal in PHYSICAL units and resets goal_start_time for timing.
        """
        goal_norm = np.clip(np.asarray(goal_norm, dtype=np.float32).reshape(6,), -1.0, 1.0)
        self.current_goal = self._denorm_goal(goal_norm)

        # Reset timing reference PER GOAL (important for manager-issued goals)
        self.goal_start_time = float(np.asarray(self.env.unwrapped.obs_dict["time"]).reshape(-1)[0])

    def _denorm_goal(self, goal_norm: np.ndarray) -> np.ndarray:
        goal_norm = np.clip(goal_norm, -1.0, 1.0)
        return self.goal_center + goal_norm * self.goal_half_range

    def _norm_goal(self, goal_phys: np.ndarray) -> np.ndarray:
        goal_phys = np.asarray(goal_phys, dtype=np.float32).reshape(6,)
        goal_norm = (goal_phys - self.goal_center) / self.goal_half_range
        return np.clip(goal_norm, -1.0, 1.0)

    def _clip_goal_phys(self, goal_phys: np.ndarray) -> np.ndarray:
        lo = self.goal_center - self.goal_half_range
        hi = self.goal_center + self.goal_half_range
        return np.clip(goal_phys, lo, hi)

    def _sample_goal(self, obs_dict: dict) -> np.ndarray:
        """
        Sample a ball-conditioned goal in NORMALIZED space [-1, 1]^6.

        Uses calculate_prediction(...) to get a plausible interception target,
        then adds noise (for robustness) and clips to valid physical bounds.
        """
        ball_pos = np.asarray(obs_dict["ball_pos"], dtype=np.float32)
        ball_vel = np.asarray(obs_dict["ball_vel"], dtype=np.float32)
        paddle_pos = np.asarray(obs_dict["paddle_pos"], dtype=np.float32)

        pred_pos, n_ideal, dt = calculate_prediction(ball_pos, ball_vel, paddle_pos)

        pred_pos = np.asarray(pred_pos, dtype=np.float32).reshape(3,)
        n_ideal = np.asarray(n_ideal, dtype=np.float32).reshape(3,)
        dt = float(dt)

        # --- noisy curriculum around prediction ---
        pos_noise = np.random.normal(loc=0.0, scale=self.pos_noise_scale).astype(np.float32)
        normal_noise = np.random.normal(loc=0.0, scale=self.normal_noise_scale, size=2).astype(np.float32)
        time_noise = float(np.random.normal(loc=0.0, scale=self.time_noise_scale))

        dt_noisy = float(np.clip(dt + time_noise, self.dt_min, self.dt_max))

        goal_phys = np.array(
            [
                pred_pos[0] + pos_noise[0],
                pred_pos[1] + pos_noise[1],
                pred_pos[2] + pos_noise[2],
                n_ideal[0] + normal_noise[0],
                n_ideal[1] + normal_noise[1],
                dt_noisy,
            ],
            dtype=np.float32,
        )

        # Clamp physical goal to avoid saturation artifacts in normalization
        goal_phys = self._clip_goal_phys(goal_phys)

        return self._norm_goal(goal_phys)

    def predict_goal_from_state(self, obs_dict: dict) -> np.ndarray:
        """
        Utility / oracle: prediction-only goal (no noise), returned in normalized space.
        """
        ball_pos = np.asarray(obs_dict["ball_pos"], dtype=np.float32)
        ball_vel = np.asarray(obs_dict["ball_vel"], dtype=np.float32)
        paddle_pos = np.asarray(obs_dict["paddle_pos"], dtype=np.float32)

        pred_pos, n_ideal, dt = calculate_prediction(ball_pos, ball_vel, paddle_pos)
        pred_pos = np.asarray(pred_pos, dtype=np.float32).reshape(3,)
        n_ideal = np.asarray(n_ideal, dtype=np.float32).reshape(3,)
        dt = float(np.clip(float(dt), self.dt_min, self.dt_max))

        goal_phys = np.array(
            [pred_pos[0], pred_pos[1], pred_pos[2], n_ideal[0], n_ideal[1], dt],
            dtype=np.float32,
        )
        goal_phys = self._clip_goal_phys(goal_phys)
        return self._norm_goal(goal_phys)

    # ------------------------------------------------
    # Gym API
    # ------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed)
        obs_dict = self.env.unwrapped.obs_dict

        self._prev_paddle_contact = False

        # Sample a new internal goal on reset
        self.current_goal = None
        self.set_goal(self._sample_goal(obs_dict))

        # Initialize reach tracking
        self.prev_reach_err = float(np.linalg.norm(np.asarray(obs_dict["reach_err"], dtype=np.float32)))

        return self._build_obs(obs_dict), info

    def step(self, action: np.ndarray):
        obs, base_reward, terminated, truncated, info = super().step(action)
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
    def _build_obs(self, obs_dict: dict) -> np.ndarray:
        reach_err = np.asarray(obs_dict["reach_err"], dtype=np.float32).reshape(3,)
        ball_vel = np.asarray(obs_dict["ball_vel"], dtype=np.float32).reshape(3,)

        paddle_n = quat_to_paddle_normal(
            np.asarray(obs_dict["paddle_ori"], dtype=np.float32).reshape(4,)
        ).reshape(3,)

        ball_xy = np.asarray(obs_dict["ball_pos"][:2], dtype=np.float32).reshape(2,)
        t = np.array([float(obs_dict["time"])], dtype=np.float32).reshape(1,)

        goal_phys = np.asarray(self.current_goal, dtype=np.float32).reshape(6,)

        state = np.hstack([reach_err, ball_vel, paddle_n, ball_xy, t])
        assert state.shape == (self.state_dim,), f"state.shape={state.shape}"

        obs_out = np.hstack([state, goal_phys])
        assert obs_out.shape == (self.observation_dim,), f"obs_out.shape={obs_out.shape}"

        return obs_out

    # ------------------------------------------------
    # Reward + goal success
    # ------------------------------------------------
    def _compute_reward(self, obs_dict: dict, hit: bool) -> Tuple[float, bool, float, float, float]:
        """
        Reward encourages:
        - smaller reach_err (paddle close to target)
        - lower velocity
        - correct timing (match goal dt)
        + sparse bonus for satisfying all constraints
        + small bonus for hit
        """
        reach_err = float(np.linalg.norm(np.asarray(obs_dict["reach_err"], dtype=np.float32)))
        vel_norm = float(np.linalg.norm(np.asarray(obs_dict["paddle_vel"], dtype=np.float32)))
        t_now = float(np.asarray(obs_dict["time"]).reshape(-1)[0])

        time_err = 0.0
        if self.goal_start_time is not None:
            target_time = float(self.goal_start_time) + float(self.current_goal[5])
            time_err = abs(t_now - target_time)
        time_err = min(time_err, 1.0)

        reward = (
            1.2 * np.exp(-2.0 * reach_err)
            + 0.8 * (1.0 - np.clip(reach_err, 0, 2))
            - 0.2 * vel_norm
            - 0.3 * time_err
        )

        reach_delta = 0.0
        if self.prev_reach_err is not None:
            reach_delta = self.prev_reach_err - reach_err
        if reach_delta < 0.0:
            reward += 0.1 * reach_delta  # negative penalty

        reward -= 0.2 * vel_norm * (reach_err > 0.2)

        success = (reach_err < self.reach_thr) and (vel_norm < self.vel_thr) and (time_err < self.time_thr)

        if success:
            reward += float(self.success_bonus)
        if hit:
            reward += 0.3

        return float(reward), bool(success), reach_err, vel_norm, time_err

    # ------------------------------------------------
    # Hit detection
    # ------------------------------------------------
    def _detect_paddle_hit(self, obs_dict: dict) -> bool:
        touching = np.asarray(obs_dict["touching_info"], dtype=np.float32).reshape(-1)
        ball_paddle = float(touching[0])
        paddle_contact = (ball_paddle > 0.5)

        hit = bool(paddle_contact and (not self._prev_paddle_contact))
        self._prev_paddle_contact = bool(paddle_contact)
        return hit