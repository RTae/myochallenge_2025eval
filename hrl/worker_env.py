from typing import Tuple, Dict, Optional, Any
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal


class TableTennisWorker(CustomEnv):
    """
    Low-level (worker) controller for HRL.

    - Action space: inherited from CustomEnv/Myosuite (muscle/control action).
    - Goal space (set by manager): 6D goal = [dx, dy, dz, dpx, dpy, dt]
    - Observation: compact goal-conditioned features:
        state (12) + goal (6) = 18 dims

      state = [
        paddle - ball (3),
        paddle_vel (3),
        paddle_normal (3),
        pelvis_xy - ball_xy (2),
        time (1)
      ]
    """

    def __init__(self, config: Config, training_stage: int = 0):
        super().__init__(config)

        # -----------------------------------------
        # Goal bounds (manager outputs within these)
        # -----------------------------------------
        self.goal_low = np.array([-1.2, -0.6, -0.4, -0.8, -0.5, 0.15], dtype=np.float32)
        self.goal_high = np.array([0.6,  0.6,  0.6,  0.8,  0.5,  1.0], dtype=np.float32)
        self.goal_dim = 6

        # -----------------------------------------
        # Observation dims: 12 state + 6 goal = 18
        # -----------------------------------------
        self.state_dim = 12
        self.observation_dim = self.state_dim + self.goal_dim  # 18

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # -----------------------------------------
        # HRL state
        # -----------------------------------------
        self.training_stage = training_stage

        self.current_goal: Optional[np.ndarray] = None
        self.goal_start_time: Optional[float] = None
        self.goal_start_ball_pos: Optional[np.ndarray] = None
        self._episode_goal_achieved: bool = False

        # For hit detection (rising edge)
        self._prev_paddle_contact: bool = False

        # Success thresholds (goal achieved)
        self.success_pos_thr = 0.12
        self.success_vel_thr = 1.2
        self.success_time_thr = 0.35
        self.success_bonus = 15.0

        # Reward weights by stage (optional)
        self.stage_cfg = [
            dict(W_POS=2.5, W_PELV=1.5, W_TIME=0.5, W_VEL=0.1),
            dict(W_POS=2.5, W_PELV=1.5, W_TIME=1.2, W_VEL=0.25),
            dict(W_POS=3.0, W_PELV=2.0, W_TIME=2.0, W_VEL=0.45),
        ]

        self.reset_hrl_state()

    # ============================================================
    # HRL helpers
    # ============================================================
    def reset_hrl_state(self) -> None:
        self.current_goal = None
        self.goal_start_time = None
        self.goal_start_ball_pos = None
        self._episode_goal_achieved = False
        self._prev_paddle_contact = False

    def set_goal(self, goal6: np.ndarray) -> None:
        goal6 = np.asarray(goal6, dtype=np.float32).reshape(-1)
        assert goal6.shape == (6,), f"Expected goal shape (6,), got {goal6.shape}"

        obs = self.env.unwrapped.obs_dict
        self.current_goal = goal6
        self.goal_start_time = float(obs["time"])
        self.goal_start_ball_pos = np.asarray(obs["ball_pos"], dtype=np.float32).copy()

    def _sample_goal(self) -> np.ndarray:
        # If training worker standalone, it needs some goal distribution
        scale = [0.4, 0.7, 1.0][min(max(self.training_stage, 0), 2)]
        return (
            self.goal_low
            + scale * (self.goal_high - self.goal_low) * np.random.rand(self.goal_dim)
        ).astype(np.float32)

    # ============================================================
    # Gym API
    # ============================================================
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = super().reset(seed=seed)

        # keep HRL internals consistent
        self.reset_hrl_state()

        # If standalone, we need a goal before building obs
        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        return self._augment_observation(), info

    def step(self, action: np.ndarray):
        # Step the base env (muscle control)
        obs, base_reward, terminated, truncated, info = super().step(action)

        # Compute goal-conditioned reward + signals
        shaped_reward, rinfo = self._compute_reward()

        # ---- Detect paddle hit (rising edge) from touching_info ----
        hit = self._detect_paddle_hit()

        # Total reward: shaped + small base
        total_reward = float(shaped_reward + 0.05 * float(base_reward))

        # Track whether goal was achieved at least once this episode
        if rinfo.get("goal_achieved", False):
            self._episode_goal_achieved = True

        # Put HRL signals into info
        info.update({
            "hit": bool(hit),  # paddle hit rising-edge
            "goal_achieved": bool(rinfo.get("goal_achieved", False)),
            "episode_goal_achieved": bool(self._episode_goal_achieved),
        })

        # On episode end, reset HRL state (next episode)
        if terminated or truncated:
            self.reset_hrl_state()

        return self._augment_observation(), total_reward, terminated, truncated, info

    # ============================================================
    # Observation
    # ============================================================
    def _augment_observation(self) -> np.ndarray:
        obs = self.env.unwrapped.obs_dict

        # Ensure goal exists
        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        ball = np.asarray(obs["ball_pos"], dtype=np.float32)
        paddle = np.asarray(obs["paddle_pos"], dtype=np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], dtype=np.float32)
        paddle_n = quat_to_paddle_normal(np.asarray(obs["paddle_ori"], dtype=np.float32))
        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], dtype=np.float32)
        t = np.array([float(obs["time"])], dtype=np.float32)

        state = np.hstack([
            paddle - ball,           # 3
            paddle_vel,              # 3
            paddle_n,                # 3
            pelvis_xy - ball[:2],    # 2
            t,                       # 1
        ]).astype(np.float32)         # 12

        out = np.hstack([state, self.current_goal]).astype(np.float32)  # 18

        # Hard assert to catch mismatches early
        assert out.shape == (18,), f"Worker obs shape mismatch: {out.shape}"
        return out

    # ============================================================
    # Hit detection
    # ============================================================
    def _detect_paddle_hit(self) -> bool:
        touching = np.asarray(
            self.env.unwrapped.obs_dict.get("touching_info", []),
            dtype=np.float32,
        )

        # You MUST verify these indices match your env.
        # Common MyoSuite convention (often):
        #   touching_info[2] = ball-paddle contact
        #   touching_info[0] = ball-table contact
        #   touching_info[1] = ball-net contact
        ball_table = touching[0] if touching.size > 0 else 0.0
        ball_net = touching[1] if touching.size > 1 else 0.0
        ball_paddle = touching[2] if touching.size > 2 else 0.0

        paddle_contact = (
            (ball_paddle > 0.5) and
            (ball_table < 0.1) and
            (ball_net < 0.1)
        )

        # rising edge => count a hit once per contact
        hit = bool(paddle_contact and (not self._prev_paddle_contact))
        self._prev_paddle_contact = bool(paddle_contact)
        return hit

    # ============================================================
    # Reward (goal achieved)
    # ============================================================
    def _compute_reward(self) -> Tuple[float, Dict[str, Any]]:
        obs = self.env.unwrapped.obs_dict
        cfg = self.stage_cfg[min(max(self.training_stage, 0), 2)]

        # Safety: HRL state must exist
        if self.current_goal is None or self.goal_start_time is None or self.goal_start_ball_pos is None:
            return 0.0, {"goal_achieved": False}

        paddle = np.asarray(obs["paddle_pos"], dtype=np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], dtype=np.float32)
        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], dtype=np.float32)
        t = float(obs["time"])

        dx, dy, dz, dpx, dpy, dt = self.current_goal
        target_time = float(self.goal_start_time + float(dt))

        ball0 = self.goal_start_ball_pos
        target_paddle = ball0 + np.array([dx, dy, dz], dtype=np.float32)
        target_pelvis = ball0[:2] + np.array([dpx, dpy], dtype=np.float32)

        pos_err = float(np.linalg.norm(paddle - target_paddle))
        pelv_err = float(np.linalg.norm(pelvis_xy - target_pelvis))
        vel_norm = float(np.linalg.norm(paddle_vel))
        time_err = float(abs(t - target_time))

        reward = 0.0
        goal_achieved = False

        if t < target_time:
            reward += float(cfg["W_POS"] * np.exp(-4.0 * pos_err))
            reward += float(cfg["W_PELV"] * np.exp(-3.0 * pelv_err))
            reward -= float(cfg["W_VEL"] * vel_norm)
        else:
            reward += float(cfg["W_POS"] * np.exp(-8.0 * pos_err))
            reward += float(cfg["W_TIME"] * np.exp(-2.0 * time_err))
            reward -= float(cfg["W_VEL"] * vel_norm)

            if (
                pos_err < self.success_pos_thr
                and vel_norm < self.success_vel_thr
                and time_err < self.success_time_thr
            ):
                reward += float(self.success_bonus)
                goal_achieved = True

        return float(reward), {
            "goal_achieved": bool(goal_achieved),
            "position_error": pos_err,
            "pelvis_error": pelv_err,
            "velocity_norm": vel_norm,
            "time_error": time_err,
        }