from typing import Tuple, Dict, Optional
import numpy as np
from loguru import logger

from myosuite.utils import gym as myo_gym
from config import Config
from utils import quat_to_paddle_normal


class TableTennisWorker(myo_gym.Env):
    """
    Goal-conditioned Worker (motor primitive)

    Observation (18D):
      [ paddle_pos_rel(3),
        paddle_vel(3),
        paddle_ori_normal(3),
        pelvis_xy_rel(2),
        time(1),
        goal(6) ]

    Goal (6D, RELATIVE TO BALL):
      [ dx, dy, dz, dpx, dpy, dt ]

    Reward enforces:
      - paddle reaching
      - pelvis positioning (NO arm-only cheating)
      - settling near target
      - explicit success bonus
    """

    metadata = {"render_modes": []}

    # ============================================================
    # Init
    # ============================================================

    def __init__(self, config: Config, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device

        # ---------------- Base env ----------------
        self.env = myo_gym.make(config.env_id)

        # ---------------- Goal bounds (RELATIVE, TIGHT) ----------------
        # [dx, dy, dz, dpx, dpy, dt]
        self.goal_low = np.array(
            [-1.2, -0.6, -0.4,   # paddle rel
             -0.8, -0.5,         # pelvis rel
              0.15],             # dt
            dtype=np.float32
        )
        self.goal_high = np.array(
            [ 0.6,  0.6,  0.6,
              0.8,  0.5,
              1.00],
            dtype=np.float32
        )

        self.goal_dim = 6

        # ---------------- Observation ----------------
        # 3 + 3 + 3 + 2 + 1 + 6 = 18
        self.observation_dim = 18

        self.observation_space = myo_gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )

        self.action_space = self.env.action_space

        # ---------------- Episode state ----------------
        self.current_goal: Optional[np.ndarray] = None
        self._episode_goal_achieved = False

        # ---------------- Curriculum ----------------
        self.training_stage = 0
        self.recent_successes = []

        # ---------------- Success thresholds ----------------
        self.success_pos_thr = 0.12     # meters
        self.success_vel_thr = 1.2      # m/s
        self.success_time_thr = 0.30    # seconds
        self.success_bonus = 15.0

    # ============================================================
    # Reset / Step
    # ============================================================

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        _, info = self.env.reset(seed=seed)
        self._episode_goal_achieved = False
        self.current_goal = None
        return self._augment_observation(), info

    def step(self, action: np.ndarray):
        _, base_reward, terminated, truncated, info = self.env.step(action)

        reward, rinfo = self._compute_reward()
        total_reward = float(reward + 0.05 * base_reward)

        if rinfo["goal_achieved"]:
            self._episode_goal_achieved = True

        if terminated or truncated:
            self._update_curriculum()
            self.current_goal = None

        return (
            self._augment_observation(),
            total_reward,
            terminated,
            truncated,
            {
                **info,
                **rinfo,
                "episode_goal_achieved": self._episode_goal_achieved,
                "is_success": bool(info.get("solved", False)),
            },
        )

    # ============================================================
    # Observation
    # ============================================================

    def _augment_observation(self) -> np.ndarray:
        obs = self.env.obs_dict

        ball_pos = np.asarray(obs["ball_pos"], dtype=np.float32)
        paddle_pos = np.asarray(obs["paddle_pos"], dtype=np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], dtype=np.float32)

        paddle_ori_q = np.asarray(obs["paddle_ori"], dtype=np.float32)
        paddle_ori = quat_to_paddle_normal(paddle_ori_q)

        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], dtype=np.float32)

        paddle_rel = paddle_pos - ball_pos
        pelvis_rel = pelvis_xy - ball_pos[:2]

        time = np.array([float(obs["time"])], dtype=np.float32)

        if self.current_goal is None:
            self.current_goal = self._sample_goal()

        state = np.hstack([
            paddle_rel,          # 3
            paddle_vel,          # 3
            paddle_ori,          # 3
            pelvis_rel,          # 2
            time,                # 1
        ])

        obs_vec = np.hstack([state, self.current_goal]).astype(np.float32)
        return obs_vec

    # ============================================================
    # Goal sampling
    # ============================================================

    def _sample_goal(self) -> np.ndarray:
        if self.training_stage == 0:
            scale = 0.4
        elif self.training_stage == 1:
            scale = 0.7
        else:
            scale = 1.0

        goal = self.goal_low + scale * (self.goal_high - self.goal_low) * np.random.rand(self.goal_dim)
        return goal.astype(np.float32)

    # ============================================================
    # Reward
    # ============================================================

    def _compute_reward(self) -> Tuple[float, Dict]:
        obs = self.env.obs_dict

        ball_pos = np.asarray(obs["ball_pos"], dtype=np.float32)
        paddle_pos = np.asarray(obs["paddle_pos"], dtype=np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], dtype=np.float32)
        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], dtype=np.float32)
        time = float(obs["time"])

        if self.current_goal is None:
            self.current_goal = self._sample_goal()

        dx, dy, dz, dpx, dpy, dt = self.current_goal

        target_time = time + dt
        target_paddle = ball_pos + np.array([dx, dy, dz])
        target_pelvis = ball_pos[:2] + np.array([dpx, dpy])

        pos_error = np.linalg.norm(paddle_pos - target_paddle)
        vel_norm = np.linalg.norm(paddle_vel)
        pelvis_error = np.linalg.norm(pelvis_xy - target_pelvis)
        time_error = abs(time - target_time)

        reward = 0.0
        goal_achieved = False

        # ---------------- Pre-target ----------------
        if time < target_time:
            reward += 2.5 * np.exp(-4.0 * pos_error)
            reward += 1.2 * np.exp(-3.0 * pelvis_error)
            reward -= 0.02 * vel_norm

        # ---------------- Post-target ----------------
        else:
            reward += 3.5 * np.exp(-8.0 * pos_error)
            reward += 2.0 * np.exp(-2.0 * time_error)
            reward -= 0.4 * vel_norm

            if (
                pos_error < self.success_pos_thr
                and vel_norm < self.success_vel_thr
                and time_error < self.success_time_thr
            ):
                reward += self.success_bonus
                goal_achieved = True

        return float(reward), {
            "goal_achieved": goal_achieved,
            "position_error": float(pos_error),
            "pelvis_error": float(pelvis_error),
            "velocity_norm": float(vel_norm),
            "time_error": float(time_error),
        }

    # ============================================================
    # Curriculum
    # ============================================================

    def _update_curriculum(self):
        self.recent_successes.append(1 if self._episode_goal_achieved else 0)
        if len(self.recent_successes) > 100:
            self.recent_successes.pop(0)

        if len(self.recent_successes) == 100:
            rate = np.mean(self.recent_successes)
            if rate > 0.6 and self.training_stage < 2:
                self.training_stage += 1
                self.recent_successes.clear()
                logger.info(f"[Worker] Curriculum → stage {self.training_stage}")

    # ============================================================

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env

    def __getattr__(self, name):
        return getattr(self.env, name)
