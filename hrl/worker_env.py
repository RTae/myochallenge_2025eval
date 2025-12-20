from typing import Tuple, Dict, Optional
import numpy as np

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal
from myosuite.utils import gym


class TableTennisWorker(CustomEnv):
    """
    Goal-conditioned Worker.
    """

    def __init__(self, config: Config, training_stage: int = 0):
        super().__init__(config)

        # Goal bounds (6D)
        self.goal_low = np.array([-1.2, -0.6, -0.4, -0.8, -0.5, 0.15], dtype=np.float32)
        self.goal_high = np.array([ 0.6,  0.6,  0.6,  0.8,  0.5, 1.00], dtype=np.float32)

        self.goal_dim = 6
        self.observation_dim = 18

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # Goal state
        self.current_goal: Optional[np.ndarray] = None
        self.goal_start_time: Optional[float] = None
        self.goal_start_ball_pos: Optional[np.ndarray] = None
        self._episode_goal_achieved = False

        # Curriculum
        self.training_stage = training_stage
        self.recent_successes = []

        # Thresholds
        self.success_pos_thr = 0.12
        self.success_vel_thr = 1.2
        self.success_time_thr = 0.35
        self.success_bonus = 15.0

        self.stage_cfg = [
            dict(W_POS=2.5, W_PELV=1.5, W_TIME=0.5, W_VEL=0.1),
            dict(W_POS=2.5, W_PELV=1.5, W_TIME=1.2, W_VEL=0.25),
            dict(W_POS=3.0, W_PELV=2.0, W_TIME=2.0, W_VEL=0.45),
        ]

    # ------------------------------------------------
    # HRL API
    # ------------------------------------------------
    def set_goal(self, goal6: np.ndarray):
        goal6 = np.asarray(goal6, dtype=np.float32)
        assert goal6.shape == (6,)

        obs = self.env.obs_dict
        self.current_goal = goal6
        self.goal_start_time = float(obs["time"])
        self.goal_start_ball_pos = np.asarray(obs["ball_pos"], dtype=np.float32).copy()

    # ------------------------------------------------
    # Gym API
    # ------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        _, info = super().reset(seed)
        self._episode_goal_achieved = False
        self.current_goal = None
        self.goal_start_time = None
        self.goal_start_ball_pos = None
        return self._augment_observation(), info

    def step(self, action: np.ndarray):
        _, base_reward, terminated, truncated, info = super().step(action)

        reward, rinfo = self._compute_reward()
        total_reward = float(reward + 0.05 * base_reward)

        if rinfo["goal_achieved"]:
            self._episode_goal_achieved = True

        if terminated or truncated:
            self._update_curriculum()
            self.current_goal = None
            self.goal_start_time = None
            self.goal_start_ball_pos = None

        info.update({
            **rinfo,
            "episode_goal_achieved": bool(self._episode_goal_achieved),
            "is_success": bool(rinfo["goal_achieved"]),
        })

        return self._augment_observation(), total_reward, terminated, truncated, info

    # ------------------------------------------------
    # Observation / Reward (unchanged)
    # ------------------------------------------------
    def _augment_observation(self) -> np.ndarray:
        obs = self.env.obs_dict

        ball_pos = np.asarray(obs["ball_pos"], dtype=np.float32)
        paddle_pos = np.asarray(obs["paddle_pos"], dtype=np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], dtype=np.float32)
        paddle_ori = quat_to_paddle_normal(np.asarray(obs["paddle_ori"], dtype=np.float32))
        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], dtype=np.float32)
        t = np.array([float(obs["time"])], dtype=np.float32)

        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        paddle_rel = paddle_pos - ball_pos
        pelvis_rel = pelvis_xy - ball_pos[:2]

        state = np.hstack([paddle_rel, paddle_vel, paddle_ori, pelvis_rel, t])
        return np.hstack([state, self.current_goal]).astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        scale = [0.4, 0.7, 1.0][self.training_stage]
        return self.goal_low + scale * (self.goal_high - self.goal_low) * np.random.rand(self.goal_dim)

    def _compute_reward(self) -> Tuple[float, Dict]:
        obs = self.env.obs_dict
        cfg = self.stage_cfg[self.training_stage]

        paddle_pos = np.asarray(obs["paddle_pos"], dtype=np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], dtype=np.float32)
        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], dtype=np.float32)
        t = float(obs["time"])

        dx, dy, dz, dpx, dpy, dt = self.current_goal
        target_time = self.goal_start_time + dt

        ball0 = self.goal_start_ball_pos
        target_paddle = ball0 + np.array([dx, dy, dz])
        target_pelvis = ball0[:2] + np.array([dpx, dpy])

        pos_err = np.linalg.norm(paddle_pos - target_paddle)
        pelv_err = np.linalg.norm(pelvis_xy - target_pelvis)
        vel_norm = np.linalg.norm(paddle_vel)
        time_err = abs(t - target_time)

        reward = 0.0
        goal_achieved = False

        if t < target_time:
            reward += cfg["W_POS"] * np.exp(-4.0 * pos_err)
            reward += cfg["W_PELV"] * np.exp(-3.0 * pelv_err)
            reward -= cfg["W_VEL"] * vel_norm
        else:
            reward += cfg["W_POS"] * np.exp(-8.0 * pos_err)
            reward += cfg["W_TIME"] * np.exp(-2.0 * time_err)
            reward -= cfg["W_VEL"] * vel_norm

            if pos_err < self.success_pos_thr and vel_norm < self.success_vel_thr and time_err < self.success_time_thr:
                reward += self.success_bonus
                goal_achieved = True

        return reward, {"goal_achieved": goal_achieved}

    def _update_curriculum(self):
        self.recent_successes.append(int(self._episode_goal_achieved))
        if len(self.recent_successes) > 100:
            self.recent_successes.pop(0)
        if len(self.recent_successes) == 100 and np.mean(self.recent_successes) > 0.6:
            self.training_stage = min(self.training_stage + 1, 2)
            self.recent_successes.clear()
