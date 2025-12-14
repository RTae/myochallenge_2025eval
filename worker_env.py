from typing import Tuple, Dict, Optional
import numpy as np
from loguru import logger
from gymnasium import spaces

from myosuite.utils import gym as myo_gym
from config import Config


class TableTennisWorker:
    """
    Goal-conditioned Worker.
    - Observation: [paddle_pos(3), paddle_vel(3), time(1), goal(7)] => 14D
    - Reward: goal-conditioned shaping + small base env reward
    - Curriculum: advances by EPISODE success rate (not step-wise)
    """

    def __init__(self, config: Config, device: str = "cpu"):
        self.config = config
        self.device = device

        self.env = myo_gym.make(config.env_id)

        # Goal space: [t, px, py, pz, vx, vy, vz]
        self.goal_low = np.array([0.0, -2.0, -1.0, 0.0, -5.0, -5.0, -5.0], dtype=np.float32)
        self.goal_high = np.array([3.0, 0.0, 1.0, 3.0, 5.0, 5.0, 5.0], dtype=np.float32)
        self.goal_space = spaces.Box(low=self.goal_low, high=self.goal_high, dtype=np.float32)

        # Obs: paddle_pos(3) + paddle_vel(3) + time(1) + goal(7)
        self.state_dim = 7
        self.observation_dim = self.state_dim + self.goal_space.shape[0]  # 14
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        self.action_space = self.env.action_space

        self.current_goal: Optional[np.ndarray] = None
        self._initial_paddle_pos: Optional[np.ndarray] = None

        # Reward parameters
        self.position_weight = 2.0
        self.velocity_weight = 1.0
        self.success_bonus = 20.0

        # Curriculum
        self.training_stage = 0
        self.recent_successes = []  # last up to 100 EPISODES
        self.total_episodes = 0

        # Episode-level success flag
        self._episode_goal_achieved = False

    # -------------------------
    # Public API
    # -------------------------
    def set_goal(self, goal: np.ndarray):
        """Set external goal (e.g., from Manager)."""
        goal = np.asarray(goal, dtype=np.float32)
        self.current_goal = np.clip(goal, self.goal_low, self.goal_high).astype(np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed)

        obs_dict = self.env.obs_dict
        self._initial_paddle_pos = np.array(obs_dict["paddle_pos"], dtype=np.float32).copy()

        self.total_episodes += 1
        self._episode_goal_achieved = False

        # If no external goal set yet, sample one
        if self.current_goal is None:
            self.current_goal = self._sample_absolute_target()

        augmented_obs = self._augment_observation()

        info = dict(info)
        info.update({
            "goal": self.current_goal.copy(),
            "training_stage": self.training_stage,
            "total_episodes": self.total_episodes,
        })
        return augmented_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Goal-conditioned shaping
        goal_reward, goal_info = self._calculate_absolute_target_reward()

        # Mark goal achieved once per episode
        if goal_info.get("goal_achieved", False):
            self._episode_goal_achieved = True

        # Combine rewards
        total_reward = float(goal_reward + 0.1 * float(base_reward))

        # If episode ended, update curriculum using EPISODE success
        if terminated or truncated:
            self.recent_successes.append(1 if self._episode_goal_achieved else 0)
            if len(self.recent_successes) > 100:
                self.recent_successes.pop(0)

            if len(self.recent_successes) == 100:
                success_rate = float(np.mean(self.recent_successes))
                if success_rate > 0.6 and self.training_stage < 2:
                    self.training_stage += 1
                    logger.info(f"[Worker] Advanced to stage {self.training_stage} (success={success_rate:.2f})")
                    self.recent_successes = []

            # Clear goal for next episode unless a manager overwrites it
            self.current_goal = None

        augmented_obs = self._augment_observation()

        info = dict(info)
        info.update({
            "goal_reward": float(goal_reward),
            "goal_achieved": bool(goal_info.get("goal_achieved", False)),
            "episode_goal_achieved": bool(self._episode_goal_achieved),
            "training_stage": int(self.training_stage),
            "total_episodes": int(self.total_episodes),
            "recent_success_rate": float(np.mean(self.recent_successes)) if self.recent_successes else 0.0,
            "position_error": float(goal_info.get("position_error", 0.0)),
            "time_error": float(goal_info.get("time_error", 0.0)),
            "velocity_error": float(goal_info.get("velocity_error", 0.0)),
        })

        return augmented_obs, total_reward, terminated, truncated, info

    # -------------------------
    # Internal helpers
    # -------------------------
    def _augment_observation(self) -> np.ndarray:
        obs_dict = self.env.obs_dict
        paddle_pos = np.array(obs_dict["paddle_pos"], dtype=np.float32).flatten()
        paddle_vel = np.array(obs_dict["paddle_vel"], dtype=np.float32).flatten()
        current_time = np.array([float(obs_dict["time"])], dtype=np.float32)

        if self.current_goal is None:
            # Should not happen in normal flow, but keep safe
            self.current_goal = self._sample_absolute_target()

        state = np.hstack([paddle_pos, paddle_vel, current_time])
        augmented_obs = np.hstack([state, self.current_goal]).astype(np.float32)
        return augmented_obs

    def _sample_absolute_target(self) -> np.ndarray:
        obs_dict = self.env.obs_dict
        current_time = float(obs_dict["time"])
        current_paddle_pos = np.array(obs_dict["paddle_pos"], dtype=np.float32)

        if self.training_stage == 0:
            target_time = current_time + float(np.random.uniform(0.1, 0.3))
            target_pos = current_paddle_pos + np.random.uniform(-0.2, 0.2, 3).astype(np.float32)
            target_vel = np.random.uniform(-1.0, 1.0, 3).astype(np.float32)

        elif self.training_stage == 1:
            target_time = current_time + float(np.random.uniform(0.15, 0.4))
            table_positions = np.array([
                [-1.0, -0.5, 1.2],
                [-1.0,  0.0, 1.2],
                [-1.0,  0.5, 1.2],
                [-0.5, -0.5, 1.5],
                [-0.5,  0.0, 1.5],
                [-0.5,  0.5, 1.5],
            ], dtype=np.float32)
            idx = int(np.random.randint(0, len(table_positions)))
            target_pos = table_positions[idx] + np.random.uniform(-0.1, 0.1, 3).astype(np.float32)
            target_vel = np.random.uniform(-2.0, 2.0, 3).astype(np.float32)

        else:
            target_time = current_time + float(np.random.uniform(0.1, 0.5))
            target_pos = np.array([
                np.random.uniform(-1.5, -0.3),
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(0.8, 2.0),
            ], dtype=np.float32)
            target_vel = np.random.uniform(-3.0, 3.0, 3).astype(np.float32)

        goal = np.array([
            float(target_time),
            float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
            float(target_vel[0]), float(target_vel[1]), float(target_vel[2]),
        ], dtype=np.float32)

        return np.clip(goal, self.goal_low, self.goal_high).astype(np.float32)

    def _calculate_absolute_target_reward(self) -> Tuple[float, Dict]:
        obs_dict = self.env.obs_dict

        paddle_pos = np.array(obs_dict["paddle_pos"], dtype=np.float32)
        paddle_vel = np.array(obs_dict["paddle_vel"], dtype=np.float32)
        current_time = float(obs_dict["time"])

        if self.current_goal is None:
            self.current_goal = self._sample_absolute_target()

        target_time = float(self.current_goal[0])
        target_pos = self.current_goal[1:4]
        target_vel = self.current_goal[4:7]

        time_error = abs(current_time - target_time)
        pos_error = float(np.linalg.norm(paddle_pos - target_pos))
        vel_error = float(np.linalg.norm(paddle_vel - target_vel))

        reward = 0.0
        goal_achieved = False

        if current_time < target_time:
            time_remaining = target_time - current_time
            if time_remaining > 1e-6:
                pos_to_go = target_pos - paddle_pos
                desired_vel = pos_to_go / time_remaining

                vel_norm = float(np.linalg.norm(paddle_vel))
                desired_norm = float(np.linalg.norm(desired_vel))
                if vel_norm > 0 and desired_norm > 0:
                    vel_alignment = float(np.dot(paddle_vel, desired_vel) / (vel_norm * desired_norm))
                    vel_alignment = max(0.0, vel_alignment)
                else:
                    vel_alignment = 0.0

                if self._initial_paddle_pos is not None:
                    initial_distance = float(np.linalg.norm(target_pos - self._initial_paddle_pos))
                    initial_distance = max(0.1, initial_distance)
                    current_progress = 1.0 - (pos_error / initial_distance)
                else:
                    current_progress = 1.0 - min(1.0, pos_error / 1.0)

                reward = (
                    self.position_weight * float(current_progress) +
                    self.velocity_weight * float(vel_alignment)
                )

        else:
            if (time_error < 0.02) and (pos_error < 0.05) and (vel_error < 0.5):
                reward = self.success_bonus - pos_error - 0.5 * vel_error
                goal_achieved = True
            else:
                reward = -pos_error - 0.5 * vel_error - time_error

        info = {
            "goal_achieved": bool(goal_achieved),
            "time_error": float(time_error),
            "position_error": float(pos_error),
            "velocity_error": float(vel_error),
            "current_time": float(current_time),
            "target_time": float(target_time),
        }
        return float(reward), info

    # Forwarding helpers
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env

    def __getattr__(self, name):
        return getattr(self.env, name)
