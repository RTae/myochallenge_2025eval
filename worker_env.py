from typing import Tuple, Dict, Optional
import numpy as np
from loguru import logger

from myosuite.utils import gym as myo_gym
from config import Config
from utils import quat_to_paddle_normal


class TableTennisWorker(myo_gym.Env):
    """
    Goal-conditioned Worker (motor primitive).

    Observation (20D):
      [paddle_pos(3), paddle_vel(3), paddle_ori(3), reach_err(3), time(1), goal(7)]

    Action:
      same as base env action_space

    Reward:
      smooth approach shaping to goal (no ball semantics besides reach_err)

    Curriculum:
      advances by EPISODE success rate (not step-wise)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device

        self.env = myo_gym.make(config.env_id)

        # Goal space: [t, px, py, pz, vx, vy, vz]
        self.goal_low = np.array([0.0, -2.0, -1.0, 0.0, -5.0, -5.0, -5.0], dtype=np.float32)
        self.goal_high = np.array([3.0, 0.0, 1.0, 3.0, 5.0, 5.0, 5.0], dtype=np.float32)
        self.goal_space = myo_gym.spaces.Box(low=self.goal_low, high=self.goal_high, dtype=np.float32)

        # Obs: paddle_pos(3)+paddle_vel(3)+paddle_ori(3)+reach_err(3)+time(1) + goal(7) = 20
        self.state_dim = 13
        self.observation_dim = self.state_dim + self.goal_space.shape[0]  # 20
        self.observation_space = myo_gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        self.action_space = self.env.action_space

        self.current_goal: Optional[np.ndarray] = None
        self._initial_paddle_pos: Optional[np.ndarray] = None

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
            "training_stage": int(self.training_stage),
            "total_episodes": int(self.total_episodes),
        })
        return augmented_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        _, base_reward, terminated, truncated, info = self.env.step(action)

        # Goal-conditioned shaping (worker reward)
        goal_reward, goal_info = self._calculate_reward()

        # Mark goal achieved once per episode
        if goal_info.get("goal_achieved", False):
            self._episode_goal_achieved = True

        # Combine rewards (base env reward is small, mostly for “alive” signals)
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
            "velocity_norm": float(goal_info.get("velocity_norm", 0.0)),
            "moving_toward_target": bool(goal_info.get("moving_toward_target", True)),
            "is_success": bool(info.get("solved", False)),
        })

        return augmented_obs, total_reward, terminated, truncated, info
    
    def _augment_observation(self) -> np.ndarray:
        obs_dict = self.env.obs_dict

        paddle_pos = np.array(obs_dict["paddle_pos"], dtype=np.float32).flatten()
        paddle_vel = np.array(obs_dict["paddle_vel"], dtype=np.float32).flatten()

        paddle_ori_q = np.array(obs_dict["paddle_ori"], dtype=np.float32).flatten()
        paddle_ori = quat_to_paddle_normal(paddle_ori_q)

        reach_err = np.array(obs_dict["reach_err"], dtype=np.float32).flatten()
        current_time = np.array([float(obs_dict["time"])], dtype=np.float32)

        if self.current_goal is None:
            self.current_goal = self._sample_absolute_target()

        state = np.hstack([
            paddle_pos,      # 3
            paddle_vel,      # 3
            paddle_ori,      # 3
            reach_err,       # 3
            current_time     # 1
        ])

        augmented_obs = np.hstack([state, self.current_goal]).astype(np.float32)

        assert augmented_obs.shape[0] == self.observation_dim, \
            f"Worker obs dim mismatch: {augmented_obs.shape[0]} vs {self.observation_dim}"

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

    def _calculate_reward(self) -> Tuple[float, Dict]:
        obs_dict = self.env.obs_dict

        paddle_pos = np.array(obs_dict["paddle_pos"], dtype=np.float32)
        paddle_vel = np.array(obs_dict["paddle_vel"], dtype=np.float32)
        current_time = float(obs_dict["time"])

        if self.current_goal is None:
            self.current_goal = self._sample_absolute_target()

        target_time = float(self.current_goal[0])
        target_pos = self.current_goal[1:4]

        # Errors
        pos_error = float(np.linalg.norm(paddle_pos - target_pos))
        vel_norm = float(np.linalg.norm(paddle_vel))
        time_error = abs(current_time - target_time)

        reward = 0.0
        goal_achieved = False

        # Direction terms
        if pos_error > 1e-6:
            direction_to_target = (target_pos - paddle_pos) / pos_error
        else:
            direction_to_target = np.zeros(3, dtype=np.float32)

        if vel_norm > 1e-6:
            vel_dir = paddle_vel / vel_norm
        else:
            vel_dir = np.zeros(3, dtype=np.float32)

        moving_toward_target = np.dot(direction_to_target, vel_dir) > 0.0

        if current_time < target_time:
            time_remaining = max(0.05, target_time - current_time)

            # Progress reward
            if self._initial_paddle_pos is not None:
                initial_dist = float(np.linalg.norm(target_pos - self._initial_paddle_pos))
                initial_dist = max(0.1, initial_dist)
                progress = 1.0 - min(1.0, pos_error / initial_dist)
            else:
                progress = 1.0 - min(1.0, pos_error)

            reward += 2.0 * progress

            # Direction alignment (gentle)
            alignment = max(0.0, float(np.dot(vel_dir, direction_to_target)))
            reward += 0.5 * alignment

            # Optimal speed (clamped)
            optimal_speed = np.clip(pos_error / time_remaining, 0.0, 3.0)
            speed_error = abs(vel_norm - optimal_speed)
            reward -= 0.2 * speed_error

            # Penalty for ballistic motion
            reward -= 0.03 * vel_norm

            # Penalize moving away
            if not moving_toward_target and pos_error > 0.05:
                reward -= 0.3

        else:
            # Continuous accuracy-based reward (no gambling)
            reward += 5.0 * np.exp(-20.0 * pos_error)
            reward -= 0.5 * vel_norm
            reward -= 1.5 * time_error

            # Success flag only (curriculum)
            if pos_error < 0.05 and vel_norm < 0.5 and time_error < 0.05:
                goal_achieved = True

        info = {
            "goal_achieved": bool(goal_achieved),
            "position_error": float(pos_error),
            "velocity_norm": float(vel_norm),
            "time_error": float(time_error),
            "moving_toward_target": bool(moving_toward_target),
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
