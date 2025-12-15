from typing import Tuple, Dict, Optional
import numpy as np
from loguru import logger

from myosuite.utils import gym as myo_gym
from config import Config
from utils import quat_to_paddle_normal


class TableTennisWorker(myo_gym.Env):
    """
    Goal-conditioned Worker (motor primitive).

    Observation (17D):
      [paddle_pos(3),
       paddle_vel(3),
       paddle_ori_normal(3),
       time(1),
       goal(7)]

    Reward:
      smooth reaching + posture stability + (IMPORTANT) success bonus
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device

        # Base env
        self.env = myo_gym.make(config.env_id)

        # Goal space: [t, px, py, pz, vx, vy, vz]
        self.goal_low = np.array([0.0, -2.0, -1.0, 0.0, -5.0, -5.0, -5.0], dtype=np.float32)
        self.goal_high = np.array([3.0,  0.0,  1.0, 3.0,  5.0,  5.0,  5.0], dtype=np.float32)
        self.goal_space = myo_gym.spaces.Box(low=self.goal_low, high=self.goal_high, dtype=np.float32)

        # Observation: 10 state + 7 goal = 17
        self.state_dim = 10
        self.observation_dim = 17

        self.observation_space = myo_gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )
        self.action_space = self.env.action_space

        # Episode state
        self.current_goal: Optional[np.ndarray] = None
        self._initial_paddle_pos: Optional[np.ndarray] = None
        self._episode_goal_achieved = False

        # Curriculum
        self.training_stage = 0
        self.recent_successes = []
        self.total_episodes = 0

        # --- Reward knobs ---
        self.success_pos_thr = 0.06
        self.success_vel_thr = 0.6 
        self.success_time_thr = 0.08
        self.success_bonus = 12.0 
        self.post_target_vel_penalty = 0.2

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def set_goal(self, goal: np.ndarray):
        goal = np.asarray(goal, dtype=np.float32)
        self.current_goal = np.clip(goal, self.goal_low, self.goal_high)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        _, info = self.env.reset(seed=seed)

        obs_dict = self.env.obs_dict
        self._initial_paddle_pos = np.array(obs_dict["paddle_pos"], dtype=np.float32)

        self.total_episodes += 1
        self._episode_goal_achieved = False

        if self.current_goal is None:
            self.current_goal = self._sample_goal()

        obs = self._augment_observation()

        return obs, {
            **info,
            "goal": self.current_goal.copy(),
            "training_stage": int(self.training_stage),
        }

    def step(self, action: np.ndarray):
        _, base_reward, terminated, truncated, info = self.env.step(action)

        reward, rinfo = self._compute_reward()

        total_reward = float(reward + 0.05 * base_reward)

        if rinfo["goal_achieved"]:
            self._episode_goal_achieved = True

        if terminated or truncated:
            self._update_curriculum()
            self.current_goal = None

        obs = self._augment_observation()

        return obs, total_reward, terminated, truncated, {
            **info,
            **rinfo,
            "episode_goal_achieved": self._episode_goal_achieved,
            "is_success": bool(info.get("solved", False)),  # keep your key
        }

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _augment_observation(self) -> np.ndarray:
        obs = self.env.obs_dict

        paddle_pos = np.asarray(obs["paddle_pos"], dtype=np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], dtype=np.float32)

        # Convert quat → paddle normal (3D)
        paddle_ori_q = np.asarray(obs["paddle_ori"], dtype=np.float32)
        paddle_ori = quat_to_paddle_normal(paddle_ori_q)

        time = np.array([float(obs["time"])], dtype=np.float32)

        if self.current_goal is None:
            self.current_goal = self._sample_goal()

        state = np.hstack([paddle_pos, paddle_vel, paddle_ori, time])
        obs_vec = np.hstack([state, self.current_goal]).astype(np.float32)

        assert obs_vec.shape[0] == self.observation_dim, \
            f"Obs dim mismatch {obs_vec.shape[0]} != {self.observation_dim}"

        return obs_vec

    # ------------------------------------------------------------------
    # Goal sampling
    # ------------------------------------------------------------------

    def _sample_goal(self) -> np.ndarray:
        obs = self.env.obs_dict
        t0 = float(obs["time"])
        p0 = np.asarray(obs["paddle_pos"], dtype=np.float32)

        if self.training_stage == 0:
            dt = np.random.uniform(0.15, 0.3)
            dp = np.random.uniform(-0.2, 0.2, size=3)
            dv = np.random.uniform(-1.0, 1.0, size=3)

        elif self.training_stage == 1:
            dt = np.random.uniform(0.2, 0.4)
            dp = np.random.uniform(-0.4, 0.4, size=3)
            dv = np.random.uniform(-2.0, 2.0, size=3)

        else:
            dt = np.random.uniform(0.2, 0.5)
            dp = np.random.uniform(-0.8, 0.8, size=3)
            dv = np.random.uniform(-3.0, 3.0, size=3)

        goal = np.array([t0 + dt, *(p0 + dp), *dv], dtype=np.float32)
        return np.clip(goal, self.goal_low, self.goal_high)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> Tuple[float, Dict]:
        obs = self.env.obs_dict

        paddle_pos = np.asarray(obs["paddle_pos"], dtype=np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], dtype=np.float32)
        torso_up = float(obs.get("torso_up", 1.0))
        time = float(obs["time"])

        # Safety
        if self.current_goal is None:
            self.current_goal = self._sample_goal()

        target_time = float(self.current_goal[0])
        target_pos = np.asarray(self.current_goal[1:4], dtype=np.float32)

        pos_error = float(np.linalg.norm(paddle_pos - target_pos))
        vel_norm = float(np.linalg.norm(paddle_vel))
        time_error = float(abs(time - target_time))

        # Direction alignment (only helps pre-target)
        if pos_error > 1e-6 and vel_norm > 1e-6:
            dir_to_target = (target_pos - paddle_pos) / pos_error
            vel_dir = paddle_vel / vel_norm
            alignment = max(0.0, float(np.dot(dir_to_target, vel_dir)))
        else:
            alignment = 0.0

        reward = 0.0
        goal_achieved = False

        if time < target_time:
            # Pre-target: move toward target without going crazy
            reward += 2.0 * np.exp(-4.0 * pos_error)     # sharper than before
            reward += 0.6 * alignment
            reward -= 0.02 * vel_norm
            reward += 0.2 * torso_up

        else:
            # Post-target: encourage being AT the target and settled
            # Smooth shaping near success basin (helps get first successes)
            reward += 3.0 * np.exp(-8.0 * pos_error)
            reward += 1.5 * np.exp(-12.0 * time_error)

            # Settle term (prevents the "arm flail")
            reward -= self.post_target_vel_penalty * vel_norm

            # Hard success condition + BIG bonus (critical)
            if (
                pos_error < self.success_pos_thr
                and vel_norm < self.success_vel_thr
                and time_error < self.success_time_thr
            ):
                goal_achieved = True
                reward += self.success_bonus

        return float(reward), {
            "goal_achieved": bool(goal_achieved),
            "position_error": float(pos_error),
            "velocity_norm": float(vel_norm),
            "time_error": float(time_error),
        }

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _update_curriculum(self):
        self.recent_successes.append(1 if self._episode_goal_achieved else 0)
        if len(self.recent_successes) > 100:
            self.recent_successes.pop(0)

        if len(self.recent_successes) == 100:
            rate = float(np.mean(self.recent_successes))
            if rate > 0.6 and self.training_stage < 2:
                self.training_stage += 1
                self.recent_successes.clear()
                logger.info(f"[Worker] Curriculum → stage {self.training_stage}")

    # ------------------------------------------------------------------

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env

    def __getattr__(self, name):
        return getattr(self.env, name)
