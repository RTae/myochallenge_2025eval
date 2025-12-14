from typing import Tuple, Dict, Optional, Any
import numpy as np

from config import Config
from myosuite.utils import gym as myo_gym

from utils import quat_to_paddle_normal

class TableTennisManager(myo_gym.Env):
    """
    Manager (high-level policy):
    - Action: worker goal (7D): [t, px, py, pz, vx, vy, vz]
    - Observation (25D):
        ball_pos(3), ball_vel(3),
        paddle_pos(3), paddle_vel(3),
        paddle_ori(3),
        reach_err(3),
        touching_info(6),
        time(1)
    - Runs frozen worker for N low-level steps per decision.
    """

    metadata = {"render_modes": []}

    def __init__(self, worker_env: Any, worker_model: Any, config: Config):
        super().__init__()
        self.config = config
        self.worker_env = worker_env
        self.worker_model = worker_model

        # Timing
        self.env_time_step = 0.01
        self.max_episode_steps = int(config.episode_len)

        # Decision frequency
        self.manager_decision_interval = 10
        self.worker_steps_per_decision = self.manager_decision_interval
        self.max_manager_decisions = max(1, self.max_episode_steps // self.manager_decision_interval)

        # Observation/action spaces
        self.observation_dim = 25
        self.action_dim = 7

        self.observation_space = myo_gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )
        
        self.action_space = myo_gym.spaces.Box(
            low=self.worker_env.goal_low,
            high=self.worker_env.goal_high,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # Tracking
        self.current_step = 0
        self.current_decision = 0
        self.total_hits = 0
        self.total_manager_episodes = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        self.worker_env.reset(seed=seed)

        self.current_step = 0
        self.current_decision = 0
        self.total_manager_episodes += 1

        manager_obs = self._augment_observation()
        info = {
            "manager_episode": int(self.total_manager_episodes),
            "total_hits": int(self.total_hits),
            "current_step": int(self.current_step),
            "current_decision": int(self.current_decision),
        }
        return manager_obs, info

    def step(self, manager_action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        manager_action = np.asarray(manager_action, dtype=np.float32)
        manager_action = np.clip(manager_action, self.worker_env.goal_low, self.worker_env.goal_high)

        # Set goal for worker
        self.worker_env.set_goal(manager_action)

        worker_reward_total = 0.0
        hit_occurred = False
        contact_info: Dict = {}

        worker_terminated = False
        worker_truncated = False
        worker_info: Dict = {}

        for _ in range(self.worker_steps_per_decision):
            self.current_step += 1

            # Worker obs (goal-conditioned)
            worker_obs = self.worker_env._augment_observation()

            # Low-level action from frozen worker policy
            worker_action, _ = self.worker_model.predict(worker_obs, deterministic=True)

            # Step worker wrapper
            _, worker_reward, worker_terminated, worker_truncated, worker_info = self.worker_env.step(worker_action)
            worker_reward_total += float(worker_reward)

            # Detect hit (touching_info[0] is Paddle contact)
            obs_dict = self._get_obs_dict()
            touching = np.array(obs_dict.get("touching_info", np.zeros(6)), dtype=np.float32).flatten()
            if len(touching) > 0 and touching[0] > 0.5 and not hit_occurred:
                hit_occurred = True
                contact_info = {
                    "hit_time": float(obs_dict["time"]),
                    "hit_position": np.array(obs_dict["paddle_pos"], dtype=np.float32).copy(),
                    "step": int(self.current_step),
                }
                self.total_hits += 1

            if worker_terminated or worker_truncated:
                break

        self.current_decision += 1

        manager_reward = self._calculate_manager_reward(
            hit_occurred=hit_occurred,
            manager_action=manager_action,
            contact_info=contact_info,
        )

        terminated = bool(worker_terminated)
        truncated = bool(
            worker_truncated
            or self.current_step >= self.max_episode_steps
            or self.current_decision >= self.max_manager_decisions
        )

        next_manager_obs = (
            self._augment_observation()
            if (not terminated and not truncated)
            else np.zeros(self.observation_dim, dtype=np.float32)
        )

        info = {
            "manager_step": int(self.current_step),
            "manager_decision": int(self.current_decision),
            "max_decisions": int(self.max_manager_decisions),
            "manager_reward": float(manager_reward),
            "hit_occurred": bool(hit_occurred),
            "total_hits": int(self.total_hits),
            "worker_reward_total": float(worker_reward_total),  # logging only
            "worker_terminated": bool(worker_terminated),
            "worker_truncated": bool(worker_truncated),
            "target_time": float(manager_action[0]),
            "target_pos": manager_action[1:4].tolist(),
            "target_vel": manager_action[4:7].tolist(),
            "is_success": bool(worker_info.get("solved", False)),
        }

        if hit_occurred:
            info.update({
                "hit_time": float(contact_info.get("hit_time", 0.0)),
                "hit_position": contact_info.get("hit_position", np.zeros(3)).tolist(),
                "hit_step": int(contact_info.get("step", 0)),
            })

        return next_manager_obs, float(manager_reward), terminated, truncated, info

    # -------------------------
    # Observation & reward
    # -------------------------
    def _get_obs_dict(self):
        # Base env obs_dict is inside worker_env.env
        return self.worker_env.env.obs_dict

    def _augment_observation(self) -> np.ndarray:
        obs_dict = self._get_obs_dict()

        ball_pos = np.array(obs_dict["ball_pos"], dtype=np.float32).flatten()
        ball_vel = np.array(obs_dict["ball_vel"], dtype=np.float32).flatten()

        paddle_pos = np.array(obs_dict["paddle_pos"], dtype=np.float32).flatten()
        paddle_vel = np.array(obs_dict["paddle_vel"], dtype=np.float32).flatten()

        paddle_ori_q = np.array(obs_dict["paddle_ori"], dtype=np.float32).flatten()
        paddle_ori = quat_to_paddle_normal(paddle_ori_q)

        reach_err = np.array(obs_dict["reach_err"], dtype=np.float32).flatten()
        touching_info = np.array(obs_dict["touching_info"], dtype=np.float32).flatten()

        current_time = np.array([float(obs_dict["time"])], dtype=np.float32)

        manager_obs = np.hstack([
            ball_pos,         # 3
            ball_vel,         # 3
            paddle_pos,       # 3
            paddle_vel,       # 3
            paddle_ori,       # 3
            reach_err,        # 3
            touching_info,    # 6
            current_time,     # 1
        ]).astype(np.float32)

        assert manager_obs.shape[0] == self.observation_dim, \
            f"Manager obs dim mismatch: {manager_obs.shape[0]} vs {self.observation_dim}"

        return manager_obs


    def _calculate_manager_reward(
        self,
        hit_occurred: bool,
        manager_action: np.ndarray,
        contact_info: Dict
    ) -> float:
        obs_dict = self._get_obs_dict()

        ball_pos = np.array(obs_dict["ball_pos"], dtype=np.float32)
        paddle_pos = np.array(obs_dict["paddle_pos"], dtype=np.float32)
        current_time = float(obs_dict["time"])

        target_time = float(manager_action[0])
        target_pos = np.array(manager_action[1:4], dtype=np.float32)

        # Dense shaping
        dist_ball_paddle = float(np.linalg.norm(ball_pos - paddle_pos))
        dist_goal_ball = float(np.linalg.norm(target_pos - ball_pos))
        timing_error = abs(target_time - current_time)

        reward = 0.0
        reward += 5.0 * np.exp(-5.0 * dist_ball_paddle)
        reward += 2.0 * np.exp(-4.0 * dist_goal_ball)
        reward += 2.0 * np.exp(-3.0 * timing_error)

        # Hit bonus + accuracy
        if hit_occurred:
            reward += 30.0
            hit_time = float(contact_info.get("hit_time", target_time))
            hit_pos = np.array(contact_info.get("hit_position", paddle_pos), dtype=np.float32)

            reward += 10.0 * np.exp(-10.0 * abs(hit_time - target_time))
            reward += 10.0 * np.exp(-10.0 * float(np.linalg.norm(hit_pos - ball_pos)))

        # Invalid timing penalties
        if target_time < current_time:
            reward -= 20.0
        if target_time > current_time + 1.5:
            reward -= 5.0 * (target_time - current_time - 1.5)

        return float(reward)

    # Forwarding helpers
    def render(self):
        return self.worker_env.render()

    def close(self):
        return self.worker_env.close()

    @property
    def unwrapped(self):
        return self

    def __getattr__(self, name):
        return getattr(self.worker_env, name)
