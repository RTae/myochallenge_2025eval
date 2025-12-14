from typing import Tuple, Dict, Optional, Any
import numpy as np

from config import Config
from myosuite.utils import gym as myo_gym


class TableTennisManager(myo_gym.Env):
    """
    Manager (high-level policy):
    - Action: worker goal (7D)  [t, px, py, pz, vx, vy, vz]
    - Observation: 18D (ball/paddle state + derived features)
    - Runs worker for N low-level steps per decision using a frozen worker policy.
    """

    def __init__(self, worker_env: Any, worker_model: Any, config: Config):
        self.config = config
        self.worker_env = worker_env
        self.worker_model = worker_model

        # Timing
        self.env_time_step = 0.01
        self.max_episode_steps = int(config.episode_len)

        # Decision frequency
        self.manager_decision_interval = 10
        self.worker_steps_per_decision = self.manager_decision_interval
        self.max_manager_decisions = self.max_episode_steps // self.manager_decision_interval

        # Observation/action spaces
        self.observation_dim = 18
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

        # Constants
        self.net_height = self.config.NET_HEIGHT

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        self.worker_env.reset(seed=seed)

        self.current_step = 0
        self.current_decision = 0
        self.total_manager_episodes += 1

        manager_obs = self._extract_manager_observation()
        info = {
            "manager_episode": self.total_manager_episodes,
            "total_hits": self.total_hits,
            "current_step": self.current_step,
            "current_decision": self.current_decision,
        }
        return manager_obs, info

    def step(self, manager_action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        manager_action = np.asarray(manager_action, dtype=np.float32)
        manager_action = np.clip(manager_action, self.worker_env.goal_low, self.worker_env.goal_high)

        # Set goal via API
        self.worker_env.set_goal(manager_action)

        worker_reward_total = 0.0
        hit_occurred = False
        contact_info: Dict = {}

        worker_terminated = False
        worker_truncated = False

        for _ in range(self.worker_steps_per_decision):
            self.current_step += 1

            # Worker obs (goal-conditioned)
            worker_obs = self.worker_env._augment_observation()

            # Low-level action from frozen worker policy
            worker_action, _ = self.worker_model.predict(worker_obs, deterministic=True)

            # IMPORTANT: step through worker wrapper (NOT base env)
            _, worker_reward, worker_terminated, worker_truncated, worker_info = self.worker_env.step(worker_action)
            worker_reward_total += float(worker_reward)

            # Detect hit (touching_info[0] > 0.5)
            obs_dict = self._get_obs_dict()
            if "touching_info" in obs_dict:
                touching = np.array(obs_dict["touching_info"]).flatten()
                if len(touching) > 0 and touching[0] > 0.5 and not hit_occurred:
                    hit_occurred = True
                    contact_info = {
                        "hit_time": float(obs_dict["time"]),
                        "hit_position": np.array(obs_dict["paddle_pos"], dtype=np.float32).copy(),
                        "hit_velocity": np.array(obs_dict["paddle_vel"], dtype=np.float32).copy(),
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

        if not terminated and not truncated:
            next_manager_obs = self._extract_manager_observation()
        else:
            next_manager_obs = np.zeros(self.observation_dim, dtype=np.float32)

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
            "is_success": worker_info['solved'], # Using base env success signal
        }

        if hit_occurred:
            info.update({
                "hit_time": float(contact_info.get("hit_time", 0.0)),
                "hit_position": contact_info.get("hit_position", np.zeros(3)).tolist(),
                "hit_velocity": contact_info.get("hit_velocity", np.zeros(3)).tolist(),
                "hit_step": int(contact_info.get("step", 0)),
            })

        return next_manager_obs, float(manager_reward), terminated, truncated, info

    # -------------------------
    # Observation & reward
    # -------------------------
    def _get_obs_dict(self):
        return self.worker_env.env.obs_dict

    def _extract_manager_observation(self) -> np.ndarray:
        obs_dict = self._get_obs_dict()

        ball_pos = np.array(obs_dict["ball_pos"], dtype=np.float32).flatten()
        ball_vel = np.array(obs_dict["ball_vel"], dtype=np.float32).flatten()
        paddle_pos = np.array(obs_dict["paddle_pos"], dtype=np.float32).flatten()
        paddle_vel = np.array(obs_dict["paddle_vel"], dtype=np.float32).flatten()
        current_time = float(obs_dict["time"])

        # Derived features
        ball_height_ratio = float(ball_pos[2]) / float(self.net_height)

        distance_to_ball = float(np.linalg.norm(ball_pos - paddle_pos))

        net_x = -0.2
        if abs(float(ball_vel[0])) > 1e-6:
            time_to_net = (net_x - float(ball_pos[0])) / float(ball_vel[0])
            time_to_net = float(np.clip(time_to_net, 0.0, 2.0))
        else:
            time_to_net = 2.0

        ball_speed = float(np.linalg.norm(ball_vel))
        if ball_speed > 0.1:
            horizontal_angle = float(np.arctan2(ball_vel[1], ball_vel[0]))
            vertical_angle = float(np.arctan2(ball_vel[2], np.linalg.norm(ball_vel[:2])))
        else:
            horizontal_angle = 0.0
            vertical_angle = 0.0

        manager_obs = np.array([
            # ball pos (3)
            float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2]),
            # ball vel (3)
            float(ball_vel[0]), float(ball_vel[1]), float(ball_vel[2]),
            # paddle pos (3)
            float(paddle_pos[0]), float(paddle_pos[1]), float(paddle_pos[2]),
            # paddle vel (3)
            float(paddle_vel[0]), float(paddle_vel[1]), float(paddle_vel[2]),
            # derived (6) => total 18
            current_time,
            ball_height_ratio,
            distance_to_ball,
            time_to_net,
            horizontal_angle,
            vertical_angle,
        ], dtype=np.float32)

        return manager_obs

    def _calculate_manager_reward(self,
                                 hit_occurred: bool,
                                 manager_action: np.ndarray,
                                 contact_info: Dict) -> float:
        reward = 0.0

        if hit_occurred:
            reward += 100.0

            target_time = float(manager_action[0])
            hit_time = float(contact_info.get("hit_time", target_time))
            timing_error = abs(hit_time - target_time)

            if timing_error < 0.02:
                reward += 20.0
            elif timing_error < 0.05:
                reward += 10.0
        else:
            reward -= 10.0

            obs_dict = self._get_obs_dict()
            ball_pos = np.array(obs_dict["ball_pos"], dtype=np.float32).flatten()
            paddle_pos = np.array(obs_dict["paddle_pos"], dtype=np.float32).flatten()
            distance = float(np.linalg.norm(ball_pos - paddle_pos))

            if distance < 0.1:
                reward -= 5.0
            elif distance < 0.3:
                reward -= 10.0
            else:
                reward -= 20.0

        # Penalize invalid timing
        current_time = float(self._get_obs_dict()["time"])
        target_time = float(manager_action[0])

        if target_time < current_time:
            reward -= 30.0
        elif target_time > current_time + 1.0:
            reward -= 5.0 * (target_time - current_time - 1.0)

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
