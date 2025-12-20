from typing import Tuple, Dict, Optional, Any
import numpy as np
from myosuite.utils import gym as myo_gym

from utils import quat_to_paddle_normal


class TableTennisManager(myo_gym.Env):
    """
    Manager chooses a 6D goal for Worker: [dx, dy, dz, dpx, dpy, dt]
    Then runs frozen worker for K steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, worker_env: Any, worker_model: Any, decision_interval: int = 10, max_episode_steps: int = 800):
        super().__init__()
        self.worker_env = worker_env
        self.worker_model = worker_model

        self.decision_interval = int(decision_interval)
        self.max_episode_steps = int(max_episode_steps)

        # Manager obs: you can change dims; keep yours (25D)
        self.observation_dim = 25
        self.observation_space = myo_gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        # Manager action = worker goal (6D)
        self.action_dim = 6
        self.action_space = myo_gym.spaces.Box(
            low=self.worker_env.goal_low,
            high=self.worker_env.goal_high,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # counters
        self.current_step = 0
        self.current_decision = 0
        self.total_hits = 0

        # recurrent worker state (for RecurrentPPO)
        self.worker_state = None

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        self.worker_env.reset(seed=seed)
        self.current_step = 0
        self.current_decision = 0
        self.worker_state = None

        obs = self._augment_observation()
        info = {"is_success": False, "total_hits": int(self.total_hits)}
        return obs, info

    def step(self, manager_action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        manager_action = np.asarray(manager_action, dtype=np.float32)
        manager_action = np.clip(manager_action, self.worker_env.goal_low, self.worker_env.goal_high)

        # Apply goal to worker (anchors dt and ball pos inside worker)
        self.worker_env.set_goal(manager_action)

        worker_reward_total = 0.0
        hit_occurred = False
        contact_info: Dict = {}

        worker_terminated = False
        worker_truncated = False
        worker_info: Dict = {}

        # episode_start flag for recurrent worker
        episode_start = np.array([False])

        for _ in range(self.decision_interval):
            self.current_step += 1

            # get current worker obs from its wrapper
            worker_obs = self.worker_env._augment_observation()

            # Predict low-level action
            # Works for RecurrentPPO and PPO:
            # - RecurrentPPO uses (state, episode_start)
            # - PPO ignores them
            try:
                worker_action, self.worker_state = self.worker_model.predict(
                    worker_obs,
                    state=self.worker_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
            except TypeError:
                worker_action, _ = self.worker_model.predict(worker_obs, deterministic=True)

            # step worker env
            _, r, worker_terminated, worker_truncated, worker_info = self.worker_env.step(worker_action)
            worker_reward_total += float(r)

            # detect hit from base obs_dict
            obs_dict = self._get_obs_dict()
            touching = np.array(obs_dict.get("touching_info", np.zeros(6)), dtype=np.float32).flatten()
            if touching.size > 0 and touching[0] > 0.5 and not hit_occurred:
                hit_occurred = True
                contact_info = {
                    "hit_time": float(obs_dict["time"]),
                    "hit_position": np.array(obs_dict["paddle_pos"], dtype=np.float32).copy(),
                    "step": int(self.current_step),
                }
                self.total_hits += 1

            if worker_terminated or worker_truncated:
                episode_start = np.array([True])  # if model is recurrent, reset next call
                break

        self.current_decision += 1

        # Manager reward: reward hit + closeness to hitting at the desired time
        manager_reward = self._calculate_manager_reward(
            hit_occurred=hit_occurred,
            goal6=manager_action,
            contact_info=contact_info,
        )

        terminated = bool(worker_terminated)
        truncated = bool(worker_truncated or self.current_step >= self.max_episode_steps)

        next_obs = self._augment_observation() if (not terminated and not truncated) else np.zeros(self.observation_dim, dtype=np.float32)

        info = {
            "manager_step": int(self.current_step),
            "manager_decision": int(self.current_decision),
            "hit_occurred": bool(hit_occurred),
            "worker_reward_total": float(worker_reward_total),
            "is_success": bool(hit_occurred),  # or something stricter
            "total_hits": int(self.total_hits),
        }
        if hit_occurred:
            info.update({
                "hit_time": float(contact_info["hit_time"]),
                "hit_step": int(contact_info["step"]),
            })

        return next_obs, float(manager_reward), terminated, truncated, info

    # -------------------------
    # Observation
    # -------------------------
    def _get_obs_dict(self):
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
            ball_pos, ball_vel,
            paddle_pos, paddle_vel,
            paddle_ori,
            reach_err,
            touching_info,
            current_time,
        ]).astype(np.float32)

        assert manager_obs.shape[0] == self.observation_dim
        return manager_obs

    # -------------------------
    # Reward (goal6 = [dx,dy,dz,dpx,dpy,dt])
    # -------------------------
    def _calculate_manager_reward(self, hit_occurred: bool, goal6: np.ndarray, contact_info: Dict) -> float:
        obs = self._get_obs_dict()
        ball_pos = np.array(obs["ball_pos"], dtype=np.float32)
        paddle_pos = np.array(obs["paddle_pos"], dtype=np.float32)
        t = float(obs["time"])

        # Use goal semantics: dt is relative to goal start time stored in worker_env.goal_start_time
        assert self.worker_env.goal_start_time is not None
        target_time = float(self.worker_env.goal_start_time + goal6[5])

        # Dense shaping: encourage being close to ball & timing
        dist_ball_paddle = float(np.linalg.norm(ball_pos - paddle_pos))
        timing_error = float(abs(target_time - t))

        reward = 0.0
        reward += 5.0 * float(np.exp(-5.0 * dist_ball_paddle))
        reward += 2.0 * float(np.exp(-3.0 * timing_error))

        if hit_occurred:
            reward += 30.0
            hit_time = float(contact_info.get("hit_time", target_time))
            reward += 10.0 * float(np.exp(-10.0 * abs(hit_time - target_time)))

        return float(reward)

    def render(self):
        return self.worker_env.render()

    def close(self):
        return self.worker_env.close()

    def __getattr__(self, name):
        return getattr(self.worker_env, name)
