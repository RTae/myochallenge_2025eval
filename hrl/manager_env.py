from typing import Tuple, Dict, Any, Optional
import numpy as np
from myosuite.utils import gym
from stable_baselines3.common.vec_env import VecEnv

from config import Config
from custom_env import CustomEnv


class TableTennisManager(CustomEnv):
    """
    High-level manager for HRL.

    Controls a frozen worker policy by issuing 6D goals.
    Uses curriculum learning based on task success.
    """

    def __init__(
        self,
        worker_env: VecEnv,
        worker_model: Any,
        config: Config,
        decision_interval: int = 10,
        max_episode_steps: int = 800,
    ):
        super().__init__(config)

        self.worker_env = worker_env
        self.worker_model = worker_model
        self.decision_interval = decision_interval
        self.max_episode_steps = max_episode_steps

        # -------------------------------------------------
        # Observation: worker obs (18) + time proxy (1) = 19
        # -------------------------------------------------
        self.worker_obs_dim = 18
        self.observation_dim = self.worker_obs_dim + 1

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Action = goal6 (same bounds as worker)
        # -------------------------------------------------
        base_worker = self.worker_env.envs[0]
        self.goal_low = np.asarray(base_worker.goal_low, np.float32)
        self.goal_high = np.asarray(base_worker.goal_high, np.float32)

        self.action_space = gym.spaces.Box(
            low=self.goal_low,
            high=self.goal_high,
            shape=(6,),
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Runtime state
        # -------------------------------------------------
        self.current_step = 0
        self._worker_obs = None

        # -------------------------------------------------
        # Curriculum
        # -------------------------------------------------
        self.curriculum_stage = 0  # 0 → hit, 1 → hit+success, 2 → success
        self.success_window = []
        self.window_size = 50
        self.advance_threshold = 0.7
        
    @property
    def sim(self):
        """
        Expose worker sim so generic callbacks can render.
        """
        return self.worker_env.envs[0].sim

    # ============================================================
    # Reset
    # ============================================================
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        self._worker_obs = self.worker_env.reset()
        self.current_step = 0
        self.total_hits = 0

        return (
            self._build_manager_obs(self._worker_obs),
            {"is_success": False},
        )

    # ============================================================
    # Step
    # ============================================================
    def step(self, action: np.ndarray):
        # -------------------------------------------------
        # 1) Set new goal on worker
        # -------------------------------------------------
        goal = np.clip(action, self.goal_low, self.goal_high).astype(np.float32)
        self.worker_env.env_method("set_goal", goal, indices=0)

        paddle_hit = False
        is_success = False
        terminated = False
        truncated = False

        # -------------------------------------------------
        # 2) Run worker for decision_interval steps
        # -------------------------------------------------
        for _ in range(self.decision_interval):
            obs_1d = self._worker_obs[0]
            worker_action, _ = self.worker_model.predict(obs_1d, deterministic=True)

            obs, _, dones, infos = self.worker_env.step([worker_action])
            self._worker_obs = obs
            self.current_step += 1

            info0 = infos[0]

            paddle_hit |= bool(info0.get("hit", False))
            is_success |= bool(info0.get("is_success", False))

            if bool(dones[0]):
                terminated = True
                break

            if self.current_step >= self.max_episode_steps:
                truncated = True
                break

        # -------------------------------------------------
        # 3) Curriculum update
        # -------------------------------------------------
        self._update_curriculum(
            paddle_hit=paddle_hit,
            is_success=is_success,
        )

        # -------------------------------------------------
        # 4) Reward
        # -------------------------------------------------
        reward = self._calculate_reward(
            paddle_hit=paddle_hit,
            is_success=is_success,
        )

        obs_out = (
            self._build_manager_obs(self._worker_obs)
            if not (terminated or truncated)
            else np.zeros(self.observation_dim, dtype=np.float32)
        )

        info = {
            "is_success": is_success,
            "is_paddle_hit": paddle_hit,
            "curriculum_stage": self.curriculum_stage,
        }

        return obs_out, float(reward), terminated, truncated, info

    # ============================================================
    # Observation builder
    # ============================================================
    def _build_manager_obs(self, worker_obs_vec: np.ndarray) -> np.ndarray:
        """
        Worker obs shape: (1, 18)
        Manager obs: [worker_obs (18), time_proxy (1)] = 19 dims
        """
        w = np.asarray(worker_obs_vec[0], dtype=np.float32)

        # Safety check
        if w.shape[0] != 18:
            ww = np.zeros(18, dtype=np.float32)
            ww[: min(18, w.shape[0])] = w[: min(18, w.shape[0])]
            w = ww

        # Normalized time proxy
        t = np.array(
            [self.current_step / max(1, self.max_episode_steps)],
            dtype=np.float32,
        )

        out = np.concatenate([w, t], axis=0)

        assert out.shape == (self.observation_dim,), f"Manager obs shape mismatch: {out.shape}"
        return out

    # ============================================================
    # Curriculum logic
    # ============================================================
    def _update_curriculum(self, paddle_hit: bool, is_success: bool) -> None:
        if self.curriculum_stage == 0:
            signal = paddle_hit
        elif self.curriculum_stage == 1:
            signal = paddle_hit or is_success
        else:
            signal = is_success

        self.success_window.append(int(signal))
        if len(self.success_window) > self.window_size:
            self.success_window.pop(0)

        if (
            len(self.success_window) == self.window_size
            and np.mean(self.success_window) >= self.advance_threshold
            and self.curriculum_stage < 2
        ):
            self.curriculum_stage += 1
            self.success_window.clear()

    # ============================================================
    # Reward
    # ============================================================
    def _calculate_reward(self, paddle_hit: bool, is_success: bool) -> float:
        reward = -1.0  # living penalty

        if self.curriculum_stage == 0:
            if paddle_hit:
                reward += 5.0

        elif self.curriculum_stage == 1:
            if paddle_hit:
                reward += 2.0
            if is_success:
                reward += 30.0

        else:  # stage 2
            if is_success:
                reward += 100.0

        return reward