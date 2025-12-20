# hrl/manager_env.py
from typing import Tuple, Dict, Optional, Any
from collections import deque
import numpy as np
from myosuite.utils import gym
from stable_baselines3.common.vec_env import VecEnv

from config import Config
from custom_env import CustomEnv


class TableTennisManager(CustomEnv):
    """
    High-level HRL manager.

    - Observes worker obs (21) + time proxy (1) = 22
    - Action is a 6D goal for worker
    - Executes frozen worker for decision_interval steps per manager step
    - Reward is SMOOTHED (low variance) using moving average of success
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
        self.decision_interval = int(decision_interval)
        self.max_episode_steps = int(max_episode_steps)

        self.worker_obs_dim = 18
        self.observation_dim = 19

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # Goal bounds from worker
        base_worker = self.worker_env.envs[0]
        self.goal_low = np.asarray(base_worker.goal_low, dtype=np.float32).reshape(6,)
        self.goal_high = np.asarray(base_worker.goal_high, dtype=np.float32).reshape(6,)

        self.action_space = gym.spaces.Box(
            low=self.goal_low,
            high=self.goal_high,
            shape=(6,),
            dtype=np.float32,
        )

        self.current_step = 0
        self._worker_obs = None

        # ---- reward smoothing ----
        self.success_buffer = deque(maxlen=20)

    @property
    def sim(self):
        # so your generic video callback can do env.sim.renderer.render_offscreen(...)
        return self.worker_env.envs[0].sim

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        self._worker_obs = self.worker_env.reset()
        self.current_step = 0
        self.success_buffer.clear()
        obs = self._build_obs()
        return obs, {"is_success": False, "is_paddle_hit": False}

    def step(self, action: np.ndarray):
        goal = np.clip(np.asarray(action, dtype=np.float32).reshape(6,), self.goal_low, self.goal_high)
        self.worker_env.env_method("set_goal", goal, indices=0)

        hit = False
        success = False

        for _ in range(self.decision_interval):
            obs_1d = np.asarray(self._worker_obs[0], dtype=np.float32)

            worker_action, _ = self.worker_model.predict(obs_1d, deterministic=True)

            obs, _, dones, infos = self.worker_env.step([worker_action])
            self._worker_obs = obs
            self.current_step += 1

            info = infos[0]
            hit |= bool(info.get("is_paddle_hit", False))

            # IMPORTANT: success is from worker's env success (solved)
            success |= bool(info.get("is_success", False))

            if bool(dones[0]):
                break
            if self.current_step >= self.max_episode_steps:
                break

        # ---- reward smoothing ----
        self.success_buffer.append(1.0 if success else 0.0)
        reward = self._reward_smoothed(hit, success)

        return self._build_obs(), float(reward), False, False, {
            "is_success": success,
            "is_paddle_hit": hit,
        }

    def _build_obs(self) -> np.ndarray:
        worker_obs = np.asarray(self._worker_obs[0], dtype=np.float32).reshape(18,)

        t = np.array(
            [self.current_step / max(1, self.max_episode_steps)],
            dtype=np.float32,
        ).reshape(1,)

        obs = np.hstack([worker_obs, t])
        assert obs.shape == (self.observation_dim,), f"obs.shape={obs.shape}"
        return obs

    def _reward_smoothed(self, hit: bool, success: bool) -> float:
        """
        Low-variance + smoothed reward:
        - small living cost
        - small hit bonus
        - terminal success bonus
        - PLUS smooth success-rate shaping
        """
        success_rate = float(np.mean(self.success_buffer)) if len(self.success_buffer) > 0 else 0.0

        r = -0.1                # living cost
        r += 0.5 * success_rate # smooth dense signal
        if hit:
            r += 0.5            # small immediate encouragement
        if success:
            r += 5.0            # terminal bonus

        return float(r)