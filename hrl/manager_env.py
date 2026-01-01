from typing import Tuple, Dict, Optional, Any
from collections import deque
import numpy as np

import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

from config import Config
from custom_env import CustomEnv
from loguru import logger


class TableTennisManager(CustomEnv):
    """
    High-level HRL manager (Residual / Delta-Control).
    
    The Manager observes the worker's state and outputs a correction (Delta)
    to the physics-based predicted goal.
    """

    def __init__(
        self,
        worker_env: VecEnv,
        worker_model: Any,
        config: Config,
        decision_interval: int = 10,
        max_episode_steps: int = 800,
        success_buffer_len: int = 20,
    ):
        # This initializes the unused base env, which is fine (ignored)
        super().__init__(config)

        self.worker_env = worker_env
        self.worker_model = worker_model

        # --------------------------------------------------
        # CRITICAL: Enforce Single Environment
        # --------------------------------------------------
        assert self.worker_env.num_envs == 1, (
            "TableTennisManager only supports num_envs=1 because it calculates "
            "a single goal correction based on the first environment."
        )
        

        # --------------------------------------------------
        # Safety check: worker must be fully trained
        # --------------------------------------------------
        worker_instance = self.worker_env.envs[0].unwrapped
        
        if hasattr(worker_instance, "get_progress"):
            t_progress = worker_instance.get_progress()
            # We allow slightly less than 1.0 just in case, but warn if low
            if t_progress < 0.95:
                logger.warning(f"WARNING: Manager initialized with Worker progress {t_progress:.2f}")

        self.decision_interval = int(decision_interval)
        self.max_episode_steps = int(max_episode_steps)

        # --------------------------------------------------
        # Observation space
        # --------------------------------------------------
        self.worker_obs_dim = np.prod(self.worker_env.observation_space.shape)
        self.observation_dim = self.worker_obs_dim + 1 # +1 for time

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        
        # --------------------------------------------------
        # Action space: RAW physical Δ-goal (Residual)
        # [dx, dy, dz, dnx, dny, ddt]
        # --------------------------------------------------
        self.action_space = gym.spaces.Box(
            low=-0.5, 
            high=0.5,
            shape=(6,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Runtime
        # --------------------------------------------------
        self.current_step = 0
        self._worker_obs = None
        self.success_buffer = deque(maxlen=int(success_buffer_len))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        # Reset the worker VecEnv (bypassing self.env)
        self._worker_obs = self.worker_env.reset()
        
        self.current_step = 0
        self.success_buffer.clear()

        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        # --------------------------------------------------
        # 1) Predict Base Goal (Heuristic)
        # --------------------------------------------------
        worker_env_access = self.worker_env.envs[0].unwrapped
        obs_dict = worker_env_access.unwrapped.obs_dict
        
        # Use the Worker's physics logic to guess where we SHOULD hit
        goal_pred = worker_env_access.predict_goal_from_state(obs_dict)

        # --------------------------------------------------
        # 2) Apply Learned Residual
        # --------------------------------------------------
        goal_delta = action.astype(np.float32)
        
        # Clamp delta to prevent impossible goals
        goal_delta = np.clip(goal_delta, -0.3, 0.3)

        goal_final = goal_pred + goal_delta

        # Set this goal for the worker
        worker_env_access.set_goal(goal_final)

        # --------------------------------------------------
        # 3) Execute Temporal Abstraction (Frozen Worker)
        # --------------------------------------------------
        goal_success_any = False
        env_success_any = False
        last_infos = {}
        
        for _ in range(self.decision_interval):
            # Predict deterministic action from frozen worker
            worker_actions, _ = self.worker_model.predict(
                self._worker_obs, deterministic=True
            )

            # Step the worker
            obs, rewards, dones, infos = self.worker_env.step(worker_actions)
            
            self._worker_obs = obs
            self.current_step += 1
            
            info = infos[0] 
            last_infos = info
            
            # Check success (Manager relies on Worker/BaseEnv to provide these keys)
            goal_success_any |= bool(info.get("is_soft_goal_success", 0.0))
            env_success_any  |= bool(info.get("is_success", 0.0))
            
            if dones[0]:
                break

        # --------------------------------------------------
        # 4) Compute Manager Reward
        # --------------------------------------------------
        self.success_buffer.append(1.0 if env_success_any else 0.0)
        success_rate = float(np.mean(self.success_buffer))
        delta_norm = float(np.linalg.norm(goal_delta))

        reward = self._compute_reward(
            goal_success=goal_success_any,
            env_success=env_success_any,
            success_rate=success_rate,
            delta_norm=delta_norm,
        )

        # --------------------------------------------------
        # 5) Build Output
        # --------------------------------------------------
        terminated = bool(env_success_any)
        truncated = bool(self.current_step >= self.max_episode_steps)

        obs_out = self._build_obs()

        info_out = {
            "goal_pred": goal_pred.copy(),
            "goal_delta": goal_delta.copy(),
            "goal_final": goal_final.copy(),
            "goal_delta_norm": delta_norm,
            **last_infos,
        }
        
        if obs_out.shape != self.observation_space.shape:
            raise ValueError(
                f"OBS SHAPE ERROR: got {obs_out.shape}, expected {self.observation_space.shape}"
            )

        return obs_out, float(reward), terminated, truncated, info_out

    def _build_obs(self) -> np.ndarray:
        worker_obs = self._worker_obs[0].astype(np.float32)

        t_frac = np.array(
            [self.current_step / self.max_episode_steps],
            dtype=np.float32
        )

        return np.concatenate([worker_obs, t_frac])

    def _compute_reward(
        self,
        *,
        goal_success: bool,
        env_success: bool,
        success_rate: float,
        delta_norm: float,
    ) -> float:
        # Base existence cost
        r = -0.05
        
        # Regularization: Penalize large deviations from physics
        r -= 0.05 * delta_norm 

        # Tier 1: Worker reached the physical target
        if goal_success:
            r += 1.0

        # Tier 2: Task Solved (Ball hit table)
        if env_success:
            r += 10.0 

        # Long-term consistency bonus
        r += 0.5 * success_rate

        return float(r)