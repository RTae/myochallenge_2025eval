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
        super().__init__(config)
        
        if isinstance(worker_env, tuple):
            worker_env = worker_env[0]
            
        if isinstance(worker_model, tuple):
            worker_model = worker_model[0]

        self.worker_env = worker_env
        self.worker_model = worker_model
        
        assert self.worker_env.num_envs == 1, (
            "TableTennisManager only supports num_envs=1."
        )
        
        if hasattr(self.worker_env, "env_method"):
            self.worker_env.env_method("set_goal_noise_scale", 0.0)
            self.worker_env.env_method("set_progress", 1.0)

        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------
        worker_instance = self.worker_env.envs[0].unwrapped
        if hasattr(worker_instance, "get_progress"):
            t_progress = worker_instance.get_progress()
            if t_progress < 0.95:
                logger.warning(f"WARNING: Manager initialized with Worker progress {t_progress:.2f}")

        self.decision_interval = int(decision_interval)
        self.max_episode_steps = int(max_episode_steps)

        # --------------------------------------------------
        # Observation space
        # --------------------------------------------------
        self.worker_obs_dim = np.prod(self.worker_env.observation_space.shape)
        self.observation_dim = self.worker_obs_dim + 1 

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        
        # --------------------------------------------------
        # Action space: 8-Dim Delta
        # [dx, dy, dz, dqw, dqx, dqy, dqz, ddt]
        # --------------------------------------------------
        self.action_space = gym.spaces.Box(
            low=-0.5, 
            high=0.5,
            shape=(7,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Runtime
        # --------------------------------------------------
        self.current_step = 0
        self._worker_obs = None
        self.success_buffer = deque(maxlen=int(success_buffer_len))
        
    @property
    def sim(self):
        return self.worker_env.envs[0].unwrapped.sim

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        self._worker_obs = self.worker_env.reset()
        self.current_step = 0
        self.success_buffer.clear()
        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        """
        Manager Step:
        1. Get physics prediction from Worker.
        2. Apply 7-Dim Delta (residual) from Manager policy.
        3. Normalize the resulting Normal Vector.
        4. Step Worker for 'decision_interval' steps.
        """
        # A. Access Worker internals
        worker_env_access = self.worker_env.envs[0].unwrapped
        obs_dict = worker_env_access.unwrapped.obs_dict
        
        # B. Predict Base Goal (Now returns 7-dim: [3-pos, 3-normal, 1-dt])
        goal_pred = worker_env_access.predict_goal_from_state(obs_dict)

        # C. Apply Manager's Learned Residual (7-dim)
        goal_delta = action.astype(np.float32)
        goal_final = goal_pred + goal_delta

        # D. RE-NORMALIZE the Normal Vector (Indices 3, 4, 5)
        # This ensures the Manager's nudge doesn't 'stretch' the normal
        norm_part = goal_final[3:6]
        norm_val = np.linalg.norm(norm_part)
        if norm_val > 1e-9:
            goal_final[3:6] = norm_part / norm_val
        else:
            # Fallback to standard 'Face' normal if vector collapses
            goal_final[3:6] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # E. Clip Time-to-Intercept (Index 6)
        goal_final[6] = np.clip(goal_final[6], 0.05, 2.0)
        
        # F. Inject into Worker
        worker_env_access.set_goal(goal_final, delta=goal_delta)
    
        accumulated_worker_reward = 0.0
        goal_success_any = False
        env_success_any = False
        last_infos = {}
        
        # G. Macro-Step: Run low-level for the decision interval
        for _ in range(self.decision_interval):
            # Predict with Worker Policy
            worker_actions, _ = self.worker_model.predict(
                self._worker_obs, deterministic=True
            )

            # Step Environment
            obs, rewards, dones, infos = self.worker_env.step(worker_actions)
            
            self._worker_obs = obs
            self.current_step += 1
            info = infos[0] 
            last_infos = info
            
            # Track if any single frame hit the success criteria
            goal_success_any |= bool(info.get("is_goal_success", 0.0))
            env_success_any  |= bool(info.get("is_success", 0.0))
            accumulated_worker_reward += rewards[0]
            
            if dones[0]:
                break

        # H. Compute Manager Reward
        self.success_buffer.append(1.0 if goal_success_any else 0.0)
        success_rate = float(np.mean(self.success_buffer)) if self.success_buffer else 0.0
        
        delta_norm = float(np.linalg.norm(goal_delta))

        reward = self._compute_reward(
            goal_success=goal_success_any,
            env_success=env_success_any,
            success_rate=success_rate,
            delta_norm=delta_norm,
            worker_reward=accumulated_worker_reward, 
        )

        # I. Build Output
        terminated = bool(env_success_any)
        truncated = bool(self.current_step >= self.max_episode_steps)
        obs_out = self._build_obs()

        info_out = {
            "goal_pred": goal_pred.copy(),
            "goal_delta": goal_delta.copy(),
            "goal_final": goal_final.copy(),
            "goal_delta_norm": delta_norm,
            "worker_reward": float(accumulated_worker_reward),
            **last_infos,
        }
        
        return obs_out, float(reward), terminated, truncated, info_out
    
    def _build_obs(self) -> np.ndarray:
        worker_obs = self._worker_obs[0].astype(np.float32)
        t_frac = np.array([self.current_step / self.max_episode_steps], dtype=np.float32)
        return np.concatenate([worker_obs, t_frac])

    def _compute_reward(
        self,
        *,
        goal_success: bool,
        env_success: bool,
        success_rate: float,
        delta_norm: float,
        worker_reward: float,
    ) -> float:
        r = -0.05
        if goal_success:
            r += 1.0
        if env_success:
            r += 10.0 
        r += 0.5 * success_rate
        r += 0.01 * worker_reward
        return float(r)