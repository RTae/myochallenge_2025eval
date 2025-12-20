from typing import Tuple, Dict, Optional, Any
import numpy as np
from myosuite.utils import gym

from config import Config
from custom_env import CustomEnv
from utils import quat_to_paddle_normal

class TableTennisWorker(CustomEnv):
    """
    Worker that incorporates reach_err into the observation.
    - Observation: rel paddle, paddle_vel, paddle_normal, pelvis rel, ball_vel, reach_err, time + goal6 => (21 dims).
    """

    def __init__(self, config: Config, training_stage: int = 0):
        super().__init__(config)

        # -----------------------------------------
        # Bound for goals (manager action space)
        # -----------------------------------------
        self.goal_low = np.array([-0.6, -0.6, -0.4, -0.8, -0.5, 0.15], np.float32)
        self.goal_high = np.array([ 0.6,  0.6,  0.6,  0.8,  0.5, 0.8], np.float32)
        self.goal_dim = 6

        # -----------------------------------------
        # Observation dims:
        #   rel paddle pos (3)
        #   paddle vel    (3)
        #   paddle normal (3)
        #   pelvis rel (2)
        #   ball vel      (3)
        #   reach_err     (3)
        #   time          (1)
        # = 18 for state
        # + 6 for goal = 24
        self.state_dim = 18
        self.observation_dim = self.state_dim + self.goal_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # HRL internals
        self.current_goal: Optional[np.ndarray] = None
        self.goal_start_time: Optional[float] = None
        self.goal_start_ball_pos: Optional[np.ndarray] = None
        self._prev_paddle_contact: bool = False

    def reset_hrl_state(self):
        self.current_goal = None
        self.goal_start_time = None
        self.goal_start_ball_pos = None
        self._prev_paddle_contact = False

    def set_goal(self, goal6: np.ndarray):
        goal6 = np.asarray(goal6, dtype=np.float32).reshape(-1)
        assert goal6.shape == (6,)
        obs = self.env.unwrapped.obs_dict
        self.current_goal = goal6
        self.goal_start_time = float(obs["time"])
        self.goal_start_ball_pos = np.asarray(obs["ball_pos"], dtype=np.float32).copy()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = super().reset(seed=seed)
        self.reset_hrl_state()

        # Standalone, sample initial goal
        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        return self._augment_obs(), info

    def step(self, action: np.ndarray):
        obs, base_reward, terminated, truncated, info = super().step(action)

        # Goal-conditioned reward can be added if desired
        # (optional — manager will handle higher level)
        total_reward = float(base_reward)

        # Add HRL signals
        touching = np.asarray(self.env.unwrapped.obs_dict.get("touching_info", []), np.float32)
        ball_paddle = touching[0] if touching.size > 0 else 0.0
        paddle_contact = bool(ball_paddle > 0.5)

        hit = paddle_contact and not self._prev_paddle_contact
        self._prev_paddle_contact = paddle_contact

        info.update({
            "hit": bool(hit),
            "is_success": bool(info.get("is_success", False)),
        })

        if terminated or truncated:
            self.reset_hrl_state()

        return self._augment_obs(), total_reward, terminated, truncated, info

    def _augment_obs(self) -> np.ndarray:
        obs = self.env.unwrapped.obs_dict

        # ensure goal exists
        if self.current_goal is None:
            self.set_goal(self._sample_goal())

        ball_pos = np.asarray(obs["ball_pos"], np.float32)
        ball_vel = np.asarray(obs["ball_vel"], np.float32)
        paddle_pos = np.asarray(obs["paddle_pos"], np.float32)
        paddle_vel = np.asarray(obs["paddle_vel"], np.float32)
        paddle_n = quat_to_paddle_normal(np.asarray(obs["paddle_ori"], np.float32))
        pelvis_xy = np.asarray(obs["pelvis_pos"][:2], np.float32)

        reach_err = np.asarray(obs["reach_err"], np.float32)  # 3 dims

        t = np.array([float(obs["time"])], np.float32)

        rel_pad = paddle_pos - ball_pos
        rel_pelvis = pelvis_xy - ball_pos[:2]

        state = np.hstack([
            rel_pad,          # 3
            paddle_vel,       # 3
            paddle_n,         # 3
            rel_pelvis,       # 2
            ball_vel,         # 3
            reach_err,        # 3
            t,                # 1
        ]).astype(np.float32)  # 18 dims

        return np.hstack([state, self.current_goal]).astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        return (
            self.goal_low + (self.goal_high - self.goal_low) * np.random.rand(self.goal_dim)
        ).astype(np.float32)