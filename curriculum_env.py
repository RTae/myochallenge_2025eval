from typing import Tuple, Dict, Optional
import numpy as np

from myosuite.utils import gym
from config import Config


class CurriculumEnv(gym.Env):
    """
    MyoSuite wrapper with:
      - default reward
      - minimal reward shaping (training only)
      - automatic annealing
      - shaping fully disabled in eval mode

    Uses ONLY keys present in info['obs_dict'].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Config,
        device: str = "cpu",

        # initial shaping weights
        w_pelvis: float = 0.03,
        w_pelvis_vel: float = 0.01,
        w_progress: float = 0.05,
        w_timing: float = 0.03,

        # annealing
        anneal_steps: int = 5_000_000,

        # timing
        ttc_threshold: float = 0.6,

        # clamps
        max_pelvis_speed: float = 1.5,
        max_paddle_speed: float = 3.0,

        # evaluation switch
        eval_mode: bool = False,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.env = gym.make(config.env_id)

        # initial weights
        self.w0_pelvis = w_pelvis
        self.w0_pelvis_vel = w_pelvis_vel
        self.w0_progress = w_progress
        self.w0_timing = w_timing

        self.anneal_steps = anneal_steps
        self.eval_mode = eval_mode

        self.ttc_threshold = ttc_threshold
        self.max_pelvis_speed = max_pelvis_speed
        self.max_paddle_speed = max_paddle_speed

        # state for finite differences
        self._prev_pelvis_xy = None
        self._prev_time = None
        self._prev_paddle_ball_dist = None
        self._global_step = 0

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------
    def _anneal_factor(self) -> float:
        if self.eval_mode:
            return 0.0
        return max(0.0, 1.0 - self._global_step / self.anneal_steps)

    # -------------------------------------------------
    # Gym API
    # -------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed)

        self._prev_pelvis_xy = None
        self._prev_time = None
        self._prev_paddle_ball_dist = None

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        info["is_success"] = info.get("solved")

        self._global_step += 1
        a = self._anneal_factor()

        if a > 0.0:
            obs_dict = info["obs_dict"]

            pelvis_xy = obs_dict["pelvis_pos"][:2]
            ball_pos = obs_dict["ball_pos"]
            ball_xy = ball_pos[:2]
            ball_vel = obs_dict["ball_vel"]

            paddle_pos = obs_dict["paddle_pos"]
            paddle_vel = obs_dict["paddle_vel"]

            t = float(obs_dict["time"])

            # -------------------------------------------------
            # Pelvis velocity via finite difference
            # -------------------------------------------------
            if self._prev_pelvis_xy is None:
                pelvis_vel_xy = np.zeros(2, dtype=np.float32)
            else:
                dt = max(1e-6, t - self._prev_time)
                pelvis_vel_xy = (pelvis_xy - self._prev_pelvis_xy) / dt

            self._prev_pelvis_xy = pelvis_xy.copy()
            self._prev_time = t

            shaping_reward = 0.0

            # =================================================
            # (1) Pelvis–ball proximity
            # =================================================
            pelvis_dist = np.linalg.norm(pelvis_xy - ball_xy)
            shaping_reward += a * self.w0_pelvis * np.exp(-pelvis_dist)

            # =================================================
            # (2) Pelvis velocity toward ball
            # =================================================
            dir_pelvis = ball_xy - pelvis_xy
            dir_pelvis /= np.linalg.norm(dir_pelvis) + 1e-8

            pelvis_forward = np.dot(pelvis_vel_xy, dir_pelvis)
            pelvis_forward = np.clip(
                pelvis_forward, 0.0, self.max_pelvis_speed
            )

            shaping_reward += a * self.w0_pelvis_vel * pelvis_forward

            # =================================================
            # (3) Paddle–ball progress
            # =================================================
            paddle_ball_dist = np.linalg.norm(paddle_pos - ball_pos)

            if self._prev_paddle_ball_dist is not None:
                progress = self._prev_paddle_ball_dist - paddle_ball_dist
                shaping_reward += a * self.w0_progress * progress

            self._prev_paddle_ball_dist = paddle_ball_dist

            # =================================================
            # (4) Timing-aware shaping
            # =================================================
            ball_speed = np.linalg.norm(ball_vel) + 1e-6
            ttc = paddle_ball_dist / ball_speed

            if ttc < self.ttc_threshold:
                dir_paddle = ball_pos - paddle_pos
                dir_paddle /= np.linalg.norm(dir_paddle) + 1e-8

                paddle_forward = np.dot(paddle_vel, dir_paddle)
                paddle_forward = np.clip(
                    paddle_forward, 0.0, self.max_paddle_speed
                )

                timing_weight = (self.ttc_threshold - ttc) / self.ttc_threshold
                shaping_reward += a * self.w0_timing * timing_weight * paddle_forward

            reward += shaping_reward
            info["reward_shaping"] = shaping_reward
            info["anneal_factor"] = a

        return obs, reward, terminated, truncated, info

    # -------------------------------------------------
    # Passthrough
    # -------------------------------------------------
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env

    def __getattr__(self, name):
        return getattr(self.env, name)
