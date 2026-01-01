from typing import Tuple, Dict, Optional
import numpy as np

from myosuite.utils import gym
from config import Config


class CurriculumEnv(gym.Env):
    """
    MyoSuite wrapper with:
      - default reward
      - minimal reward shaping (training only)
      - annealed shaping
      - SubprocVecEnv-safe curriculum variable
    """


    def __init__(
        self,
        config: Config,
        device: str = "cpu",

        # shaping weights
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

        eval_mode: bool = False,
    ):
        super().__init__()
        self.env = gym.make(config.env_id)

        # ---- curriculum variable ----
        self.curriculum_level: float = float(
            getattr(config, "curriculum_level", 1.0)
        )

        # shaping params
        self.w0_pelvis = w_pelvis
        self.w0_pelvis_vel = w_pelvis_vel
        self.w0_progress = w_progress
        self.w0_timing = w_timing

        self.anneal_steps = anneal_steps
        self.eval_mode = eval_mode

        self.ttc_threshold = ttc_threshold
        self.max_pelvis_speed = max_pelvis_speed
        self.max_paddle_speed = max_paddle_speed

        # state
        self._prev_pelvis_xy = None
        self._prev_time = None
        self._prev_paddle_ball_dist = None
        self._global_step = 0

    # -------------------------------------------------
    def _anneal_factor(self) -> float:
        if self.eval_mode:
            return 0.0
        return max(0.0, 1.0 - self._global_step / self.anneal_steps)

    # -------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        self._prev_pelvis_xy = None
        self._prev_time = None
        self._prev_paddle_ball_dist = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        info["is_success"] = bool(info.get("solved", False))

        obs_dict = info.get("obs_dict", None)
        if obs_dict is None:
            return obs, reward, terminated, truncated, info

        self._global_step += 1
        a = self._anneal_factor()
        if a <= 0.0:
            return obs, reward, terminated, truncated, info

        # -------------------------------------------------
        pelvis_xy = obs_dict["pelvis_pos"][:2]
        ball_pos = obs_dict["ball_pos"]
        ball_xy = ball_pos[:2]
        ball_vel = obs_dict["ball_vel"]

        paddle_pos = obs_dict["paddle_pos"]
        paddle_vel = obs_dict["paddle_vel"]
        t = float(obs_dict["time"])

        # pelvis velocity
        if self._prev_pelvis_xy is None:
            pelvis_vel_xy = np.zeros(2)
        else:
            dt = max(1e-6, t - self._prev_time)
            pelvis_vel_xy = (pelvis_xy - self._prev_pelvis_xy) / dt

        self._prev_pelvis_xy = pelvis_xy.copy()
        self._prev_time = t

        shaping_reward = 0.0

        # (1) pelvis–ball proximity
        pelvis_dist = np.linalg.norm(pelvis_xy - ball_xy)
        shaping_reward += a * self.w0_pelvis * np.exp(-pelvis_dist)

        # (2) pelvis velocity toward ball
        dir_pelvis = ball_xy - pelvis_xy
        dir_pelvis /= np.linalg.norm(dir_pelvis) + 1e-8
        pelvis_forward = np.clip(
            np.dot(pelvis_vel_xy, dir_pelvis), 0.0, self.max_pelvis_speed
        )
        shaping_reward += a * self.w0_pelvis_vel * pelvis_forward

        # (3) paddle–ball progress
        paddle_ball_dist = np.linalg.norm(paddle_pos - ball_pos)
        if self._prev_paddle_ball_dist is not None:
            progress = self._prev_paddle_ball_dist - paddle_ball_dist
            progress = np.clip(progress, -0.1, 0.1)
            shaping_reward += a * self.w0_progress * progress
        self._prev_paddle_ball_dist = paddle_ball_dist

        # (4) timing shaping
        ball_speed = np.linalg.norm(ball_vel) + 1e-6
        ttc = paddle_ball_dist / ball_speed

        if ttc < self.ttc_threshold:
            dir_paddle = ball_pos - paddle_pos
            dir_paddle /= np.linalg.norm(dir_paddle) + 1e-8
            paddle_forward = np.clip(
                np.dot(paddle_vel, dir_paddle),
                0.0,
                self.max_paddle_speed,
            )
            timing_weight = (self.ttc_threshold - ttc) / self.ttc_threshold
            shaping_reward += a * self.w0_timing * timing_weight * paddle_forward
            
        if info["is_success"]:
            reward += 0.2 * a

        reward += shaping_reward
        info["reward_shaping"] = shaping_reward
        info["anneal_factor"] = a
        info["curriculum_level"] = self.curriculum_level

        return obs, reward, terminated, truncated, info

    # -------------------------------------------------
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)
