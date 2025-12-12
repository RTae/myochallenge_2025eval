# worker_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config
from hrl_utils import build_worker_obs, WorkerReward, HitDetector, PADDLE_FACE_RADIUS
from loguru import logger


class WorkerEnv(gym.Env):
    """Low-level worker (contact-conditioned)."""

    metadata = {"render_modes": []}

    def __init__(self, config: Config):
        super().__init__()
        from myosuite.utils import gym as myo_gym

        self.cfg = config
        self.base_env = myo_gym.make(config.env_id)

        # -----------------------------
        # Hit detection
        # -----------------------------
        self.hit_detector = HitDetector(dv_thr=1.5)

        # -----------------------------
        # Reward function
        # -----------------------------
        self.reward_fn = WorkerReward(
            hit_bonus=10.0,
            sweet_spot_bonus=2.0,
            approach_scale=0.2,
            inactivity_penalty=-0.05,
            energy_penalty_coef=0.001,
        )

        self.warmup_steps = 500_000

        self.goal = np.zeros(3, dtype=np.float32)
        self.t_in_macro = 0
        self.hit_count = 0
        self.step_count = 0

        # Infer obs shape
        self.base_env.reset()
        obs_dict = self.base_env.obs_dict
        self.goal = self._sample_goal()

        worker_obs = build_worker_obs(
            obs_dict, self.goal, self.t_in_macro, self.cfg
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=worker_obs.shape,
            dtype=np.float32,
        )

        self.action_space = self.base_env.action_space

    # --------------------------------------------------
    def _sample_goal(self):
        """Desired paddle velocity."""
        g = np.random.normal(
            loc=0.0,
            scale=self.cfg.goal_std,
            size=(3,),
        ).astype(np.float32)

        return np.clip(g, -self.cfg.goal_bound, self.cfg.goal_bound)

    # --------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed)
        obs_dict = info.get("obs_dict", {}) if info else self.base_env.obs_dict

        self.hit_detector.reset(obs_dict)
        self.reward_fn.reset()

        # Deterministic goal during evaluation
        if getattr(self.cfg, "eval_mode", False):
            self.goal = np.zeros(self.cfg.goal_dim, dtype=np.float32)
        else:
            self.goal = self._sample_goal()

        self.t_in_macro = 0
        self.hit_count = 0
        self.step_count = 0

        return build_worker_obs(
            obs_dict, self.goal, self.t_in_macro, self.cfg
        ), {}

    # --------------------------------------------------
    def step(self, action):
        obs, _, terminated, truncated, info = self.base_env.step(action)
        obs_dict = info.get("obs_dict", {}) if info else self.base_env.obs_dict

        self.t_in_macro += 1
        self.step_count += 1

        # --------------------------------------------------
        # Hit detection with proximity gating
        # --------------------------------------------------
        hit_raw, contact_force, dv = self.hit_detector.step(obs_dict)

        ball_pos = obs_dict["ball_pos"]
        paddle_pos = obs_dict["paddle_pos"]
        near_paddle = np.linalg.norm(ball_pos - paddle_pos) < 1.2 * PADDLE_FACE_RADIUS

        hit = hit_raw and near_paddle

        if hit:
            self.hit_count += 1
            self.hit_detector.reset(obs_dict)  # prevent double counting

            if self.step_count % 1000 == 0:
                logger.info(
                    f"[WORKER] HIT dv={dv:.2f} m/s, force={contact_force:.2f} N"
                )

        # --------------------------------------------------
        # Energy warmup
        # --------------------------------------------------
        warmup_factor = 0.2 if self.step_count < self.warmup_steps else 1.0
        original_energy_coef = self.reward_fn.energy_penalty_coef
        self.reward_fn.energy_penalty_coef = original_energy_coef * warmup_factor

        # --------------------------------------------------
        # Reward
        # --------------------------------------------------
        r_int, reward_components = self.reward_fn(
            obs_dict=obs_dict,
            hit=hit,
            goal=self.goal,
            external_dv=dv,
            external_contact_force=contact_force,
        )

        self.reward_fn.energy_penalty_coef = original_energy_coef

        # --------------------------------------------------
        # Macro-goal transition
        # --------------------------------------------------
        if self.t_in_macro >= self.cfg.high_level_period or terminated or truncated:
            if not getattr(self.cfg, "eval_mode", False):
                self.goal = self._sample_goal()
            self.t_in_macro = 0

        obs = build_worker_obs(
            obs_dict, self.goal, self.t_in_macro, self.cfg
        )

        info = info or {}
        info.update({
            "hit": hit,
            "hit_rate": self.hit_count / max(1, self.step_count),
            "contact_force": contact_force,
            "dv": dv,
            "intrinsic_reward": r_int,
            "energy_penalty_coef": original_energy_coef * warmup_factor,
        })
        info.update(reward_components)

        return obs, r_int, terminated, truncated, info
