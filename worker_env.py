# worker_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config
from hrl_utils import build_worker_obs, WorkerReward, HitDetector
from loguru import logger


class WorkerEnv(gym.Env):
    """Low-level worker (contact-conditioned) with automatic curriculum."""

    metadata = {"render_modes": []}

    def __init__(self, config: Config):
        super().__init__()
        from myosuite.utils import gym as myo_gym

        self.cfg = config
        self.base_env = myo_gym.make(config.env_id)

        # -----------------------------
        # Curriculum state
        # -----------------------------
        self.curriculum_stage = 0
        self.curriculum_window = 200
        self.curriculum_stats = {
            "hit": [],
            "dv": [],
        }

        # -----------------------------
        # Hit detection (initially easy)
        # -----------------------------
        self.hit_detector = HitDetector(dv_thr=0.5)

        # -----------------------------
        # Reward function
        # -----------------------------
        self.reward_fn = WorkerReward(
            hit_bonus=6.0,
            sweet_spot_bonus=2.0,
            approach_scale=0.2,
            inactivity_penalty=-0.05,
            energy_penalty_coef=0.001,
        )

        self.warmup_steps = 500_000

        # -----------------------------
        # Episode horizon
        # -----------------------------
        self.max_worker_steps = self.cfg.worker_episode_len
        self.worker_step_count = 0

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

        self.worker_step_count = 0
        self.t_in_macro = 0
        self.hit_count = 0
        self.step_count = 0

        self.curriculum_stats = {"hit": [], "dv": []}

        if getattr(self.cfg, "eval_mode", False):
            self.goal = np.zeros(self.cfg.goal_dim, dtype=np.float32)
        else:
            self.goal = self._sample_goal()

        return build_worker_obs(
            obs_dict, self.goal, self.t_in_macro, self.cfg
        ), {}

    # --------------------------------------------------
    def _update_curriculum(self):
        if len(self.curriculum_stats["hit"]) < self.curriculum_window:
            return

        hit_rate = np.mean(self.curriculum_stats["hit"])
        dv_mean = np.mean(self.curriculum_stats["dv"])

        # Stage 0 → 1
        if self.curriculum_stage == 0 and hit_rate > 0.02:
            self.curriculum_stage = 1
            logger.info("[CURRICULUM] → Stage 1 (commitment)")
            self.hit_detector.dv_thr = 1.0
            self.reward_fn.hit_bonus = 10.0

        # Stage 1 → 2
        elif self.curriculum_stage == 1 and hit_rate > 0.05 and dv_mean > 0.8:
            self.curriculum_stage = 2
            logger.info("[CURRICULUM] → Stage 2 (strong hits)")
            self.hit_detector.dv_thr = 1.5
            self.reward_fn.hit_bonus = 15.0

    # --------------------------------------------------
    def step(self, action):
        obs, _, terminated, truncated, info = self.base_env.step(action)
        obs_dict = info.get("obs_dict", {}) if info else self.base_env.obs_dict

        self.t_in_macro += 1
        self.step_count += 1
        self.worker_step_count += 1

        # -----------------------------
        # Hit detection
        # -----------------------------
        hit, contact_force, dv = self.hit_detector.step(obs_dict)

        self.curriculum_stats["hit"].append(float(hit))
        self.curriculum_stats["dv"].append(dv)

        if len(self.curriculum_stats["hit"]) > self.curriculum_window:
            self.curriculum_stats["hit"].pop(0)
            self.curriculum_stats["dv"].pop(0)

        self._update_curriculum()

        if hit:
            self.hit_count += 1
            if self.step_count % 1000 == 0:
                logger.info(
                    f"[WORKER] HIT dv={dv:.2f} m/s, force={contact_force:.2f} N"
                )

        # -----------------------------
        # Energy warmup
        # -----------------------------
        warmup_factor = 0.2 if self.step_count < self.warmup_steps else 1.0
        original_energy_coef = self.reward_fn.energy_penalty_coef
        self.reward_fn.energy_penalty_coef = original_energy_coef * warmup_factor

        # -----------------------------
        # Reward
        # -----------------------------
        r_int, reward_components = self.reward_fn(
            obs_dict=obs_dict,
            hit=hit,
            goal=self.goal,
            external_dv=dv,
            external_contact_force=contact_force,
        )

        self.reward_fn.energy_penalty_coef = original_energy_coef

        # -----------------------------
        # Macro-goal transition
        # -----------------------------
        if self.t_in_macro >= self.cfg.high_level_period or terminated or truncated:
            if not getattr(self.cfg, "eval_mode", False):
                self.goal = self._sample_goal()
            self.t_in_macro = 0

        # -----------------------------
        # Force episode end
        # -----------------------------
        if self.worker_step_count >= self.max_worker_steps:
            truncated = True

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
            "curriculum_stage": self.curriculum_stage,
        })
        info.update(reward_components)

        return obs, r_int, terminated, truncated, info
