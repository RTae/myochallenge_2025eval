from typing import Tuple, Dict, Optional
import numpy as np

from myosuite.utils import gym
from config import Config


class CurriculumEnv(gym.Env):
    """
    MyoSuite environment with curriculum on:
    - ball initial position (XY)
    - ball initial velocity (XYZ)

    Curriculum ONLY affects reset():
    - reward unchanged
    - termination unchanged
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device
        self.env = gym.make(config.env_id)

        self.np_random = np.random.default_rng()

        # Cache indices
        self._ball_qpos_id = self._find_ball_qpos_id()
        self._ball_qvel_id = self._find_ball_qvel_id()

        self._base_ball_qpos = None
        self._base_ball_qvel = None

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _find_ball_qpos_id(self) -> slice:
        model = self.env.unwrapped.sim.model
        for j in range(model.njnt):
            name = model.joint_id2name(j)
            if name and "ball" in name.lower():
                adr = model.jnt_qposadr[j]
                dim = model.jnt_dofnum[j]
                return slice(adr, adr + dim)
        raise RuntimeError("❌ Ball qpos not found")

    def _find_ball_qvel_id(self) -> slice:
        model = self.env.unwrapped.sim.model
        for j in range(model.njnt):
            name = model.joint_id2name(j)
            if name and "ball" in name.lower():
                adr = model.jnt_dofadr[j]
                dim = model.jnt_dofnum[j]
                return slice(adr, adr + dim)
        raise RuntimeError("❌ Ball qvel not found")

    # -------------------------------------------------
    # Gym API
    # -------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Curriculum level
        level = float(getattr(self.config, "curriculum_level", 1.0))
        level = np.clip(level, 0.0, 1.0)

        sim = self.env.unwrapped.sim
        qpos = sim.data.qpos
        qvel = sim.data.qvel

        # Cache canonical reset state
        if self._base_ball_qpos is None:
            self._base_ball_qpos = qpos[self._ball_qpos_id].copy()
        if self._base_ball_qvel is None:
            self._base_ball_qvel = qvel[self._ball_qvel_id].copy()

        # ---------------- Stage 1: Ball position ----------------
        # XY offset: ±0.05 → ±0.50
        max_xy_offset = 0.05 + 0.45 * level
        offset_xy = self.np_random.uniform(
            low=-max_xy_offset,
            high=max_xy_offset,
            size=2
        )

        qpos[self._ball_qpos_id][0:2] = (
            self._base_ball_qpos[0:2] + offset_xy
        )

        # ---------------- Stage 2: Ball speed ----------------
        # Speed magnitude: small → full
        # Early: ~0.2 m/s, Late: ~2.5 m/s
        max_speed = 0.2 + 2.3 * level

        # Random direction (mostly towards paddle)
        direction = np.array([
            -1.0,                              # towards player
            self.np_random.uniform(-0.3, 0.3),
            self.np_random.uniform(0.2, 0.6)
        ])
        direction /= np.linalg.norm(direction) + 1e-8

        speed = self.np_random.uniform(0.1, max_speed)
        qvel[self._ball_qvel_id][:3] = direction * speed

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        info["is_success"] = bool(info.get("solved", False))

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
