from typing import Tuple, Dict, Optional
import numpy as np

from myosuite.utils import gym
from config import Config


class CurriculumEnv(gym.Env):
    """
    Plain MyoSuite environment with curriculum on ball initialization.
    Curriculum ONLY affects reset(), not reward or termination.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device
        self.env = gym.make(config.env_id)

        # Cache ball joint index once
        self._ball_qpos_id = self._find_ball_qpos_id()

    # -------------------------------------------------
    # Curriculum helper
    # -------------------------------------------------
    def _find_ball_qpos_id(self) -> slice:
        """
        Find the qpos slice corresponding to the ball joint.
        This is robust to different MyoSuite XMLs.
        """
        model = self.env.unwrapped.sim.model

        for j in range(model.njnt):
            name = model.joint_id2name(j)
            if name is not None and "ball" in name.lower():
                qpos_adr = model.jnt_qposadr[j]
                qpos_dim = model.jnt_dofnum[j]
                return slice(qpos_adr, qpos_adr + qpos_dim)

        raise RuntimeError("❌ Could not find ball joint in the model.")

    # -------------------------------------------------
    # Gym API
    # -------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed)

        # ---------------- Curriculum logic ----------------
        level = float(getattr(self.config, "curriculum_level", 1.0))
        level = np.clip(level, 0.0, 1.0)

        # XY offset range grows with curriculum
        # early: ±0.05 m, later: ±0.5 m
        max_offset = 0.05 + 0.45 * level

        offset_xy = np.random.uniform(
            low=-max_offset,
            high=max_offset,
            size=2
        )

        # Apply offset to ball qpos (x, y)
        qpos = self.env.unwrapped.sim.data.qpos
        qpos[self._ball_qpos_id][0:2] += offset_xy

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        info.update({
            "is_success": bool(info.get("solved", False)),
        })

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env

    def __getattr__(self, name):
        return getattr(self.env, name)
