from typing import Tuple, Dict, Optional
import numpy as np

import mujoco
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

        # Cache ball joint indices
        self._ball_body_id = self._find_pingpong_body_id()
        self._ball_qpos_id, self._ball_qvel_id = self._get_free_joint_slices_for_body(self._ball_body_id)

        # Canonical reset cache
        self._base_ball_qpos = None

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _find_pingpong_body_id(self) -> int:
        mj_model = self.env.unwrapped.sim.model.ptr

        for b in range(mj_model.nbody):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b)
            if name is not None and name.lower() == "pingpong":
                return b

        raise RuntimeError("❌ Could not find body named 'pingpong'.")




    def _get_free_joint_slices_for_body(self, body_id: int) -> tuple[slice, slice]:
        mj_model = self.env.unwrapped.sim.model.ptr

        jnt_adr = int(mj_model.body_jntadr[body_id])
        if jnt_adr < 0:
            raise RuntimeError(f"Body {body_id} has no joint.")

        jnt_type = int(mj_model.jnt_type[jnt_adr])
        if jnt_type != int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError("pingpong body is not a FREE joint body (unexpected).")

        qpos_adr = int(mj_model.jnt_qposadr[jnt_adr])
        qvel_adr = int(mj_model.jnt_dofadr[jnt_adr])

        qpos_slice = slice(qpos_adr, qpos_adr + 7)  # (x,y,z, qw,qx,qy,qz)
        qvel_slice = slice(qvel_adr, qvel_adr + 6)  # (vx,vy,vz, wx,wy,wz)
        return qpos_slice, qvel_slice

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

        # Cache canonical ball position ONCE
        if self._base_ball_qpos is None:
            self._base_ball_qpos = qpos[self._ball_qpos_id].copy()

        # =================================================
        # Stage 1: Ball position curriculum
        # =================================================
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

        # =================================================
        # Stage 2: Ball speed curriculum (FIXED)
        # =================================================
        # Speed: 0.3 m/s → 2.5 m/s
        min_speed = 0.3
        max_speed = 0.3 + 2.2 * level
        speed = self.np_random.uniform(min_speed, max_speed)

        # Direction: mostly towards player
        direction = np.array([
            -1.0,
            self.np_random.uniform(-0.25, 0.25),
            self.np_random.uniform(0.05, 0.25 + 0.35 * level),
        ])
        direction /= np.linalg.norm(direction) + 1e-8
        vel = direction * speed

        # 🔑 IMPORTANT: reset all DoFs first
        qvel_slice = qvel[self._ball_qvel_id]
        qvel_slice[:] = 0.0

        # Apply translational velocity safely
        n = min(len(qvel_slice), 3)
        qvel_slice[:n] = vel[:n]

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        info["is_success"] = bool(info.get("solved", False))

        # Curriculum logging
        info.update(self._get_curriculum_info())

        return obs, reward, terminated, truncated, info

    # -------------------------------------------------
    # Curriculum logging
    # -------------------------------------------------
    def _get_curriculum_info(self) -> Dict:
        level = float(getattr(self.config, "curriculum_level", 1.0))
        return {
            "curriculum_level": level,
            "max_offset_xy": 0.05 + 0.45 * level,
            "max_ball_speed": 0.3 + 2.2 * level,
        }

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
