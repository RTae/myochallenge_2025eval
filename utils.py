import numpy as np
from loguru import logger
import Config
import os

def flatten_obs(obs_dict):
    """
    Build a 1D float32 vector from the MyoSuite obs_dict.

    Keys (with shapes from your dump):
      time           (1,)
      pelvis_pos     (3,)
      body_qpos      (58,)
      body_qvel      (58,)
      ball_pos       (3,)
      ball_vel       (3,)
      paddle_pos     (3,)
      paddle_vel     (3,)
      paddle_ori     (4,)
      reach_err      (3,)
      palm_pos       (3,)
      palm_err       (3,)
      touching_info  (6,)
      act            (273,)

    Total: 424 dims
    """
    parts = [
        obs_dict["time"],
        obs_dict["pelvis_pos"],
        obs_dict["body_qpos"],
        obs_dict["body_qvel"],
        obs_dict["ball_pos"],
        obs_dict["ball_vel"],
        obs_dict["paddle_pos"],
        obs_dict["paddle_vel"],
        obs_dict["paddle_ori"],
        obs_dict["reach_err"],
        obs_dict["palm_pos"],
        obs_dict["palm_err"],
        obs_dict["touching_info"],
        obs_dict["act"],
    ]

    arrays = []
    for p in parts:
        arr = np.asarray(p, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        arrays.append(arr)

    return np.concatenate(arrays, axis=-1)

class HitDetector:
    """Detects ball-paddle contact using velocity change."""

    def __init__(self, dv_thr: float = 2.5, ball_mass: float = 0.0027, paddle_face_radius: float = 0.05):
        self.dv_thr = dv_thr
        self.ball_mass = ball_mass
        self.paddle_face_radius = paddle_face_radius
        self._prev_ball_vel = None

    def reset(self, obs_dict: dict):
        self._prev_ball_vel = np.array(obs_dict["ball_vel"], dtype=np.float32)

    def step(self, obs_dict: dict, dt: float = 0.01):
        ball_vel = np.array(obs_dict["ball_vel"], dtype=np.float32)

        dv = 0.0 if self._prev_ball_vel is None else float(
            np.linalg.norm(ball_vel - self._prev_ball_vel)
        )

        paddle_pos = obs_dict["paddle_pos"]
        ball_pos = obs_dict["ball_pos"]
        near_paddle = np.linalg.norm(ball_pos - paddle_pos) < 1.2 * self.paddle_face_radius

        hit = (dv > self.dv_thr) and near_paddle

        if hit:
            contact_force = self.ball_mass * dv / dt
            self._prev_ball_vel = ball_vel.copy()
            return True, contact_force, dv

        self._prev_ball_vel = ball_vel.copy()
        return False, 0.0, dv


# For video callback
def predict_fn(model, obs, env_instance):
    action, _ = model.predict(obs, deterministic=True)
    return action

def prepare_experiment_directory(cfg: Config):
    """
    Creates logs/exp1, exp2, ... automatically.
    Updates cfg.logdir to the new experiment folder.
    """
    base = cfg.logdir
    os.makedirs(base, exist_ok=True)

    existing = [d for d in os.listdir(base) if d.startswith("exp")]
    exp_nums = []
    for e in existing:
        try:
            exp_nums.append(int(e.replace("exp", "")))
        except Exception:
            pass

    next_id = 1 if not exp_nums else max(exp_nums) + 1
    exp_dir = os.path.join(base, f"exp{next_id}")
    os.makedirs(exp_dir, exist_ok=True)

    cfg.logdir = exp_dir
    logger.info(f"📁 Created new experiment folder: {exp_dir}")
    return exp_dir