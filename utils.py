import numpy as np
from loguru import logger
from config import Config
import os

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
def make_predict_fn(model):
    def predict_fn(obs, env_instance):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return predict_fn

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