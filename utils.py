from loguru import logger
from config import Config
import numpy as np
from typing import Optional
from stable_baselines3.common.vec_env import VecNormalize
import os

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

def quat_to_paddle_normal(q: np.ndarray) -> np.ndarray:
    """
    Paddle hitting normal = local +X axis
    Quaternion format: (x, y, z, w)
    """
    q = q.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)

    x, y, z, w = q

    # X axis of rotation matrix
    return np.array([
        1.0 - 2.0 * (y*y + z*z),
        2.0 * (x*y + w*z),
        2.0 * (x*z - w*y),
    ], dtype=np.float32)

def resume_vecnormalize_on_training_env(
    training_env,
    load_vecnorm_path: Optional[str],
    *,
    training: bool,
    norm_reward: bool,
):
    """
    This keeps your build_worker_vec() stack intact.

    If build_worker_vec returns VecNormalize already:
      - we load stats and attach to the underlying venv to preserve wrappers.
    If it returns a plain VecEnv:
      - we wrap it with VecNormalize.load.

    Returns the (possibly replaced) env.
    """
    if not load_vecnorm_path:
        return training_env
    if not os.path.exists(load_vecnorm_path):
        logger.warning(f"[Worker] VecNormalize load path not found: {load_vecnorm_path}")
        return training_env

    # If training_env is already VecNormalize, preserve its underlying venv
    if isinstance(training_env, VecNormalize):
        base_venv = training_env.venv
        logger.info(f"[Worker] Loading VecNormalize stats onto existing VecNormalize. path={load_vecnorm_path}")
        new_env = VecNormalize.load(load_vecnorm_path, base_venv)
    else:
        logger.info(f"[Worker] Wrapping training env with VecNormalize.load. path={load_vecnorm_path}")
        new_env = VecNormalize.load(load_vecnorm_path, training_env)

    new_env.training = training
    new_env.norm_reward = norm_reward
    return new_env
