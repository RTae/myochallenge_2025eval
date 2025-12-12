from loguru import logger
from config import Config
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