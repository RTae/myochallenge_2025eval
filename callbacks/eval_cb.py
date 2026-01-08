
import os
from loguru import logger

from stable_baselines3.common.callbacks import BaseCallback

class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper to disk.
    Intended to be used as `callback_on_new_best` inside an EvalCallback.
    """
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        # Check if the environment is wrapped in VecNormalize
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(self.save_path)
            if self.verbose > 0:
                logger.info(f"Saved VecNormalize stats to {self.save_path}")
        else:
            if self.verbose > 0:
                logger.warning("Environment not wrapped in VecNormalize, skipping save.")
        return True