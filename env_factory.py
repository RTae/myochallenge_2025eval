from typing import Callable
from loguru import logger

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor

from config import Config
from hrl.worker_env import TableTennisWorker
from hrl.manager_env import TableTennisManager
from custom_env import CustomEnv


def make_subproc_env(num_envs: int, thunk_fn: Callable):
    return SubprocVecEnv([thunk_fn(i) for i in range(num_envs)])


# ============================================================
# Worker Vec (Low-Level)
# ============================================================
def build_worker_vec(cfg: Config, num_envs: int) -> VecNormalize:
    def make_env(rank: int):
        def _init():
            import torch
            # Prevent threads from fighting over CPU resources
            torch.set_num_threads(1)
            
            env = TableTennisWorker(cfg)
            return env
        return _init
    
    logger.info(f"Creating {num_envs} low-level environments")
    venv = make_subproc_env(num_envs, make_env)
    venv = VecMonitor(venv, info_keywords=("is_success",))
    return VecNormalize(venv, norm_obs=True, norm_reward=False)


# ============================================================
# Manager Vec (High-Level)
# ============================================================
def build_manager_vec(
    cfg: Config,
    num_envs: int,
    worker_model_loader,
    worker_env_loader,
    worker_model_path: str,
    worker_env_path: str,
    decision_interval: int,
    max_episode_steps: int,
):
    """
    Each manager env owns:
      - its own worker env (VecNormalize-wrapped)
      - its own frozen worker policy
    """

    def make_env(rank: int):
        def _init():
            import torch
            # Prevent threads from fighting over CPU resources
            torch.set_num_threads(1)
            
            worker_vec = worker_env_loader(worker_env_path)
            worker_model = worker_model_loader(worker_model_path)

            if hasattr(worker_vec, "env_method"):
                worker_vec.env_method("set_goal_noise_scale", 0.0)
                worker_vec.env_method("set_progress", 1.0)

            env = TableTennisManager(
                worker_env=worker_vec,
                worker_model=worker_model,
                config=cfg,
                decision_interval=decision_interval,
                max_episode_steps=max_episode_steps,
            )
            return env

        return _init

    logger.info(f"Creating {num_envs} high-level environments")
    venv = make_subproc_env(num_envs, make_env)
    venv = VecMonitor(venv, info_keywords=("is_success",))

    return VecNormalize(venv, norm_obs=False, norm_reward=False)

def create_default_env(cfg: Config, num_envs: int) -> VecNormalize:
    def make_env(rank: int):
        def _init():
            env = CustomEnv(cfg)
            return env
        return _init

    logger.info(f"Creating {num_envs} plain environments")
    venv = make_subproc_env(num_envs, make_env)
    
    venv = VecMonitor(venv, info_keywords=("is_success"))
    
    return VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=cfg.ppo_gamma,
    )