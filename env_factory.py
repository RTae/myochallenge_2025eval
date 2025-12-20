from typing import Callable
from loguru import logger
import os

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from config import Config

from hrl.worker_env import TableTennisWorker
from hrl.manager_env import TableTennisManager
from dr_spcrl.curriculum_env import CurriculumEnv
from custom_env import CustomEnv


def make_subproc_env(num_envs: int, thunk_fn: Callable):
    return SubprocVecEnv([thunk_fn(i) for i in range(num_envs)])


def build_worker_vec(cfg: Config, num_envs: int) -> VecNormalize:
    def make_env(rank: int):
        def _init():
            env = TableTennisWorker(cfg)
            return Monitor(env, info_keywords=("is_success",))
        return _init

    venv = make_subproc_env(num_envs, make_env)

    # Worker obs normalization is OK
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv

def build_manager_vec(
    cfg,
    num_envs: int,
    worker_model_loader,
    worker_env_loader,
    worker_model_path: str,
    worker_env_path: str,
    decision_interval: int,
    max_episode_steps: int,
):
    """
    Vectorized manager environment using SubprocVecEnv.

    Each manager env owns:
      - its own worker env (VecNormalize-wrapped)
      - its own frozen worker policy
    """

    def make_env(rank: int):
        def _init():
            worker = TableTennisWorker(cfg)

            worker_vec = worker_env_loader(worker_env_path, lambda: worker)
            worker_model = worker_model_loader(worker_model_path)
            manager_env = TableTennisManager(
                worker_env=worker_vec,
                worker_model=worker_model,
                config=cfg,
                decision_interval=decision_interval,
                max_episode_steps=max_episode_steps,
            )

            return Monitor(
                manager_env,
                info_keywords=(
                    "is_success",
                    "is_paddle_hit",
                ),
            )

        return _init

    venv = make_subproc_env(num_envs, make_env)
    venv = VecNormalize(
        venv,
        norm_obs=False,
        norm_reward=False,
    )

    return venv

def build_curriculum_vec(cfg: Config, num_envs: int, eval_mode: bool = False):
    def make_env(rank):
        def _init():
            env = CurriculumEnv(cfg, eval_mode=eval_mode)
            return Monitor(env, info_keywords=("is_success",))
        return _init

    venv = make_subproc_env(num_envs, make_env)

    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    return venv

def create_default_env(cfg: Config, num_envs: int) -> VecNormalize:    
    def make_env(rank: int):
        def _init():
            env = CustomEnv(cfg)
            return Monitor(env, info_keywords=("is_success",))
        return _init
    
    logger.info(f"Creating {num_envs} parallel plain environments")
    
    venv = make_subproc_env(num_envs, make_env)
    
    venv = VecNormalize(
        venv, 
        norm_obs=True, 
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=cfg.ppo_gamma,
    )
    
    return venv