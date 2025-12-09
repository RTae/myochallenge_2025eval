# env_factory.py
import os
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from worker_env import WorkerEnv
from manager_env import ManagerEnv
from config import Config


def make_worker_env(cfg: Config):
    """Creates one worker env instance for SubprocVecEnv."""
    def _init():
        env = WorkerEnv(cfg)
        return Monitor(env)
    return _init


def make_manager_env(cfg: Config):
    """Creates one manager env instance for SubprocVecEnv."""
    def _init():
        env = ManagerEnv(cfg, worker_model_path="worker.zip")
        return Monitor(env)
    return _init


def build_vec_env(worker: bool, cfg: Config, eval_env=False):
    """
    Creates parallelized SubprocVecEnv + VecNormalize.
    """
    if worker:
        env_fns = [make_worker_env(cfg) for _ in range(cfg.num_envs)]
    else:
        env_fns = [make_manager_env(cfg) for _ in range(cfg.num_envs)]

    vec_env = SubprocVecEnv(env_fns)

    # Normalize observations + rewards
    vec_env = VecNormalize(
        vec_env,
        training=not eval_env,
        norm_obs=True,
        norm_reward=True,
        gamma=cfg.norm_gamma,
    )
    return vec_env
