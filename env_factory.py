# env_factory.py
import os
from typing import Callable, List

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from config import Config
from worker_env import WorkerEnv
from manager_env import ManagerEnv


def make_env(worker: bool, cfg: Config, rank: int) -> Callable:
    """
    Factory for a single environment instance (for SubprocVecEnv).
    """

    def _init():
        if worker:
            env = WorkerEnv(cfg)
        else:
            worker_model_path = os.path.join(cfg.logdir, "worker", "worker.zip")
            env = ManagerEnv(cfg, worker_model_path=worker_model_path)

        env = Monitor(env)
        return env

    return _init


def build_vec_env(worker: bool, cfg: Config, eval_env: bool = False):
    """
    Build a parallel VecNormalize(SubprocVecEnv) for worker or manager.
    """
    env_fns: List[Callable] = [
        make_env(worker=worker, cfg=cfg, rank=i) for i in range(cfg.num_envs)
    ]

    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(
        vec_env,
        gamma=cfg.norm_gamma,
        norm_obs=True,
        norm_reward=True,
    )
    return vec_env
