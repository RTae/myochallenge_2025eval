# env_factory.py
from typing import Callable

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from config import Config
from worker_env import WorkerEnv
from manager_env import ManagerEnv


def build_vec_env(worker: bool, cfg: Config, eval_env: bool = False, worker_model_path: str | None = None):
    """
    Build a parallel VecEnv over WorkerEnv or ManagerEnv.
    No VecNormalize here to keep worker<->manager integration simple.
    """

    def make_env(rank: int) -> Callable:
        def _init():
            if worker:
                env = WorkerEnv(cfg)
            else:
                if worker_model_path is None:
                    raise ValueError("worker_model_path required when worker=False")
                env = ManagerEnv(cfg, worker_model_path=worker_model_path)

            env = Monitor(env)
            env.reset(seed=cfg.seed + rank)
            return env

        return _init

    env_fns = [make_env(i) for i in range(cfg.num_envs)]

    if cfg.num_envs == 1:
        return DummyVecEnv(env_fns)
    else:
        return SubprocVecEnv(env_fns)
