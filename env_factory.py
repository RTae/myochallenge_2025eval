from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,          # 👈 ADD THIS
)

def build_vec_env(worker: bool, cfg, worker_model_path: str = None):
    def make_env():
        if worker:
            from worker_env import WorkerEnv
            return WorkerEnv(cfg)
        else:
            from manager_env import ManagerEnv
            return ManagerEnv(cfg, worker_model_path)

    if worker:
        env = SubprocVecEnv([make_env for _ in range(cfg.num_envs)])
    else:
        env = DummyVecEnv([make_env])

    env = VecMonitor(env)

    return env