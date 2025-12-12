from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def build_vec_env(worker: bool, cfg, worker_model_path: str = None):
    def make_env():
        if worker:
            from worker_env import WorkerEnv
            return WorkerEnv(cfg)
        else:
            from manager_env import ManagerEnv
            return ManagerEnv(cfg, worker_model_path)

    if worker:
        return SubprocVecEnv([make_env for _ in range(cfg.num_envs)])
    else:
        return DummyVecEnv([make_env])   # manager MUST be single-env
