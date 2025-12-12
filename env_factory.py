from stable_baselines3.common.vec_env import SubprocVecEnv

def build_env(cfg):
    def make_env():
        from env_policy import CustomEnv
        return CustomEnv(cfg)

    return SubprocVecEnv([make_env for _ in range(cfg.num_envs)])
