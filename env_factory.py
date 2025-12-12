from stable_baselines3.common.vec_env import SubprocVecEnv
from loguru import logger

def build_env(cfg):
    def make_env():
        from env_policy import CustomEnv
        return CustomEnv(cfg)

    logger.info(f"🌍 Creating {cfg.num_envs} parallel environments.")
    return SubprocVecEnv([make_env for _ in range(cfg.num_envs)])
