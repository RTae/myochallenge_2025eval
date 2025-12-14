from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from loguru import logger

def build_env(cfg):
    def make_env():
        from myosuite.utils import gym as myo_gym
        return Monitor(myo_gym.make(cfg.env_id))

    logger.info(f"Creating {cfg.num_envs} parallel environments")
    env = SubprocVecEnv([make_env for _ in range(cfg.num_envs)])
    return env