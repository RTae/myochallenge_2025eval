from typing import Optional
from loguru import logger

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from config import Config

from worker_env import TableTennisWorker
from manager_env import TableTennisManager
from custom_env import CustomEnv
from curriculum_env import CurriculumEnv


def create_worker_vector_env(cfg: Config, num_envs: int) -> VecNormalize:
    """
    Create vectorized Worker environments for training.
    
    Args:
        cfg: Configuration
        num_envs: Number of parallel environments
    
    Returns:
        Vectorized and normalized Worker environment
    """
    def make_env(rank: int):
        def _init():
            # Create Worker environment
            env = TableTennisWorker(cfg)
            return Monitor(env, info_keywords=("is_success",))
        return _init
    
    logger.info(f"Creating {num_envs} parallel Worker environments")
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    # Add normalization
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=False, 
        clip_obs=10.0
    )
    
    return env


def create_manager_vector_env(cfg: Config, 
                              worker_model: RecurrentPPO,
                              num_envs: int) -> VecNormalize:
    """
    Create vectorized Manager environments with pre-trained Workers.
    
    Args:
        cfg: Configuration
        worker_model_path: Path to trained Worker model (.zip)
        num_envs: Number of parallel environments
    
    Returns:
        Vectorized and normalized Manager environment
    """
    def make_env(rank: int):
        def _init():

            # Create Worker environment (for physics)
            worker_env = TableTennisWorker(cfg)
            

            # Load manger
            manager_env = TableTennisManager(
                worker_env=worker_env,
                worker_model=worker_model,
                config=cfg
            )
            
            return Monitor(manager_env, info_keywords=("is_success",))
        return _init
    
    logger.info(f"Creating {num_envs} parallel Manager environments")
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    # Add normalization
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=False, 
        clip_obs=10.0,
        gamma=cfg.ppo_gamma
    )
    
    return env

def create_plain_vector_env(cfg: Config, num_envs: int) -> VecNormalize:
    """
    Create vectorized plain environments for training.
    
    Args:
        cfg: Configuration
        num_envs: Number of parallel environments
    
    Returns:
        Vectorized and normalized plain environment
    """ 
    
    def make_env(rank: int):
        def _init():
            # Create plain environment
            env = CustomEnv(cfg)
            return Monitor(env, info_keywords=("is_success",))
        return _init
    
    logger.info(f"Creating {num_envs} parallel plain environments")
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    # Add normalization
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    return env

def create_curriculum_vector_env(cfg: Config, num_envs: int):
    def make_env(rank):
        def _init():
            env = CurriculumEnv(cfg)
            return Monitor(env, info_keywords=("is_success",))
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=cfg.ppo_gamma,
    )
    return env

def build_env(cfg: Config, 
              env_type: str,
              worker_model: Optional[RecurrentPPO] = None) -> VecNormalize:
    """
    Main factory function to create environments.
    
    Args:
        cfg: Configuration
        worker: If True, create Worker env; if False, create Manager env
        worker_model_path: Required if worker=False (path to trained Worker model)
    
    Returns:
        Vectorized environment ready for training
    """
    if env_type == "worker":
        logger.info("Building Worker environments...")
        return create_worker_vector_env(cfg, num_envs=cfg.num_envs)
    if env_type == "manager":
        if worker_model is None:
            raise ValueError(
                "The worker_model is required for Manager environment. "
                "Please provide a trained Worker model."
            )
        
        logger.info("Building Manager environments with trained Workers...")
        return create_manager_vector_env(
            cfg=cfg,
            worker_model=worker_model,
            num_envs=cfg.num_envs
        )
    if env_type == "plain":
        return create_plain_vector_env(cfg, num_envs=cfg.num_envs)

    if env_type == "curriculum":
        return create_curriculum_vector_env(cfg, num_envs=cfg.num_envs)
    
    raise ValueError(f"Unknown env_type: {env_type}")