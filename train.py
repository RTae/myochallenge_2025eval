import os
import ray
import psutil
import torch
import numpy as np
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from loguru import logger
from myosuite.utils import gym
from utils.callbacks import MyoUnifiedVideoCallback


# =====================================================
#  Safe Observation Wrapper
# =====================================================
class SafeObsWrapper(gym.ObservationWrapper):
    """Ensure observations never fail Box range checks."""
    def __init__(self, env):
        super().__init__(env)
        low = np.full(env.observation_space.shape, -np.inf, dtype=np.float32)
        high = np.full(env.observation_space.shape, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, obs):
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)


def main():
    # =====================================================
    #  ENVIRONMENT CONFIGURATION
    # =====================================================
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["EGL_DEVICE_ID"] = "0"
    os.environ.pop("DISPLAY", None)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"

    ENV_ID = "myoChallengeTableTennisP2-v0"
    SAFE_ENV_NAME = "MyoSafeWrapper-v0"
    TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", 10_000_000))
    EVAL_TIMESTEPS = int(os.environ.get("EVAL_TIMESTEPS", 100_000))  # eval less often
    WORKSPACE_ROOT = os.getenv("WORKSPACE_DIR", os.getcwd())
    LOG_PATH = os.path.join(WORKSPACE_ROOT, "logs/rllib_tabletennis")
    STORAGE_PATH = "file://" + os.path.abspath(LOG_PATH)
    VIDEO_DIR = os.path.join(LOG_PATH, "rllib_videos")
    BEST_MODEL_DIR = os.path.join(LOG_PATH, "rllib_best")
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    # =====================================================
    #  REGISTER SAFE ENVIRONMENT
    # =====================================================
    def make_safe_env(cfg=None):
        # ⚡ try 0.0015 (safe speedup). If unstable, set back to 0.001.
        env = gym.make(ENV_ID, sim_timestep=0.0015)
        return SafeObsWrapper(env)

    tune.register_env(SAFE_ENV_NAME, make_safe_env)
    logger.info(f"✅ Registered safe environment '{SAFE_ENV_NAME}'")

    # =====================================================
    #  INITIALIZE RAY
    # =====================================================
    ray.init(ignore_reinit_error=True, local_mode=False)
    NUM_CORES = psutil.cpu_count(logical=True)
    NUM_GPUS = torch.cuda.device_count()
    logger.info(f"Cluster resources → {NUM_CORES} CPU cores, {NUM_GPUS} GPUs")

    # =====================================================
    #  PPO CONFIGURATION
    #  7 workers × 4 CPUs = 28 cores; 56 envs total (7×8)
    #  Per-iter samples: 56 envs × 128 fragment = 7168
    # =====================================================
    config = (
        PPOConfig()
        .environment(env=SAFE_ENV_NAME)
        .framework("torch")
        .resources(
            num_gpus=NUM_GPUS,            # use both GPUs for learner
            num_cpus_for_driver=2,        # reserve ~2 for driver/learner
        )
        .env_runners(
            num_env_runners=7,
            num_cpus_per_env_runner=4,    # ✅ 7×4 = 28 cores (matches your hardware)
            num_gpus_per_env_runner=0.15, # small GPU slice for inference (optional)
            num_envs_per_env_runner=8,    # strong vectorization → less IPC overhead
            rollout_fragment_length=128,  # shorter rollouts → less sync waiting
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            train_batch_size=7168,        # ✅ exact multiple of rollout (56×128)
            sgd_minibatch_size=1024,      # 7 minibatches/iter
            num_sgd_iter=10,
            vf_clip_param=10.0,
            clip_param=0.15,              # a bit tighter for stability
            entropy_coeff=0.03,           # more exploration → more swing attempts
            kl_coeff=0.1,                 # stabilize exploration
            use_kl_loss=True,
        )
        .rollouts(
            enable_connectors=False,
            horizon=2000,                 # allow long trajectories for contact timing
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=3,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False},
        )
        .reporting(
            metrics_num_episodes_for_smoothing=5,
            min_time_s_per_iteration=10,  # print progress every ~10 s
        )
        .callbacks(
            lambda: MyoUnifiedVideoCallback(
                eval_env_id=ENV_ID,
                eval_freq=EVAL_TIMESTEPS,  # less frequent → fewer stalls
                video_dir=VIDEO_DIR,
                best_model_dir=BEST_MODEL_DIR,
                n_eval_episodes=3,
                verbose=1,
            )
        )
    )

    # =====================================================
    #  RUN TRAINING
    # =====================================================
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            name="rllib_myo_tabletennis_p2",
            storage_path=STORAGE_PATH,
            verbose=2,
            stop={"timesteps_total": TOTAL_TIMESTEPS},
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=3,
                checkpoint_at_end=True,
            ),
        ),
    )

    tuner.fit()
    ray.shutdown()
    logger.success("✅ Optimized PPO training complete!")


if __name__ == "__main__":
    main()
