import os
import ray
from myosuite.utils import gym
import numpy as np
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from loguru import logger
from utils.callbacks import MyoUnifiedVideoCallback


# --- Fix wrapper: expands observation space to infinite range ---
class SafeObsWrapper(gym.ObservationWrapper):
    """Ensure observations never fail Box range checks."""
    def __init__(self, env):
        super().__init__(env)
        low = np.full(env.observation_space.shape, -np.inf, dtype=np.float32)
        high = np.full(env.observation_space.shape, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, obs):
        # Optionally clip to a large but finite range for safety
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)


def main():
    # =====================================================
    #  ENVIRONMENT & RENDER CONFIG
    # =====================================================
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["EGL_DEVICE_ID"] = "0"
    os.environ.pop("DISPLAY", None)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    ENV_ID = "myoChallengeTableTennisP2-v0"
    SAFE_ENV_NAME = "MyoSafeWrapper-v0"
    TOTAL_TIMESTEPS = 1_000_000
    LOG_PATH = "./logs/rllib_tabletennis_unified"
    STORAGE_PATH = "file://" + os.path.abspath(LOG_PATH)

    # =====================================================
    #  REGISTER SAFE ENVIRONMENT WITH RAY
    # =====================================================
    def make_safe_env(cfg=None):
        env = gym.make(ENV_ID)
        return SafeObsWrapper(env)

    # register custom env into Ray Tune registry
    tune.register_env(SAFE_ENV_NAME, make_safe_env)
    logger.info(f"✅ Registered safe environment '{SAFE_ENV_NAME}' for Ray RLlib")

    # =====================================================
    #  INITIALIZE RAY
    # =====================================================
    ray.init(ignore_reinit_error=True)
    logger.info(f"Cluster resources: {ray.cluster_resources()}")

    # =====================================================
    #  RLlib PPO CONFIGURATION
    # =====================================================
    config = (
        PPOConfig()
        .environment(env=SAFE_ENV_NAME)  # ✅ registered through Tune
        .framework("torch")
        .resources(num_gpus=2)
        .env_runners(
            num_env_runners=12,
            rollout_fragment_length=512,
            num_cpus_per_env_runner=2,
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            train_batch_size=40960,
            sgd_minibatch_size=1024,
            num_sgd_iter=10,
            vf_clip_param=10.0,
            clip_param=0.2,
            entropy_coeff=0.01,
        )
        .rollouts(enable_connectors=False)
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=3,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False},
        )
        .reporting(metrics_num_episodes_for_smoothing=5)
        .callbacks(
            lambda: MyoUnifiedVideoCallback(
                eval_env_id=ENV_ID,  # still the base env for actual rendering
                eval_freq=10_000,
                video_dir=f"{LOG_PATH}/rllib_videos",
                best_model_dir=f"{LOG_PATH}/rllib_best",
                n_eval_episodes=3,
                verbose=1,
            )
        )
    )

    # =====================================================
    #  RUN PPO TRAINING (Ray AIR API)
    # =====================================================
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            name="rllib_myo_tabletennis_p2_unified",
            storage_path=STORAGE_PATH,
            stop={"timesteps_total": TOTAL_TIMESTEPS},
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=3,
                checkpoint_at_end=True,
            ),
        ),
    )

    tuner.fit()
    ray.shutdown()
    logger.success("✅ Training complete with unified video callback")


if __name__ == "__main__":
    main()
