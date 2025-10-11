import os
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from utils.callbacks import VideoEvalCallback
from loguru import logger


# =============================
#  CONFIGURATION
# =============================

ENV_ID = "myoChallengeTableTennisP2-v0"
N_ENVS = 25
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ = 50_000

LOG_DIR = "./logs/logs_tabletennis_p2_full/"
VIDEO_DIR = os.path.join(LOG_DIR, "videos")
BEST_MODEL_DIR = os.path.join(LOG_DIR, "best_model")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# --- Optional: cleaner warnings ---
warnings.filterwarnings("ignore", message=".*Unused kwargs found.*")
warnings.filterwarnings("ignore", message=".*obs returned by the `step.*")

# --- Limit threads per process (important for MuJoCo parallelization) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def main():
    # =============================
    #  CREATE PARALLEL ENVIRONMENTS
    # =============================

    logger.info(f"Creating {N_ENVS} parallel environments for training...")

    train_env = make_vec_env(
        ENV_ID,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
    )

    # =============================
    #  DEFINE PPO MODEL
    # =============================

    model = PPO(
        "MlpPolicy",
        train_env,
        device="cuda",
        verbose=1,
        n_steps=4096,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        tensorboard_log=LOG_DIR,
    )

    # =============================
    #  CALLBACKS
    # =============================

    eval_callback = EvalCallback(
        train_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    video_callback = VideoEvalCallback(
        eval_env_id=ENV_ID,
        eval_freq=EVAL_FREQ,
        video_dir=VIDEO_DIR,
        best_model_dir=BEST_MODEL_DIR,
        n_eval_episodes=3,
        verbose=1,
    )

    callback_list = CallbackList([eval_callback, video_callback])

    # =============================
    #  TRAINING
    # =============================

    logger.info(f"🚀 Starting PPO training with eval & video every {EVAL_FREQ} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list)
    model.save(os.path.join(LOG_DIR, "ppo_tabletennis_p2_final"))
    logger.success(f"✅ Training complete. Model saved at: {LOG_DIR}")


if __name__ == "__main__":
    main()
