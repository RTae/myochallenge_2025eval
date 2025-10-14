import os
import torch
import psutil
import numpy as np

from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from myosuite.utils import gym
from utils.callbacks import VideoCallback


# =====================================================
#  SAFE OBSERVATION WRAPPER
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


def make_env(env_id,seed=0):
    """Factory so SubprocVecEnv can create isolated envs."""
    def _thunk():
        env = gym.make(env_id)
        env = SafeObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk


def main():
    # =====================================================
    #  ENV / RUNTIME SETUP
    # =====================================================
    os.environ.pop("DISPLAY", None)

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "1")

    ENV_ID = os.environ.get("ENV_ID", "myoChallengeTableTennisP2-v0")
    WORKSPACE_ROOT = os.getenv("WORKSPACE_DIR", os.getcwd())

    # =====================================================
    #  AUTO-INCREMENT EXPERIMENT FOLDER
    # =====================================================
    base_exp_name = "exp_tabletennis"
    i = 1
    while os.path.exists(os.path.join(WORKSPACE_ROOT, "logs", f"{base_exp_name}_run{i}")):
        i += 1
    EXPERIMENT_NAME = f"{base_exp_name}_run{i}"

    # Unified experiment directory
    EXP_DIR = os.path.join(WORKSPACE_ROOT, "logs", EXPERIMENT_NAME)

    TB_LOG = os.path.join(EXP_DIR, "tb")
    MODELS_DIR = os.path.join(EXP_DIR, "models")
    VIDEO_DIR = os.path.join(EXP_DIR, "videos")
    BEST_DIR = os.path.join(EXP_DIR, "best")
    EVAL_DIR = os.path.join(EXP_DIR, "eval")
    STATS_DIR = os.path.join(EXP_DIR, "vecnormalize")

    for d in [TB_LOG, MODELS_DIR, VIDEO_DIR, BEST_DIR, EVAL_DIR, STATS_DIR]:
        os.makedirs(d, exist_ok=True)

    logger.info(f"📂 Logging to {EXP_DIR}")

    # =====================================================
    #  HYPERPARAMETERS
    # =====================================================
    TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", 10_000_000))
    EVAL_FREQ_STEPS = int(os.environ.get("EVAL_FREQ_STEPS", 1_000_000))
    SEED = int(os.environ.get("SEED", 42))

    total_cpus = psutil.cpu_count(logical=True)
    default_envs = max(4, total_cpus - 2)
    N_ENVS = int(os.environ.get("N_ENVS", str(default_envs)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {N_ENVS} parallel envs on {total_cpus} CPUs, device={device}")

    # =====================================================
    #  ENVIRONMENT SETUP
    # =====================================================
    train_env = SubprocVecEnv([make_env(ENV_ID, seed=SEED + i) for i in range(N_ENVS)])
    train_env = VecMonitor(train_env)

    norm_stats_path = os.path.join(STATS_DIR, "stats.pkl")
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    if os.path.exists(norm_stats_path):
        logger.info(f"Loading VecNormalize stats from {norm_stats_path}")
        train_env = VecNormalize.load(norm_stats_path, train_env)
    train_env.training = True

    eval_env = SubprocVecEnv([make_env(ENV_ID, seed=SEED + 10_000 + i) for i in range(2)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)
    eval_env.training = False
    eval_env.obs_rms = train_env.obs_rms

    # =====================================================
    #  PPO CONFIGURATION
    # =====================================================
    n_steps = int(os.environ.get("N_STEPS", "2048"))
    batch_per_update = n_steps * N_ENVS
    logger.info(f"n_steps={n_steps}, total rollout per update={batch_per_update}")

    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        ortho_init=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=TB_LOG,
        seed=SEED,
        n_steps=n_steps,
        batch_size=int(os.environ.get("BATCH_SIZE", "1024")),
        n_epochs=int(os.environ.get("N_EPOCHS", "10")),
        learning_rate=float(os.environ.get("LR", "3e-4")),
        gamma=0.99,
        ent_coef=float(os.environ.get("ENT_COEF", "0.03")),
        clip_range=float(os.environ.get("CLIP_RANGE", "0.15")),
        vf_coef=float(os.environ.get("VF_COEF", "0.5")),
        clip_range_vf=float(os.environ.get("VF_CLIP", "10.0")),
        max_grad_norm=0.5,
        target_kl=float(os.environ.get("TARGET_KL", "0.1")),
        policy_kwargs=policy_kwargs,
        device=device,
    )

    # =====================================================
    #  CALLBACKS (CHECKPOINT / EVAL / VIDEO)
    # =====================================================
    checkpoint_cb = CheckpointCallback(
        save_freq=EVAL_FREQ_STEPS // max(N_ENVS, 1),
        save_path=MODELS_DIR,
        name_prefix="ppo_myo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=BEST_DIR,
        n_eval_episodes=3,
        eval_freq=EVAL_FREQ_STEPS // max(N_ENVS, 1),
        deterministic=True,
        render=False,
        log_path=EVAL_DIR,
    )

    video_cb = VideoCallback(
        eval_env_id=ENV_ID,
        eval_freq=EVAL_FREQ_STEPS // max(N_ENVS, 1),
        video_dir=VIDEO_DIR,
        best_model_dir=BEST_DIR,
        n_eval_episodes=3,
        verbose=1,
        num_worker=max(N_ENVS, 1),
    )

    # =====================================================
    #  TRAINING
    # =====================================================
    logger.info("🚀 Starting SB3 PPO training")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb, video_cb],
        progress_bar=True,
    )

    # Save normalization stats
    train_env.save(norm_stats_path)
    logger.success("✅ Training complete")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
