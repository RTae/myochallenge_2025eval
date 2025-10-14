import os
import torch
import psutil
import numpy as np

from loguru import logger
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from myosuite.utils import gym
from utils.callbacks import VideoCallback, MetricCallback


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


def make_env(env_id, seed=0):
    """Factory so SubprocVecEnv can create isolated envs."""
    def _thunk():
        env = gym.make(env_id)
        env = SafeObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk


def linear_schedule(v0, v_end):
    return lambda progress_remaining: v_end + (v0 - v_end) * progress_remaining


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
    base_exp_name = "tabletennis_sac"
    i = 1
    while os.path.exists(os.path.join(WORKSPACE_ROOT, "logs", f"{base_exp_name}_run{i}")):
        i += 1
    EXPERIMENT_NAME = f"{base_exp_name}_run{i}"

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
    EVAL_FREQ_STEPS = int(os.environ.get("EVAL_FREQ_STEPS", 200_000))
    VIDEO_FREQ_STEPS = int(os.environ.get("VIDEO_FREQ_STEPS", 1_000_000))
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
    #  SAC CONFIGURATION
    # =====================================================
    logger.info("⚙️ Initializing SAC model")

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=TB_LOG,
        seed=SEED,
        learning_rate=linear_schedule(3e-4, 1e-5),
        buffer_size=1_000_000,
        batch_size=512,
        tau=0.005,
        gamma=0.98,
        train_freq=64,
        gradient_steps=64,
        ent_coef="auto",  # automatic entropy tuning
        policy_kwargs=dict(
            net_arch=[512, 512],
            log_std_init=-1.0,
        ),
        device=device,
    )

    # =====================================================
    #  CALLBACKS (CHECKPOINT / EVAL / VIDEO)
    # =====================================================
    checkpoint_cb = CheckpointCallback(
        save_freq=EVAL_FREQ_STEPS // max(N_ENVS, 1),
        save_path=MODELS_DIR,
        name_prefix="sac_myo",
        save_replay_buffer=True,
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
        eval_freq=VIDEO_FREQ_STEPS // max(N_ENVS, 1),
        video_dir=VIDEO_DIR,
        best_model_dir=BEST_DIR,
        n_eval_episodes=3,
        verbose=1,
        num_worker=max(N_ENVS, 1),
    )

    metric_cb = MetricCallback()

    # =====================================================
    #  TRAINING
    # =====================================================
    logger.info("🚀 Starting SB3 SAC training")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb, video_cb, metric_cb],
        progress_bar=True,
    )

    train_env.save(norm_stats_path)
    logger.success("✅ SAC training complete")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
