# train_all.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from config import Config
from env_factory import build_vec_env

from loguru import logger

from callbacks.video_callback import VideoCallback
from hrl_utils import flatten_myo_obs_worker, make_hierarchical_predictor



# ============================================================
# Create experiment directory logs/expN/
# ============================================================

def prepare_experiment_directory(cfg: Config):
    """
    Creates logs/exp1, exp2,... automatically.
    Updates cfg.logdir to point to the new experiment folder.
    """

    base = cfg.logdir  # default = "./logs"
    os.makedirs(base, exist_ok=True)

    # find existing exp folders
    existing = [d for d in os.listdir(base) if d.startswith("exp")]
    exp_nums = []

    for e in existing:
        try:
            num = int(e.replace("exp", ""))
            exp_nums.append(num)
        except:
            pass

    next_id = 1 if len(exp_nums) == 0 else max(exp_nums) + 1

    exp_dir = os.path.join(base, f"exp{next_id}")
    os.makedirs(exp_dir, exist_ok=True)

    # update config logdir
    cfg.logdir = exp_dir
    logger.info(f"📁 Created new experiment folder: {exp_dir}")

    return exp_dir



# ============================================================
# TRAIN WORKER (LOW-LEVEL CONTROLLER)
# ============================================================

def train_worker(cfg: Config):

    logger.info("Training worker...")

    worker_logdir = os.path.join(cfg.logdir, "worker")
    os.makedirs(worker_logdir, exist_ok=True)

    # -------- Parallel Envs --------
    env = build_vec_env(worker=True, cfg=cfg)
    eval_env = build_vec_env(worker=True, cfg=cfg, eval_env=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=worker_logdir,
        log_path=worker_logdir,
        eval_freq=cfg.eval_freq,
        deterministic=True,
        n_eval_episodes=cfg.eval_episodes,
    )

    # -------- PPO Model --------
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=worker_logdir,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        learning_rate=cfg.ppo_lr,
        gae_lambda=cfg.ppo_lambda,
        clip_range=cfg.ppo_clip,
        n_epochs=cfg.ppo_epochs,
        seed=cfg.seed,
    )

    # -------- VideoCallback for Worker --------
    def worker_predict(obs, env_instance):
        obs_vec = flatten_myo_obs_worker(obs).reshape(1, -1)
        action, _ = model.predict(obs_vec, deterministic=True)
        return action

    video_cb = VideoCallback(cfg, mode="worker", predict_fn=worker_predict)

    # -------- TRAIN --------
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[eval_callback, video_cb],
    )

    # -------- SAVE --------
    model.save(os.path.join(cfg.logdir, "worker.zip"))
    env.save(os.path.join(cfg.logdir, "worker_norm.pkl"))

    logger.info("Worker saved → worker.zip + worker_norm.pkl")

    env.close()
    eval_env.close()



# ============================================================
# TRAIN MANAGER (HIGH-LEVEL CONTROLLER)
# ============================================================

def train_manager(cfg: Config):

    logger.info("Training manager...")

    manager_logdir = os.path.join(cfg.logdir, "manager")
    os.makedirs(manager_logdir, exist_ok=True)

    # -------- Parallel Envs --------
    env = build_vec_env(worker=False, cfg=cfg)
    eval_env = build_vec_env(worker=False, cfg=cfg, eval_env=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=manager_logdir,
        log_path=manager_logdir,
        eval_freq=cfg.eval_freq,
        deterministic=True,
        n_eval_episodes=cfg.eval_episodes,
    )

    # -------- PPO Model --------
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=manager_logdir,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        learning_rate=cfg.ppo_lr,
        gae_lambda=cfg.ppo_lambda,
        clip_range=cfg.ppo_clip,
        n_epochs=cfg.ppo_epochs,
        seed=cfg.seed,
    )

    # -------- HRL Predictor --------
    from stable_baselines3 import PPO as PPO_LOAD
    worker_model = PPO_LOAD.load(os.path.join(cfg.logdir, "worker.zip"))

    hrl_predict = make_hierarchical_predictor(cfg, model, worker_model)

    video_cb = VideoCallback(cfg, mode="manager", predict_fn=hrl_predict)

    # -------- TRAIN --------
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[eval_callback, video_cb],
    )

    # -------- SAVE --------
    model.save(os.path.join(cfg.logdir, "manager.zip"))
    env.save(os.path.join(cfg.logdir, "manager_norm.pkl"))

    logger.info("Manager saved → manager.zip + manager_norm.pkl")

    env.close()
    eval_env.close()



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    cfg = Config()

    # create logs/expN folder automatically
    prepare_experiment_directory(cfg)

    train_worker(cfg)
    train_manager(cfg)

    logger.info("\n🎉 HRL Training Complete!")
