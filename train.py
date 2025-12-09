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
        # OBS: raw MyoSuite obs dict
        obs_vec = flatten_myo_obs_worker(obs).reshape(1, -1)
        action, _ = model.predict(obs_vec, deterministic=True)
        return action

    video_cb = VideoCallback(
        cfg,
        mode="worker",
        predict_fn=worker_predict
    )

    # -------- TRAIN --------
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[eval_callback, video_cb],
    )

    # -------- SAVE MODELS --------
    model.save("worker.zip")
    env.save("worker_norm.pkl")

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

    # -------- Hierarchical Predictor (Manager + Worker) --------
    from stable_baselines3 import PPO as PPO_LOAD
    worker_model = PPO_LOAD.load("worker.zip")

    hrl_predict = make_hierarchical_predictor(cfg, model, worker_model)

    video_cb = VideoCallback(
        cfg,
        mode="manager",
        predict_fn=hrl_predict
    )

    # -------- TRAIN --------
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[eval_callback, video_cb],
    )

    # -------- SAVE MODELS --------
    model.save("manager.zip")
    env.save("manager_norm.pkl")

    logger.info("Manager saved → manager.zip + manager_norm.pkl")

    env.close()
    eval_env.close()



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    cfg = Config()

    train_worker(cfg)
    train_manager(cfg)

    print("\n🎉 HRL Training Complete with Parallel Env + VecNormalize + Video Recording!")
