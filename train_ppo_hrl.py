import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import Config
from env_factory import build_worker_vec, build_manager_vec
from callbacks.infologger_callback import InfoLoggerCallback
from callbacks.video_callback import VideoCallback
from hrl.worker_env import TableTennisWorker
from hrl.manager_env import TableTennisManager
from utils import prepare_experiment_directory, make_predict_fn


# ==================================================
# Worker loader (IMPORTANT: must match save format)
# ==================================================
def load_worker_model(path: str):
    # SB3 supports .pkl filenames
    return PPO.load(path, device="cpu")

def load_worker_vecnormalize(path: str, env: TableTennisWorker):
    venv = DummyVecEnv([lambda: env])
    vecnorm = VecNormalize.load(path, venv)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


def main():
    # ==================================================
    # Config & directories
    # ==================================================
    cfg = Config()
    prepare_experiment_directory(cfg)
    
    WORKER_DIR = os.path.join(cfg.logdir, "worker")
    MANAGER_DIR = os.path.join(cfg.logdir, "manager")
    os.makedirs(WORKER_DIR, exist_ok=True)
    os.makedirs(MANAGER_DIR, exist_ok=True)

    info_cb = InfoLoggerCallback()

    # ==================================================
    # 1) Train WORKER
    # ==================================================
    cfg.logdir = WORKER_DIR
    # worker_env = build_worker_vec(
    #     cfg=cfg,
    #     num_envs=cfg.num_envs,
    # )

    # worker_model = PPO(
    #     "MlpPolicy",
    #     worker_env,
    #     verbose=1,
    #     tensorboard_log=cfg.logdir,
    #     n_steps=1024,
    #     batch_size=256,
    #     learning_rate=3e-4,
    #     gamma=0.97,
    #     gae_lambda=cfg.ppo_lambda,
    #     clip_range=cfg.ppo_clip_range,
    #     n_epochs=cfg.ppo_epochs,
    #     max_grad_norm=cfg.ppo_max_grad_norm,
    #     policy_kwargs=dict(net_arch=[128, 128]),
    #     seed=cfg.seed,
    # )

    # # ---- Worker evaluation ----
    # eval_worker_env = build_worker_vec(cfg=cfg, num_envs=1)
    # eval_worker_env.training = False
    # eval_worker_env.norm_reward = False

    # eval_worker_cb = EvalCallback(
    #     eval_worker_env,
    #     best_model_save_path=os.path.join(cfg.logdir, "best"),
    #     log_path=os.path.join(cfg.logdir, "eval"),
    #     eval_freq=int(cfg.eval_freq//cfg.num_envs),
    #     n_eval_episodes=cfg.eval_episodes,
    #     deterministic=True,
    #     render=False,
    # )

    # # # ---- Worker video ----
    # video_worker_cb = VideoCallback(
    #     env_func=TableTennisWorker,
    #     env_args={"config": cfg},
    #     cfg=cfg,
    #     predict_fn=make_predict_fn(worker_model),
    # )

    # worker_model.learn(
    #     total_timesteps=100_000,
    #     callback=CallbackList([
    #         eval_worker_cb,
    #         info_cb,
    #         video_worker_cb,
    #     ]),
    # )

    # # # ---- Save WORKER (.pkl) ----
    worker_model_path = os.path.join(cfg.logdir, "worker_model.pkl")
    worker_env_path = os.path.join(cfg.logdir, "vecnormalize.pkl")
    # worker_model.save(worker_model_path)
    # worker_env.save(os.path.join(cfg.logdir, "vecnormalize.pkl"))

    # worker_env.close()
    # eval_worker_env.close()

    # ==================================================
    # 2) Train MANAGER
    # ==================================================
    cfg.logdir = MANAGER_DIR
    manager_env = build_manager_vec(
        cfg=cfg,
        num_envs=cfg.num_envs,
        worker_model_path=worker_model_path,
        worker_env_path=worker_env_path,
        decision_interval=cfg.episode_len // 10,
        max_episode_steps=cfg.episode_len,
        worker_model_loader=load_worker_model,
        worker_env_loader=load_worker_vecnormalize,
    )

    manager_model = PPO(
        "MlpPolicy",
        manager_env,
        verbose=1,
        tensorboard_log=cfg.logdir,
        n_steps=4096,
        batch_size=512,
        learning_rate=1e-4,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=cfg.ppo_clip_range,
        n_epochs=cfg.ppo_epochs,
        max_grad_norm=cfg.ppo_max_grad_norm,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=cfg.seed,
    )

    # ---- Manager evaluation ----
    eval_manager_env = build_manager_vec(
        cfg=cfg,
        num_envs=1,
        worker_model_path=worker_model_path,
        worker_env_path=worker_env_path,
        decision_interval=cfg.episode_len // 10,
        max_episode_steps=cfg.episode_len,
        worker_model_loader=load_worker_model,
        worker_env_loader=load_worker_vecnormalize,
    )

    eval_manager_cb = EvalCallback(
        eval_manager_env,
        best_model_save_path=os.path.join(cfg.logdir, "best"),
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=int(cfg.eval_freq//cfg.num_envs),
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )

    # ---- Manager video ----
    video_worker_env = load_worker_vecnormalize(
        worker_env_path,
        TableTennisWorker(cfg),
    )
    frozen_worker_model = load_worker_model(worker_model_path)

    video_manager_cb = VideoCallback(
        env_func=TableTennisManager,
        env_args={
            "worker_env": video_worker_env,
            "worker_model": frozen_worker_model,
            "config": cfg,
        },
        cfg=cfg,
        predict_fn=make_predict_fn(manager_model),
    )

    manager_model.learn(
        total_timesteps=100_000,
        callback=CallbackList([
            eval_manager_cb,
            info_cb,
            video_manager_cb,
        ]),
    )

    # ---- Save MANAGER (.pkl) ----
    manager_model.save(os.path.join(cfg.logdir, "manager_model.pkl"))

    manager_env.close()
    eval_manager_env.close()
    video_worker_env.close()


if __name__ == "__main__":
    main()
