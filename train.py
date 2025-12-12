# train_all.py
import os
import copy
import numpy as np

from loguru import logger
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from lattice.ppo.policies import LatticeRecurrentActorCriticPolicy
from lattice.sac.policies import LatticeSACPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from myosuite.utils import gym

from config import Config
from env_factory import build_vec_env
from callbacks.video_callback import VideoCallback
from callbacks.infologger_callback import InfoLoggerCallback
from hrl_utils import build_worker_obs, make_hierarchical_predictor
from stable_baselines3 import PPO as PPO_LOAD


def prepare_experiment_directory(cfg: Config):
    """
    Creates logs/exp1, exp2, ... automatically.
    Updates cfg.logdir to the new experiment folder.
    """
    base = cfg.logdir
    os.makedirs(base, exist_ok=True)

    existing = [d for d in os.listdir(base) if d.startswith("exp")]
    exp_nums = []
    for e in existing:
        try:
            exp_nums.append(int(e.replace("exp", "")))
        except Exception:
            pass

    next_id = 1 if not exp_nums else max(exp_nums) + 1
    exp_dir = os.path.join(base, f"exp{next_id}")
    os.makedirs(exp_dir, exist_ok=True)

    cfg.logdir = exp_dir
    logger.info(f"📁 Created new experiment folder: {exp_dir}")
    return exp_dir


# ============================================================
# TRAIN WORKER
# ============================================================
def train_worker(cfg: Config):
    logger.info("Training worker ...")

    worker_logdir = os.path.join(cfg.logdir, "worker")
    os.makedirs(worker_logdir, exist_ok=True)
    
    cfg.eval_mode = False
    env = build_vec_env(worker=True, cfg=cfg)

    cfg_eval = copy.deepcopy(cfg)
    cfg_eval.eval_mode = True
    eval_env = build_vec_env(worker=True, cfg=cfg_eval)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=worker_logdir,
        log_path=worker_logdir,
        eval_freq=cfg.eval_freq,
        deterministic=True,
        n_eval_episodes=cfg.eval_episodes,
    )

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

    # --- Lattice SAC ---
    # model = SAC(policy=LatticeSACPolicy,
    #     env=vec_env,
    #     device='auto',
    #     learning_rate=3e-4,
    #     buffer_size=300_000,
    #     learning_starts=10000,
    #     batch_size=256,
    #     tau=0.02,
    #     gamma=0.98,
    #     train_freq=(8, "step"),
    #     gradient_steps=8,
    #     action_noise=None,
    #     replay_buffer_class=None,
    #     ent_coef="auto",
    #     target_update_interval=1,
    #     target_entropy="auto",
    #     use_sde=False,
    #     sde_sample_freq=1,
    #     policy_kwargs=dict(
    #         **policy_kwargs,
    #         use_lattice=True,
    #         use_expln=True,
    #         log_std_init=0.0,
    #         # activation_fn=nn.GELU,
    #         std_clip=(1e-3, 1),
    #         expln_eps=1e-6,
    #         clip_mean=2.0,
    #         std_reg=0.0
    #     ),)
    
    # --- Lattice PPO ---
    # model = RecurrentPPO(policy=LatticeRecurrentActorCriticPolicy, 
    #     env=vec_env,
    #     device='auto',
    #     batch_size=32,
    #     n_steps=128,
    #     learning_rate=2.55673e-05,
    #     ent_coef=3.62109e-06,
    #     clip_range=0.3,
    #     gamma=0.99,
    #     gae_lambda=0.9,
    #     max_grad_norm=0.7,
    #     vf_coef=0.835671,
    #     n_epochs=10,
    #     use_sde=False,
    #     sde_sample_freq=1,
    #     policy_kwargs=dict(
    #         **policy_kwargs,
    #         use_lattice=True,
    #         use_expln=True,
    #         ortho_init=False,
    #         log_std_init=0.0,
    #         # activation_fn=nn.ReLU,
    #         std_clip=(1e-3, 10),
    #         expln_eps=1e-6,
    #         full_std=False,
    #         std_reg=0.0,
    #     ),)

    # --- Callbacks: Video + Evaluation ---
    eval_env = make_vec_env(cfg.env_id, cfg.seed + 999, 1,
                            wrapper_fn=lambda e: SubgoalRewardWrapper(e, subgoal_interval=getattr(cfg, "horizon_H", 20), intrinsic_coef=0.1))
    eval_env = VecMonitor(eval_env)

        # Use zero goal + phase=0 just to visualize
        zero_goal = np.zeros(cfg.goal_dim, dtype=np.float32)
        worker_obs = build_worker_obs(
            obs_dict=obs_dict,
            goal=zero_goal,
            t_in_macro=0,
            cfg=cfg,
        ).reshape(1, -1)

        action, _ = model.predict(worker_obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    video_cb = VideoCallback(cfg, mode="worker", predict_fn=worker_predict)
    info_cb = InfoLoggerCallback()

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList([eval_callback, video_cb, info_cb]),
    )

    worker_path = os.path.join(worker_logdir, "worker.zip")
    model.save(worker_path)
    logger.info(f"💾 Saved worker model → {worker_path}")

    env.close()
    eval_env.close()


# ============================================================
# TRAIN MANAGER
# ============================================================
def train_manager(cfg: Config):
    logger.info("Training manager ...")

    manager_logdir = os.path.join(cfg.logdir, "manager")
    os.makedirs(manager_logdir, exist_ok=True)

    worker_model_path = os.path.join(cfg.logdir, "worker", "worker.zip")
    worker_model_path = os.path.abspath(worker_model_path)

    if not os.path.exists(worker_model_path):
        raise FileNotFoundError(f"Expected worker model at: {worker_model_path}")

    env = build_vec_env(worker=False, cfg=cfg, worker_model_path=worker_model_path)
    eval_env = build_vec_env(worker=False, cfg=cfg, worker_model_path=worker_model_path)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=manager_logdir,
        log_path=manager_logdir,
        eval_freq=cfg.eval_freq,
        deterministic=True,
        n_eval_episodes=cfg.eval_episodes,
    )

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

    worker_model = PPO_LOAD.load(worker_model_path)
    hrl_predict = make_hierarchical_predictor(cfg, model, worker_model)

    video_cb = VideoCallback(cfg, mode="manager", predict_fn=hrl_predict)
    info_cb = InfoLoggerCallback()

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList([eval_callback, video_cb, info_cb]),
    )

    manager_path = os.path.join(manager_logdir, "manager.zip")
    model.save(manager_path)
    logger.info(f"💾 Saved manager model → {manager_path}")

    env.close()
    eval_env.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    base_cfg = Config()
    prepare_experiment_directory(base_cfg)

    worker_cfg = copy.deepcopy(base_cfg)
    manager_cfg = copy.deepcopy(base_cfg)

    worker_cfg.total_timesteps = 40_000_000
    worker_cfg.ppo_lr = 1e-4
    worker_cfg.ppo_gamma = 0.99

    manager_cfg.total_timesteps = 5_000_000
    manager_cfg.ppo_lr = 3e-4
    manager_cfg.ppo_gamma = 0.995
    manager_cfg.ppo_n_steps = 512
    manager_cfg.ppo_batch_size = 256

    train_worker(worker_cfg)
    train_manager(manager_cfg)

    logger.info("🎉 Training Complete!")
