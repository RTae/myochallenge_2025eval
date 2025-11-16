import os
from loguru import logger
import multiprocessing as mp
from tqdm import tqdm
from myosuite.utils import gym

from config import Config
from pd_controller import MorphologyAwareController
from mppi_planner import MPPIPlanner
from callbacks.video_callback import VideoCallback
from callbacks.eval_callback import EvalCallback
from utils.helper import next_exp_dir

os.environ["MUJOCO_GL"] = "egl"
os.environ.pop("DISPLAY", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")


def run(cfg: Config):

    exp_dir = next_exp_dir()
    cfg.logdir = exp_dir

    env = gym.make(cfg.env_id)
    env.reset(seed=cfg.seed)

    controller = MorphologyAwareController(env, kp=8.0, kd=1.5)

    planner = MPPIPlanner(
        env_id=cfg.env_id,
        horizon=cfg.horizon_H,
        pop=cfg.es_batch * 8,
        sigma=cfg.es_sigma,
        lam=getattr(cfg, "mppi_lambda", 1.0),
        workers=cfg.cem_workers,
        seed=cfg.seed,
    )

    # -------------------------------
    # MPC² Policy usable for eval + video
    # -------------------------------
    def mpc2_policy(obs, policy_env):
        """General policy: uses the *evaluation or video env*."""
        if not hasattr(mpc2_policy, "step"):
            mpc2_policy.step = 0
            mpc2_policy.z_star = policy_env.unwrapped.sim.data.qpos.copy()

        if mpc2_policy.step % 20 == 0:
            q_now = policy_env.unwrapped.sim.data.qpos.copy()
            mpc2_policy.z_star = planner.plan(q_now)

        mpc2_policy.step += 1
        return controller.compute_action(mpc2_policy.z_star)

    # -------------------------------
    # CALLBACKS
    # -------------------------------
    video_cb = VideoCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        logdir=exp_dir,
        video_freq=cfg.video_freq,
        eval_episodes=cfg.eval_episodes,
    )
    video_cb.attach_predictor(mpc2_policy)

    eval_cb = EvalCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        eval_freq=cfg.eval_freq,
        eval_episodes=3,
        logdir=exp_dir
    )
    eval_cb.attach_predictor(mpc2_policy)
    eval_cb._init_callback()
    eval_cb._on_training_start()

    # --------------------------------
    # TRAIN LOOP
    # --------------------------------
    total_steps = 0
    episode = 0
    total_reward = 0
    max_steps = cfg.total_timesteps

    logger.info("Starting training...")
    pbar = tqdm(total=max_steps)

    env.reset()

    while total_steps < max_steps:

        if total_steps % 20 == 0:
            q_now = env.unwrapped.sim.data.qpos.copy()
            z_star = planner.plan(q_now)

        act = controller.compute_action(z_star)
        _, rew, terminated, truncated, _ = env.s
