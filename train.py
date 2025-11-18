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
        pop=cfg.pop_size,
        sigma=cfg.es_sigma,
        lam=cfg.mppi_lambda,
        workers=cfg.cem_workers,
        seed=cfg.seed,
    )

    PLAN_INTERVAL = 20

    z_star = env.unwrapped.sim.data.qpos.copy()

    def policy_fn(obs):
        return controller.compute_action(z_star)

    video_cb = VideoCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        logdir=exp_dir,
        video_freq=cfg.video_freq,
        eval_episodes=cfg.eval_episodes,
    )
    video_cb.attach_predictor(policy_fn)

    eval_cb = EvalCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        eval_freq=cfg.eval_freq,
        eval_episodes=3,
        logdir=exp_dir
    )
    eval_cb.attach_predictor(policy_fn)
    eval_cb._init_callback()

    total_steps = 0
    episode = 0
    total_reward = 0.0
    max_steps = cfg.total_timesteps

    env.reset()
    logger.info("Starting training...")
    pbar = tqdm(total=max_steps)

    while total_steps < max_steps:

        if total_steps % PLAN_INTERVAL == 0:
            q_now = env.unwrapped.sim.data.qpos.copy()
            z_star = planner.plan(q_now)

        act = controller.compute_action(z_star)
        _, rew, terminated, truncated, info = env.step(act)

        total_steps += 1
        total_reward += rew
        pbar.update(1)

        video_cb.step(total_steps)
        eval_cb._on_step()

        if terminated or truncated:
            env.reset()
            total_reward = 0.0
            episode += 1

            q_now = env.unwrapped.sim.data.qpos.copy()
            z_star = planner.plan(q_now)

        if total_steps % cfg.train_log_freq == 0:
            logger.info(f"Step={total_steps} | Total reward={total_reward:.2f}")

    pbar.close()
    eval_cb._on_training_end()
    planner.close()
    env.close()

    logger.info("Training complete.")
    logger.info(f"Final total reward: {total_reward:.2f}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = Config()
    run(cfg)
