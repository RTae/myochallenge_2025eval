import os
from loguru import logger
import multiprocessing as mp
from tqdm import tqdm
from myosuite.utils import gym

from config import Config
from pd_controller import MorphologyAwareController
from cem_planner import ParallelCEMPlanner
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

    planner = ParallelCEMPlanner(
        env_id=cfg.env_id,
        horizon=cfg.horizon_H,
        pop=cfg.es_batch * 8,
        elites=cfg.elites,
        sigma=cfg.es_sigma,
        workers=cfg.cem_workers,
        seed=cfg.seed
    )

    video_cb = VideoCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        logdir=exp_dir,
        video_freq=cfg.video_freq,
        eval_episodes=cfg.eval_episodes,
        verbose=0,
    )

    eval_cb = EvalCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        eval_freq=cfg.eval_freq,
        eval_episodes=3,
        logdir=exp_dir
    )
    eval_cb._init_callback()
    eval_cb._on_training_start()

    total_steps = 0
    episode = 0
    total_reward = 0
    max_steps = cfg.total_timesteps

    env.reset()

    logger.info("Starting training...")
    
    pbar = tqdm(total=max_steps)

    while total_steps < max_steps:

        q_now = env.unwrapped.sim.data.qpos.copy()
        z_star = planner.plan(q_now)

        for _ in range(3):
            act = controller.compute_action(z_star)
            _, rew, terminated, truncated, _ = env.step(act)

            total_steps += 1
            total_reward += rew
            pbar.update(1)

            video_cb.step(total_steps)
            eval_cb._on_step()

            if terminated or truncated:
                env.reset()
                total_reward = 0
                episode += 1
                break

        if(total_steps % 10000 == 0):
            logger.info(f"Total reward: {total_reward}")

    pbar.close()
    eval_cb._on_training_end()
    env.close()

    logger.info("Training complete.")
    logger.info(f"Total reward: {total_reward}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    cfg = Config()
    run(cfg)
