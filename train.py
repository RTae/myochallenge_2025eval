import os
from loguru import logger
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
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


def make_policy_fn(planner, cfg):
    """
    The official MCP²-style MPC policy.
    Each environment (train/eval/video) gets its own controller.
    """

    state = {"step": 0, "z_star": None}

    def policy_fn(obs, policy_env):
        # Each callback env gets its own controller
        if not hasattr(policy_fn, "controller_dict"):
            policy_fn.controller_dict = {}

        if policy_env not in policy_fn.controller_dict:
            policy_fn.controller_dict[policy_env] = MorphologyAwareController(
                policy_env, kp=8.0, kd=1.5
            )

        ctrl = policy_fn.controller_dict[policy_env]

        # Replan every plan_internal steps
        if state["z_star"] is None or state["step"] % cfg.plan_internal == 0:
            # Use full observation (task-specific planning)
            obs_vec = obs if isinstance(obs, np.ndarray) else obs.get("obs", None)
            if obs_vec is None:
                obs_vec = policy_env.unwrapped.sim.data.qpos.copy()

            state["z_star"] = planner.plan(obs_vec)

        state["step"] += 1
        return ctrl.compute_action(state["z_star"])

    return policy_fn



def run(cfg: Config):

    exp_dir = next_exp_dir()
    cfg.logdir = exp_dir

    env = gym.make(cfg.env_id)
    env.reset(seed=cfg.seed)

    # Main controller only for stepping env
    controller = MorphologyAwareController(env, kp=8.0, kd=1.5)

    # MPPI planner
    planner = MPPIPlanner(
        env_id=cfg.env_id,
        horizon=cfg.horizon_H,
        pop=cfg.pop_size,
        sigma=cfg.es_sigma,
        lam=cfg.mppi_lambda,
        workers=cfg.cem_workers,
        seed=cfg.seed,
    )

    # MPC² global policy
    policy_fn = make_policy_fn(planner, cfg)

    # ---- Video callback ----
    video_cb = VideoCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        logdir=exp_dir,
        video_freq=cfg.video_freq,
        eval_episodes=cfg.eval_episodes,
    )
    video_cb.attach_predictor(policy_fn)

    # ---- Eval callback ----
    eval_cb = EvalCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        eval_freq=cfg.eval_freq,
        eval_episodes=3,
        logdir=exp_dir,
    )
    eval_cb.attach_predictor(policy_fn)
    eval_cb._init_callback()

    # ---- Train loop ----
    total_steps = 0
    episode = 0
    total_reward = 0.0

    env.reset()
    pbar = tqdm(total=cfg.total_timesteps)
    logger.info("Starting training...")

    z_star = env.unwrapped.sim.data.qpos.copy()

    while total_steps < cfg.total_timesteps:

        # MPPI replanning
        if total_steps % cfg.plan_internal == 0:
            obs_vec = env.get_obs_dict(env._get_obs()).get("obs",
                      env.unwrapped.sim.data.qpos.copy())
            z_star = planner.plan(obs_vec)

        # Low-level PD action
        act = controller.compute_action(z_star)
        obs, rew, terminated, truncated, info = env.step(act)

        total_reward += rew
        total_steps += 1
        pbar.update(1)

        video_cb.step(total_steps)
        eval_cb._on_step()

        if terminated or truncated:
            env.reset()
            total_reward = 0.0
            episode += 1

        if total_steps % cfg.train_log_freq == 0:
            logger.info(f"Step {total_steps} | Reward={total_reward:.2f}")

    pbar.close()
    eval_cb._on_training_end()
    planner.close()
    env.close()

    logger.info(f"Training complete. Final reward={total_reward:.2f}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = Config()
    run(cfg)
