import os
import numpy as np
from loguru import logger
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from myosuite.utils import gym

from config import Config
from callbacks.video_callback import VideoCallback
from callbacks.eval_callback import EvalCallback
from utils.helper import next_exp_dir

os.environ["MUJOCO_GL"] = "egl"
os.environ.pop("DISPLAY", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")


class MorphologyAwareController:
    def __init__(self, env, kp=10.0, kd=1.5):
        self.env = env
        self.kp = kp
        self.kd = kd

    def compute_action(self, q_target):
        q = self.env.unwrapped.sim.data.qpos.copy()
        qd = self.env.unwrapped.sim.data.qvel.copy()

        n = min(len(q_target), len(q), len(qd))
        q_target = q_target[:n]
        q = q[:n]
        qd = qd[:n]

        e = q_target - q
        u_raw = self.kp * e - self.kd * qd
        u = 1 / (1 + np.exp(-u_raw))

        act = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        act[:len(u)] = np.clip(u, 0.0, 1.0)
        return act


def rollout_worker(args):
    q_target, env_id, horizon = args

    import numpy as np
    from myosuite.utils import gym as myogym

    env = myogym.make(env_id)
    env.reset()

    ctrl = MorphologyAwareController(env)
    total_cost = 0.0

    for _ in range(horizon):
        u = ctrl.compute_action(q_target)
        obs, reward, terminated, truncated, info = env.step(u)

        q = env.unwrapped.sim.data.qpos.copy()
        n = min(len(q_target), len(q))
        e = q[:n] - q_target[:n]
        total_cost += float(np.dot(e, e))

        if terminated or truncated:
            break

    env.close()
    return total_cost


class ParallelCEMPlanner:
    def __init__(self, env_id, horizon=10, pop=64, elites=6, sigma=0.15, seed=42):
        self.env_id = env_id
        self.horizon = horizon
        self.pop = pop
        self.elites = elites
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def plan(self, base_qpos):
        n_dim = len(base_qpos)
        samples = self.rng.normal(base_qpos, self.sigma, size=(self.pop, n_dim))
        args = [(samples[i], self.env_id, self.horizon) for i in range(self.pop)]

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            costs = list(pool.map(rollout_worker, args))

        elite_idx = np.argsort(costs)[:self.elites]
        z_star = samples[elite_idx].mean(axis=0)
        return z_star


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
        seed=cfg.seed,
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
    logger.info("🚀 Starting training loop")

    while total_steps < max_steps:

        q_now = env.unwrapped.sim.data.qpos.copy()
        z_star = planner.plan(q_now)

        for _ in range(3):
            u = controller.compute_action(z_star)
            _, rew, terminated, truncated, _ = env.step(u)

            total_steps += 1
            total_reward += rew

            video_cb.step(total_steps)
            eval_cb._on_step()

            if terminated or truncated:
                logger.info(f"Episode {episode} finished | Reward = {total_reward:.2f}")
                env.reset()
                total_reward = 0.0
                episode += 1
                break

        if total_steps % 1000 == 0:
            logger.info(f"[Step {total_steps}] Running reward = {total_reward:.2f}")

    eval_cb._on_training_end()

    env.close()
    logger.info("🎉 Training complete.")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    cfg = Config()
    run(cfg)
