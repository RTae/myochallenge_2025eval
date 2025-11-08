import os
import numpy as np
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from myosuite.utils import gym

from config import Config
from utils.callbacks import VideoCallback
from utils.helper import next_exp_dir

os.environ["MUJOCO_GL"] = "egl"
os.environ.pop("DISPLAY", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")

class MorphologyAwareController:
    """Simple joint-space PD controller mapping posture error to muscle activation."""
    def __init__(self, env, kp=10.0, kd=2.0):
        self.env = env
        self.kp = kp
        self.kd = kd

    def compute_action(self, q_target):
        q = self.env.sim.data.qpos.copy()
        qd = self.env.sim.data.qvel.copy()
        e_q = q_target - q
        u = self.kp * e_q - self.kd * qd
        u = np.tanh(u)
        return np.clip(u, 0.0, 1.0)


class ParallelCEMPlanner:
    """Sampling-based high-level posture planner with parallel rollout evaluation."""
    def __init__(self, env_id, horizon=10, pop=64, elites=6, sigma=0.15, seed=42):
        self.env_id = env_id
        self.horizon = horizon
        self.pop = pop
        self.elites = elites
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def plan(self, base_qpos):
        """Parallel rollout-based planning using multiple processes."""
        samples = self.rng.normal(base_qpos, self.sigma, size=(self.pop, len(base_qpos)))

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            costs = list(pool.map(self.evaluate_rollout, samples))

        elite_idx = np.argsort(costs)[:self.elites]
        z_star = samples[elite_idx].mean(axis=0)
        return z_star

    def evaluate_rollout(self, q_target):
        """Each worker creates its own MyoSuite environment (MuJoCo not thread-safe)."""
        import numpy as np
        from myosuite.utils import gym as myogym

        env = myogym.make(self.env_id)
        env.reset()
        ctrl = MorphologyAwareController(env)
        cost = 0.0

        for _ in range(self.horizon):
            u = ctrl.compute_action(q_target)
            obs, rew, done, info = env.step(u)
            q = env.sim.data.qpos.copy()
            e = q - q_target
            cost += np.dot(e, e)
            if done:
                break

        env.close()
        return cost


def run(cfg: Config):
    # --- Setup logs ---
    exp_dir = next_exp_dir()
    cfg.logdir = exp_dir

    # --- Make env ---
    env = gym.make(cfg.env_id)
    env.reset(seed=cfg.seed)

    # --- Controller + Planner ---
    controller = MorphologyAwareController(env, kp=10.0, kd=2.0)
    planner = ParallelCEMPlanner(
        env_id=cfg.env_id,
        horizon=cfg.horizon_H,
        pop=cfg.es_batch * 8,
        elites=max(4, cfg.es_batch // 2),
        sigma=cfg.es_sigma,
        seed=cfg.seed,
    )

    # Video logging ---
    video_cb = VideoCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        logdir=exp_dir,
        video_freq=cfg.video_freq,
        eval_episodes=cfg.eval_episodes,
        verbose=0,
    )

    _ = env.reset()
    done = False
    total_steps = 0
    episode = 0
    total_reward = 0.0
    max_steps = cfg.total_timesteps

    # Initialize video callback
    video_cb._init_callback(model=None)
    video_cb._on_training_start(locals(), globals())

    while total_steps < max_steps:
        q_now = env.sim.data.qpos.copy()
        z_star = planner.plan(q_now)

        for _ in range(3):
            u = controller.compute_action(z_star)
            _, rew, done, info = env.step(u)
            total_reward += rew
            total_steps += 1

            if done:
                logger.info(f"Episode {episode} done | Step={total_steps} | Reward={total_reward:.2f}")
                _ = env.reset()
                total_reward = 0.0
                episode += 1
                video_cb._on_step()
                break

        if total_steps % 200 == 0:
            logger.info(f"Step {total_steps:6d} | Partial reward={total_reward:.2f}")

    video_cb._on_training_end()
    env.close()
    logger.info("✅ Finished parallel MPC² run.")


# =====================================================
#  Entry point
# =====================================================
if __name__ == "__main__":
    cfg = Config()
    run(cfg)
