import os
import numpy as np
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from myosuite.utils import gym

from config import Config
from callbacks.video_callback import VideoCallback
from callbacks.eval_callback import EvalCallback
from utils.helper import next_exp_dir

os.environ["MUJOCO_GL"] = "egl"
os.environ.pop("DISPLAY", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")


# ======================================================
#  Low-Level Controller (PD → muscle excitation)
# ======================================================
class MorphologyAwareController:
    """Simple PD → muscle excitation controller."""
    def __init__(self, env, kp=10.0, kd=1.5):
        self.env = env
        self.kp = kp
        self.kd = kd

    def compute_action(self, q_target):
        q = self.env.sim.data.qpos.copy()
        qd = self.env.sim.data.qvel.copy()

        e = q_target - q
        u_raw = self.kp * e - self.kd * qd

        u = 1 / (1 + np.exp(-u_raw))
        return np.clip(u, 0.0, 1.0)


# ======================================================
#  Parallel CEM Planner
# ======================================================
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

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            costs = list(pool.map(self.evaluate_rollout, samples))

        elite_idx = np.argsort(costs)[:self.elites]
        return samples[elite_idx].mean(axis=0)

    def evaluate_rollout(self, q_target):
        import numpy as np
        from myosuite.utils import gym as myogym

        env = myogym.make(self.env_id)
        env.reset()

        ctrl = MorphologyAwareController(env)
        cost = 0.0

        for _ in range(self.horizon):
            u = ctrl.compute_action(q_target)
            obs, reward, terminated, truncated, info = env.step(u)

            q = env.sim.data.qpos.copy()
            e = q - q_target
            cost += np.dot(e, e)

            if terminated or truncated:
                break

        env.close()
        return cost


# ======================================================
#  Main RL Loop
# ======================================================
def run(cfg: Config):

    # Log directory
    exp_dir = next_exp_dir()
    cfg.logdir = exp_dir

    # Make env
    env = gym.make(cfg.env_id)
    env.reset(seed=cfg.seed)

    controller = MorphologyAwareController(env, kp=8.0, kd=1.5)
    planner = ParallelCEMPlanner(
        env_id=cfg.env_id,
        horizon=cfg.horizon_H,
        pop=cfg.es_batch * 8,
        elites=cfg.elites,
        sigma=cfg.es_sigma,
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
    eval_cb._init_callback(model=None)
    eval_cb._on_training_start(locals(), globals())

    total_steps = 0
    episode = 0
    total_reward = 0

    env.reset()
    max_steps = cfg.total_timesteps

    logger.info(f"Starting training")

    while total_steps < max_steps:

        q_now = env.sim.data.qpos.copy()
        z_star = planner.plan(q_now)

        for _ in range(3):
            u = controller.compute_action(z_star)
            _, rew, terminated, truncated, _ = env.step(u)

            total_steps += 1
            total_reward += rew

            video_cb.step(total_steps)
            eval_cb._on_step()

            if terminated or truncated:
                logger.info(f"Episode {episode} done | Reward={total_reward:.2f}")
                env.reset()
                total_reward = 0
                episode += 1
                break

        if total_steps % 1000 == 0:
            logger.info(f"Step {total_steps} | Reward so far={total_reward:.2f}")

    eval_cb._on_training_end()
    env.close()

    logger.info("Finished training.")


if __name__ == "__main__":
    cfg = Config()
    run(cfg)
