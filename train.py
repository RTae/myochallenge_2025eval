import os
import numpy as np
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from myosuite.utils import gym

from config import Config
from callbacks.videoCallback import VideoCallback
from utils.helper import next_exp_dir

os.environ["MUJOCO_GL"] = "egl"
os.environ.pop("DISPLAY", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")


# ======================================================
#  Low-Level Controller (PD → muscle excitation)
# ======================================================
class MorphologyAwareController:
    """
    Simple joint-space PD controller mapping posture error to muscle excitation.
    Works for MyoSuite by constraining to [0,1].
    """
    def __init__(self, env, kp=10.0, kd=1.5):
        self.env = env
        self.kp = kp
        self.kd = kd

    def compute_action(self, q_target):
        q = self.env.sim.data.qpos.copy()
        qd = self.env.sim.data.qvel.copy()

        e = q_target - q
        u_raw = self.kp * e - self.kd * qd

        # MyoSuite actions MUST be in [0,1]
        u = 1 / (1 + np.exp(-u_raw))     # sigmoid

        return np.clip(u, 0.0, 1.0)


# ======================================================
#  Parallel CEM Planner
# ======================================================
class ParallelCEMPlanner:
    """
    Parallel CEM rollout planner.
    Works for high-dim MyoSuite envs (muscle-based).
    """
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
        z_star = samples[elite_idx].mean(axis=0)
        return z_star

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

    # --- Log directory ---
    exp_dir = next_exp_dir()
    cfg.logdir = exp_dir

    # --- Make env ---
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

    # --- Video logger ---
    video_cb = VideoCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        logdir=exp_dir,
        video_freq=cfg.video_freq,
        eval_episodes=cfg.eval_episodes,
        verbose=0,
    )

    # initialize callback
    video_cb._init_callback(model=None)
    video_cb._on_training_start(locals(), globals())

    done = False
    total_steps = 0
    episode = 0
    total_reward = 0

    obs, info = env.reset()

    max_steps = cfg.total_timesteps

    while total_steps < max_steps:

        q_now = env.sim.data.qpos.copy()

        # high-level posture target
        z_star = planner.plan(q_now)

        # execute 3 low-level steps toward target posture
        for _ in range(3):
            u = controller.compute_action(z_star)

            obs, rew, terminated, truncated, info = env.step(u)
            total_steps += 1
            total_reward += rew

            if terminated or truncated:
                logger.info(f"Episode {episode} done | Reward={total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
                episode += 1
                break

        if total_steps % 1000 == 0:
            logger.info(f"Step {total_steps} | Reward so far={total_reward:.2f}")

        video_cb._on_step()

    video_cb._on_training_end()
    env.close()
    logger.info("🎉 Finished Parallel MPC-like run.")


if __name__ == "__main__":
    cfg = Config()
    run(cfg)
