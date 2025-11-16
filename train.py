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

        u = 1 / (1 + np.exp(-u_raw))   # sigmoid → [0,1]
        return np.clip(u, 0.0, 1.0)


# ======================================================
#  MP-SAFE TOP LEVEL ROLLOUT WORKER
# ======================================================
def rollout_worker(args):
    """
    Worker function for parallel CEM rollouts.
    Must be OUTSIDE the class for multiprocessing to work.
    """
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

        q = env.sim.data.qpos.copy()
        e = q - q_target
        total_cost += float(np.dot(e, e))

        if terminated or truncated:
            break

    env.close()
    return total_cost


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

        # Sample around current posture
        samples = self.rng.normal(base_qpos, self.sigma, size=(self.pop, n_dim))

        # Package args for multiprocessing
        args = [(samples[i], self.env_id, self.horizon) for i in range(self.pop)]

        # Run rollouts in parallel
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            costs = list(pool.map(rollout_worker, args))

        elite_idx = np.argsort(costs)[:self.elites]
        z_star = samples[elite_idx].mean(axis=0)
        return z_star


# ======================================================
#  Main RL Loop
# ======================================================
def run(cfg: Config):

    # Create experiment directory
    exp_dir = next_exp_dir()
    cfg.logdir = exp_dir

    # Create main env
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

    # ------------------------------
    # Video callback
    # ------------------------------
    video_cb = VideoCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        logdir=exp_dir,
        video_freq=cfg.video_freq,
        eval_episodes=cfg.eval_episodes,
        verbose=0,
    )

    # ------------------------------
    # Eval callback
    # ------------------------------
    eval_cb = EvalCallback(
        env_id=cfg.env_id,
        seed=cfg.seed,
        eval_freq=cfg.eval_freq,
        eval_episodes=3,
        logdir=exp_dir
    )
    eval_cb._init_callback()
    eval_cb._on_training_start()

    # Training state
    total_steps = 0
    episode = 0
    total_reward = 0
    max_steps = cfg.total_timesteps

    env.reset()
    logger.info("🚀 Starting training loop")

    # ==================================================
    #  MAIN TRAINING LOOP
    # ==================================================
    while total_steps < max_steps:

        q_now = env.sim.data.qpos.copy()

        # High-level plan
        z_star = planner.plan(q_now)

        # Execute 3 low-level steps toward posture
        for _ in range(3):
            u = controller.compute_action(z_star)
            _, rew, terminated, truncated, _ = env.step(u)

            total_steps += 1
            total_reward += rew

            # Callbacks
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

    # End callbacks
    eval_cb._on_training_end()

    env.close()
    logger.info("🎉 Training complete.")


# ======================================================
#  Entry Point
# ======================================================
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)

    cfg = Config()
    run(cfg)
