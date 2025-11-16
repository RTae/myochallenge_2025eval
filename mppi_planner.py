import numpy as np
import multiprocessing as mp
from myosuite.utils import gym
from pd_controller import MorphologyAwareController

_env = None
_ctrl = None
_horizon = None


def _worker_init(env_id, horizon):
    global _env, _ctrl, _horizon

    _env = gym.make(env_id)
    _env.reset()

    _ctrl = MorphologyAwareController(_env)
    _horizon = horizon


def _worker_rollout(q_target):
    global _env, _ctrl, _horizon

    _env.reset()
    q_target = np.asarray(q_target, dtype=np.float32)

    total_cost = 0.0

    for _ in range(_horizon):
        u = _ctrl.compute_action(q_target)
        _, _, terminated, truncated, _ = _env.step(u)

        q = _env.unwrapped.sim.data.qpos.copy()
        n = min(len(q_target), len(q))
        e = q[:n] - q_target[:n]
        total_cost += float(np.dot(e, e))

        if terminated or truncated:
            break

    return total_cost


class MPPIPlanner:
    def __init__(
        self,
        env_id: str,
        horizon: int = 10,
        pop: int = 64,
        sigma: float = 0.15,
        lam: float = 1.0,
        workers: int = 8,
        seed: int = 42,
    ):
        self.env_id = env_id
        self.horizon = horizon
        self.pop = pop
        self.sigma = sigma
        self.lam = lam
        self.workers = workers
        self.rng = np.random.default_rng(seed)

        self.pool = mp.Pool(
            processes=self.workers,
            initializer=_worker_init,
            initargs=(self.env_id, self.horizon),
        )

    def plan(self, base_qpos):
        base_qpos = np.asarray(base_qpos, dtype=np.float32)
        n_dim = len(base_qpos)

        samples = self.rng.normal(base_qpos, self.sigma, size=(self.pop, n_dim))

        costs = self.pool.map(_worker_rollout, list(samples))
        costs = np.asarray(costs, dtype=np.float32)

        c_min = costs.min()
        weights = np.exp(-(costs - c_min) / self.lam)
        weights_sum = weights.sum() + 1e-8

        z_star = (weights[:, None] * samples).sum(axis=0) / weights_sum
        return z_star
