import numpy as np
import multiprocessing as mp
from cem_worker import worker_init, worker_rollout

class ParallelCEMPlanner:
    def __init__(self, env_id, horizon=10, pop=64, elites=6,
                 sigma=0.15, workers=8, seed=42):

        self.env_id = env_id
        self.horizon = horizon
        self.pop = pop
        self.elites = elites
        self.sigma = sigma
        self.workers = workers
        self.rng = np.random.default_rng(seed)

        self.pool = mp.Pool(
            processes=self.workers,
            initializer=worker_init,
            initargs=(self.env_id, self.horizon)
        )

    def plan(self, base_qpos):
        base_qpos = np.asarray(base_qpos, dtype=np.float32)
        n_dim = len(base_qpos)

        samples = self.rng.normal(base_qpos, self.sigma,
                                  size=(self.pop, n_dim))

        costs = self.pool.map(worker_rollout, samples)

        elite_idx = np.argsort(costs)[:self.elites]
        return samples[elite_idx].mean(axis=0)
