import os
import numpy as np
import multiprocessing as mp
from myosuite.utils import gym

# --------------------------------
# Quaternion → forward direction
# --------------------------------
def quat_to_forward(q):
    """Convert quaternion [w, x, y, z] to forward (local z-axis) direction."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)],
    ])
    return R[:, 2]   # local z-axis (paddle face normal)


# --------------------------------
# Task-specific Table Tennis cost
# --------------------------------
def compute_tt_cost(obs, info):
    ball_pos = obs["ball_pos"]
    ball_vel = obs["ball_vel"]
    paddle_pos = obs["paddle_pos"]

    t = 0.15
    ball_future = ball_pos + t * ball_vel

    offset = np.array([-0.03, 0.0, 0.02])
    target_paddle = ball_future + offset

    reach_loss = np.sum((paddle_pos - target_paddle) ** 2)
    err_loss = np.sum(obs["reach_err"] ** 2)

    touching = obs["touching_info"]
    hit_reward = 1.0 if touching[0] > 0.5 else 0.0
    hit_loss = -hit_reward

    paddle_quat = obs["paddle_ori"]          # shape (4,)
    paddle_forward = quat_to_forward(paddle_quat)
    desired_forward = np.array([0.0, 0.0, 1.0])
    ori_loss = np.sum((paddle_forward - desired_forward) ** 2)

    cost = (
        3.0 * reach_loss +
        1.0 * err_loss +
        5.0 * hit_loss +
        0.5 * ori_loss +
        1e-3
    )

    return float(cost)


# =========================
# Worker globals
# =========================
_WORKER_ENV = None
_WORKER_CTRL = None
_WORKER_H = None


def _worker_init(env_id, horizon, base_seed):
    global _WORKER_ENV, _WORKER_CTRL, _WORKER_H
    from pd_controller import MorphologyAwareController
    from myosuite.utils import gym as myogym

    worker_seed = base_seed + (os.getpid() % 10000)

    env = myogym.make(env_id)
    env.reset(seed=worker_seed)

    ctrl = MorphologyAwareController(env)

    _WORKER_ENV = env
    _WORKER_CTRL = ctrl
    _WORKER_H = horizon


def _worker_rollout(q_target):
    global _WORKER_ENV, _WORKER_CTRL, _WORKER_H

    env = _WORKER_ENV
    ctrl = _WORKER_CTRL
    H = _WORKER_H

    total_cost = 0.0
    env.reset()

    for _ in range(H):
        act = ctrl.compute_action(q_target)
        obs, _, terminated, truncated, info = env.step(act)

        obs_dict = env.unwrapped.get_obs_dict(env.unwrapped.sim)
        total_cost += compute_tt_cost(obs_dict, info)

        if terminated or truncated:
            break

    return total_cost


# =========================
# MPPI Planner (MCP² high level)
# =========================
class MPPIPlanner:
    def __init__(self, env_id, horizon, pop, sigma, lam, workers, seed):
        self.env_id = env_id
        self.horizon = horizon
        self.pop = pop
        self.sigma = sigma
        self.lam = lam
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        ctx = mp.get_context("spawn")
        cpu_cnt = ctx.cpu_count() or 1
        self.workers = min(workers, cpu_cnt) if workers is not None else cpu_cnt
        if self.workers <= 0:
            self.workers = 1

        self.pool = ctx.Pool(
            processes=self.workers,
            initializer=_worker_init,
            initargs=(self.env_id, self.horizon, self.seed),
        )

    def plan(self, obs_dict):
        base_qpos = obs_dict["body_qpos"]
        samples = self.rng.normal(
            base_qpos, self.sigma, size=(self.pop, len(base_qpos))
        )

        costs = np.array(self.pool.map(_worker_rollout, samples))

        beta = np.min(costs)
        weights = np.exp(-(costs - beta) / self.lam)
        weights /= np.sum(weights)

        z_star = np.sum(weights[:, None] * samples, axis=0)
        return z_star

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
