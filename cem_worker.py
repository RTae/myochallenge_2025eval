import numpy as np
from myosuite.utils import gym
from pd_controller import MorphologyAwareController

_env = None
_ctrl = None
_horizon = None

def worker_init(env_id, horizon):
    global _env, _ctrl, _horizon

    _env = gym.make(env_id)
    _env.reset()

    _ctrl = MorphologyAwareController(_env)
    _horizon = horizon

def worker_rollout(q_target):
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
