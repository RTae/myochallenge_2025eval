# mppi_planner.py
import numpy as np
from myosuite.utils import gym as myogym
from pd_controller import MorphologyAwareController


def compute_tt_cost(obs_dict, info):
    ball_pos = obs_dict["ball_pos"]
    ball_vel = obs_dict["ball_vel"]
    paddle_pos = obs_dict["paddle_pos"]
    paddle_ori = obs_dict["paddle_ori"]
    reach_err = obs_dict["reach_err"]
    touching = obs_dict["touching_info"]  # [paddle, own, opp, ground, net, env]

    t_prediction = 0.15
    ball_future = ball_pos + t_prediction * ball_vel

    offset = np.array([-0.03, 0.0, 0.02])
    target_paddle = ball_future + offset

    reach_loss = np.sum((paddle_pos - target_paddle) ** 2)
    err_loss = np.sum(reach_err ** 2)

    hit_reward = 1.0 if touching[0] > 0.5 else 0.0
    hit_loss = -hit_reward

    desired_ori = np.array([0.0, 0.0, 1.0])
    ori_loss = np.sum((paddle_ori - desired_ori) ** 2)

    control_loss = 0.001

    cost = (
        3.0 * reach_loss +
        1.0 * err_loss +
        5.0 * hit_loss +
        0.5 * ori_loss +
        control_loss
    )
    return float(cost)


class MPPIPlanner:
    def __init__(self, env_id, horizon=10, pop=64, sigma=0.15, lam=1.0, workers=1, seed=42):
        self.env_id = env_id
        self.horizon = horizon
        self.pop = pop
        self.sigma = sigma
        self.lam = lam
        self.workers = workers
        self.rng = np.random.default_rng(seed)

    def _rollout_cost(self, q_target):
        env = myogym.make(self.env_id)
        obs, _ = env.reset()
        ctrl = MorphologyAwareController(env, kp=8.0, kd=1.5)

        total_cost = 0.0

        for _ in range(self.horizon):
            act = ctrl.compute_action(q_target)
            obs, _, terminated, truncated, info = env.step(act)

            obs_dict = env.get_obs_dict(obs)
            total_cost += compute_tt_cost(obs_dict, info)

            if terminated or truncated:
                break

        env.close()
        return total_cost

    def plan(self, base_qpos):
        n_dim = len(base_qpos)
        noise = self.rng.normal(size=(self.pop, n_dim))
        samples = base_qpos[None, :] + self.sigma * noise

        costs = np.zeros(self.pop, dtype=np.float32)
        for i in range(self.pop):
            costs[i] = self._rollout_cost(samples[i])

        c_min = np.min(costs)
        exp_arg = -(costs - c_min) / max(self.lam, 1e-6)
        weights = np.exp(exp_arg)
        weights_sum = np.sum(weights) + 1e-8
        z_star = (weights[:, None] * samples).sum(axis=0) / weights_sum

        return z_star
