import numpy as np
from myosuite.utils import gym


def compute_tt_cost(obs, info):
    ball_pos = obs["ball_pos"]
    ball_vel = obs["ball_vel"]
    paddle_pos = obs["paddle_pos"]
    paddle_ori = obs["paddle_ori"]
    reach_err = obs["reach_err"]
    touching = obs["touching_info"]

    # predict ball future 150ms
    t = 0.15
    ball_future = ball_pos + t * ball_vel

    offset = np.array([-0.03, 0.0, 0.02])
    target_paddle = ball_future + offset

    reach_loss = np.sum((paddle_pos - target_paddle)**2)
    err_loss = np.sum(reach_err**2)

    # contact reward (touching_info[0] = paddle contact)
    hit_reward = 1.0 if touching[0] > 0.5 else 0.0
    hit_loss = -hit_reward

    desired_ori = np.array([0, 0, 1])
    ori_loss = np.sum((paddle_ori - desired_ori)**2)

    cost = (
        3.0 * reach_loss +
        1.0 * err_loss +
        5.0 * hit_loss +
        0.5 * ori_loss +
        0.001
    )

    return float(cost)


class MPPIPlanner:
    def __init__(self, env_id, horizon, pop, sigma, lam, workers, seed):
        self.env_id = env_id
        self.horizon = horizon
        self.pop = pop
        self.sigma = sigma
        self.lam = lam
        self.workers = workers
        self.rng = np.random.default_rng(seed)

    def plan(self, obs_dict):

        base_qpos = obs_dict["body_qpos"]

        samples = self.rng.normal(
            base_qpos, self.sigma, size=(self.pop, len(base_qpos))
        )

        costs = np.zeros(self.pop)

        for i in range(self.pop):
            costs[i] = self._rollout_cost(samples[i])

        # MPPI weights
        beta = np.min(costs)
        weights = np.exp(-(costs - beta) / self.lam)
        weights /= np.sum(weights)

        z_star = np.sum(weights[:, None] * samples, axis=0)
        return z_star

    def _rollout_cost(self, q_target):

        env = gym.make(self.env_id)
        env.reset()

        from pd_controller import MorphologyAwareController
        ctrl = MorphologyAwareController(env)

        total_cost = 0.0

        for _ in range(self.horizon):
            act = ctrl.compute_action(q_target)
            obs, _, terminated, truncated, info = env.step(act)

            obs_dict = env.unwrapped.get_obs_dict(env.unwrapped.sim)
            total_cost += compute_tt_cost(obs_dict, info)

            if terminated or truncated:
                break

        env.close()
        return total_cost
