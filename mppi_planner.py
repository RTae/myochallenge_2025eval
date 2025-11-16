import numpy as np
from myosuite.utils import gym
from pd_controller import MorphologyAwareController


def parse_tt_obs(obs: np.ndarray):
    obs = np.asarray(obs).ravel()
    idx = 0

    pelvis_pos = obs[idx:idx+3]; idx += 3
    body_qpos = obs[idx:idx+58]; idx += 58
    body_vel  = obs[idx:idx+58]; idx += 58
    ball_pos  = obs[idx:idx+3];  idx += 3
    ball_vel  = obs[idx:idx+3];  idx += 3
    paddle_pos = obs[idx:idx+3]; idx += 3
    paddle_vel = obs[idx:idx+3]; idx += 3
    paddle_ori = obs[idx:idx+3]; idx += 3
    reach_err  = obs[idx:idx+3]; idx += 3
    muscle_act = obs[idx:idx+273]; idx += 273
    touching   = obs[idx:idx+6]; idx += 6

    return {
        "pelvis_pos": pelvis_pos,
        "body_qpos": body_qpos,
        "body_vel": body_vel,
        "ball_pos": ball_pos,
        "ball_vel": ball_vel,
        "paddle_pos": paddle_pos,
        "paddle_vel": paddle_vel,
        "paddle_ori": paddle_ori,
        "reach_err": reach_err,
        "muscle_act": muscle_act,
        "touching": touching,
    }


def compute_tt_cost_from_obs(obs: np.ndarray):
    d = parse_tt_obs(obs)
    ball_pos = d["ball_pos"]
    paddle_pos = d["paddle_pos"]
    reach_err = d["reach_err"]
    touching = d["touching"]

    # ball–paddle distance
    dist = float(np.linalg.norm(ball_pos - paddle_pos))

    # reach error norm
    reach_norm = float(np.linalg.norm(reach_err))

    # ball height penalty (encourage staying above some height, e.g. 0.3 m)
    height_penalty = max(0.0, 0.3 - float(ball_pos[2]))

    # contacts: touching[3]=ground, [4]=net, [5]=env
    ground = touching[3]
    net = touching[4]
    env_contact = touching[5]
    contact_penalty = 5.0 * float(ground + net) + 1.0 * float(env_contact)

    cost = (
        1.0 * dist +
        0.5 * reach_norm +
        2.0 * height_penalty +
        contact_penalty
    )
    return cost


class MPPIPlanner:
    def __init__(
        self,
        env_id: str,
        horizon: int = 10,
        pop: int = 64,
        sigma: float = 0.1,
        lam: float = 1.0,
        workers: int | None = None,  # kept for Config compatibility, unused
        seed: int = 42,
        w_track: float = 0.1,
        w_task: float = 1.0,
    ):
        self.env_id = env_id
        self.horizon = horizon
        self.pop = pop
        self.sigma = sigma
        self.lam = lam
        self.rng = np.random.default_rng(seed)
        self.w_track = w_track
        self.w_task = w_task

    def _rollout_cost(self, q_target: np.ndarray, seed: int) -> float:
        env = gym.make(self.env_id)
        env.reset(seed=seed)

        ctrl = MorphologyAwareController(env, kp=8.0, kd=1.5)
        total_cost = 0.0

        for _ in range(self.horizon):
            act = ctrl.compute_action(q_target)
            obs, rew, terminated, truncated, info = env.step(act)

            q = env.unwrapped.sim.data.qpos.copy()
            n = min(len(q_target), len(q))
            track_cost = float(np.sum((q[:n] - q_target[:n]) ** 2))

            tt_cost = compute_tt_cost_from_obs(obs)

            total_cost += self.w_track * track_cost + self.w_task * tt_cost

            if terminated or truncated:
                break

        env.close()
        return total_cost

    def plan(self, base_qpos: np.ndarray) -> np.ndarray:
        n_dim = len(base_qpos)
        samples = self.rng.normal(base_qpos, self.sigma, size=(self.pop, n_dim))

        costs = np.zeros(self.pop, dtype=np.float64)
        for i in range(self.pop):
            seed = int(self.rng.integers(0, 2**31 - 1))
            costs[i] = self._rollout_cost(samples[i], seed=seed)

        beta = np.min(costs)
        weights = np.exp(-(costs - beta) / self.lam)
        weights /= np.sum(weights) + 1e-8

        z_star = np.sum(samples * weights[:, None], axis=0)
        return z_star
