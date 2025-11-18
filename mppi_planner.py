import numpy as np
import multiprocessing as mp
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

    dist = float(np.linalg.norm(ball_pos - paddle_pos))
    reach_norm = float(np.linalg.norm(reach_err))
    height_penalty = max(0.0, 0.3 - float(ball_pos[2]))
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


# ---------- persistent worker globals ----------

_worker_env = None
_worker_ctrl = None
_worker_horizon = None
_worker_w_track = None
_worker_w_task = None


def _init_worker(env_id, horizon, kp, kd, w_track, w_task, base_seed):
    global _worker_env, _worker_ctrl, _worker_horizon
    global _worker_w_track, _worker_w_task

    _worker_env = gym.make(env_id)
    _worker_env.reset(seed=base_seed)
    _worker_ctrl = MorphologyAwareController(_worker_env, kp=kp, kd=kd)
    _worker_horizon = horizon
    _worker_w_track = w_track
    _worker_w_task = w_task


def _worker_rollout(q_target: np.ndarray) -> float:
    env = _worker_env
    ctrl = _worker_ctrl
    horizon = _worker_horizon
    w_track = _worker_w_track
    w_task = _worker_w_task

    total_cost = 0.0

    env.reset()  # fresh rollout each time
    for _ in range(horizon):
        act = ctrl.compute_action(q_target)
        obs, rew, terminated, truncated, info = env.step(act)

        q = env.unwrapped.sim.data.qpos.copy()
        n = min(len(q_target), len(q))
        track_cost = float(np.sum((q[:n] - q_target[:n]) ** 2))

        tt_cost = compute_tt_cost_from_obs(obs)

        total_cost += w_track * track_cost + w_task * tt_cost

        if terminated or truncated:
            break

    return total_cost


class MPPIPlanner:
    def __init__(
        self,
        env_id: str,
        horizon: int = 10,
        pop: int = 64,
        sigma: float = 0.1,
        lam: float = 1.0,
        workers: int | None = None,
        seed: int = 42,
        w_track: float = 0.1,
        w_task: float = 1.0,
        kp_ctrl: float = 8.0,
        kd_ctrl: float = 1.5,
    ):
        self.env_id = env_id
        self.horizon = horizon
        self.pop = pop
        self.sigma = sigma
        self.lam = lam
        self.rng = np.random.default_rng(seed)
        self.w_track = w_track
        self.w_task = w_task

        if workers is None or workers <= 0:
            workers = max(1, mp.cpu_count() // 2)
        self.workers = workers

        self._pool = mp.Pool(
            processes=self.workers,
            initializer=_init_worker,
            initargs=(env_id, horizon, kp_ctrl, kd_ctrl, w_track, w_task, seed),
        )

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def plan(self, base_qpos: np.ndarray) -> np.ndarray:
        n_dim = len(base_qpos)
        samples = self.rng.normal(base_qpos, self.sigma, size=(self.pop, n_dim))

        # parallel cost eval
        costs = np.array(self._pool.map(_worker_rollout, [s for s in samples]),
                         dtype=np.float64)

        beta = float(np.min(costs))
        weights = np.exp(-(costs - beta) / self.lam)
        weights_sum = float(np.sum(weights)) + 1e-8
        weights /= weights_sum

        z_star = np.sum(samples * weights[:, None], axis=0)
        return z_star
