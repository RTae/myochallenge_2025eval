import numpy as np
import multiprocessing as mp
from myosuite.utils import gym
from pd_controller import MorphologyAwareController


# ==========================================================
#  OBS EXTRACTION (FIX: handles dict or array)
# ==========================================================
def _extract_obs_vector(obs):
    """Extract the true flat observation vector from MyoSuite."""
    if isinstance(obs, dict):
        if "obs" in obs:
            return np.asarray(obs["obs"], dtype=np.float32)

        # fallback – flatten all numeric entries
        vals = []
        for v in obs.values():
            if isinstance(v, (list, tuple, np.ndarray)):
                vals.extend(np.asarray(v).ravel())
        return np.asarray(vals, dtype=np.float32)

    return np.asarray(obs, dtype=np.float32)


# ==========================================================
#  PARSER (unchanged)
# ==========================================================
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


# ==========================================================
#  LONG-HORIZON TT COST
# ==========================================================
def compute_tt_cost(obs: np.ndarray) -> float:
    d = parse_tt_obs(obs)

    ball_pos    = d["ball_pos"]
    ball_vel    = d["ball_vel"]
    paddle_pos  = d["paddle_pos"]
    paddle_vel  = d["paddle_vel"]
    reach_err   = d["reach_err"]
    touching    = d["touching"]

    # --------------------------------------------------
    # 1) Basic shaping terms (smooth tracking)
    # --------------------------------------------------
    dist_ball_paddle = float(np.linalg.norm(ball_pos - paddle_pos))
    reach_norm       = float(np.linalg.norm(reach_err))

    # keep ball above a "floor" height (e.g. 0.2 m)
    floor_height   = 0.2
    height_floor_pen = max(0.0, floor_height - float(ball_pos[2]))

    # --------------------------------------------------
    # 2) Contact flags
    # touching: [paddle, own, opponent, ground, net, env]
    # --------------------------------------------------
    contact_paddle  = bool(touching[0])
    contact_ground  = bool(touching[3])
    contact_net     = bool(touching[4])
    contact_env     = bool(touching[5])

    # Strong penalties for bad contacts
    contact_penalty = (
        10.0 * float(contact_ground) +   # ball on ground
        8.0  * float(contact_net)    +   # ball hits net
        2.0  * float(contact_env)        # other environment collisions
    )

    # --------------------------------------------------
    # 3) Impact-aware return objective
    # --------------------------------------------------
    if contact_paddle:
        # ---- At impact: we want the ball to go toward opponent and up ----
        # Assume player on negative-x, opponent on positive-x → push ball +x
        desired_dir = np.array([1.0, 0.0, 0.5], dtype=np.float32)

        v = np.asarray(ball_vel, dtype=np.float32)
        speed = float(np.linalg.norm(v)) + 1e-8
        desired_dir_norm = float(np.linalg.norm(desired_dir)) + 1e-8

        # cosine similarity between actual and desired ball velocity
        cos_sim = float(np.dot(v, desired_dir) / (speed * desired_dir_norm))
        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        # good if cos_sim ≈ 1
        impact_dir_cost = 1.0 - cos_sim   # in [0, 2]

        # encourage non-trivial outgoing speed
        target_speed = 3.0
        speed_err = max(0.0, target_speed - speed)

        # make sure ball is not too low at impact
        net_clear_height = 0.7
        impact_height_pen = max(0.0, net_clear_height - float(ball_pos[2]))

        impact_cost = (
            4.0 * impact_dir_cost +
            0.5 * speed_err +
            3.0 * impact_height_pen
        )

    else:
        # ---- Pre-impact: shape paddle to meet future ball ----
        # Predict short-horizon ball position (e.g. 0.12 s into future)
        dt_predict = 0.12
        ball_future = ball_pos + ball_vel * dt_predict

        # distance between paddle and predicted ball position
        pred_dist = float(np.linalg.norm(ball_future - paddle_pos))

        # relative velocity: we want paddle to "move with" the ball near impact
        rel_vel = float(np.linalg.norm(ball_vel - paddle_vel))

        # encourage ball moving toward opponent side before impact
        # (slightly reward v_x > 0)
        vx = float(ball_vel[0])
        toward_opponent_cost = -0.2 * max(0.0, vx)  # negative = small reward

        impact_cost = (
            0.8 * pred_dist +
            0.1 * rel_vel +
            toward_opponent_cost
        )

    # --------------------------------------------------
    # 4) Total cost
    # --------------------------------------------------
    cost = (
        1.0 * dist_ball_paddle +   # keep paddle near ball
        0.3 * reach_norm       +   # keep paddle near task MPL
        1.0 * height_floor_pen +   # keep ball above floor
        contact_penalty        +   # avoid ground/net/env
        impact_cost                # long-horizon impact shaping
    )

    return float(cost)


# ==========================================================
#  PERSISTENT WORKER GLOBALS
# ==========================================================
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
    env.reset()

    for _ in range(horizon):
        act = ctrl.compute_action(q_target)
        obs, _, terminated, truncated, _ = env.step(act)

        q = env.unwrapped.sim.data.qpos.copy()
        n = min(len(q_target), len(q))
        track_cost = np.sum((q[:n] - q_target[:n]) ** 2)

        tt_cost = compute_tt_cost(obs)

        total_cost += w_track * track_cost + w_task * tt_cost

        if terminated or truncated:
            break

    return float(total_cost)


# ==========================================================
#  MPPI PLANNER
# ==========================================================
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

        costs = np.array(self._pool.map(_worker_rollout, samples),
                         dtype=np.float64)

        beta = float(np.min(costs))
        weights = np.exp(-(costs - beta) / self.lam)
        weights /= np.sum(weights) + 1e-8

        return np.sum(samples * weights[:, None], axis=0)
