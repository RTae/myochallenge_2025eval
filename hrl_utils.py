# hrl_utils.py
import numpy as np


# ================================================================
#  FLATTEN WORKER OBSERVATION (base 424 dims)
# ================================================================
def flatten_myo_obs_worker(obs_dict):
    """
    Build a 1D float32 vector from the MyoSuite obs_dict.

    Keys (with shapes from your dump):
      time           (1,)
      pelvis_pos     (3,)
      body_qpos      (58,)
      body_qvel      (58,)
      ball_pos       (3,)
      ball_vel       (3,)
      paddle_pos     (3,)
      paddle_vel     (3,)
      paddle_ori     (4,)
      reach_err      (3,)
      palm_pos       (3,)
      palm_err       (3,)
      touching_info  (6,)
      act            (273,)

    Total: 424 dims
    """
    parts = [
        obs_dict["time"],
        obs_dict["pelvis_pos"],
        obs_dict["body_qpos"],
        obs_dict["body_qvel"],
        obs_dict["ball_pos"],
        obs_dict["ball_vel"],
        obs_dict["paddle_pos"],
        obs_dict["paddle_vel"],
        obs_dict["paddle_ori"],
        obs_dict["reach_err"],
        obs_dict["palm_pos"],
        obs_dict["palm_err"],
        obs_dict["touching_info"],
        obs_dict["act"],
    ]

    arrays = []
    for p in parts:
        arr = np.asarray(p, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        arrays.append(arr)

    return np.concatenate(arrays, axis=-1)


# ================================================================
#  FLATTEN MANAGER OBSERVATION (16D compact state)
# ================================================================
def flatten_myo_obs_manager(obs_dict):
    """
    Manager sees a compact state:
      ball_pos    (3,)
      ball_vel    (3,)
      paddle_pos  (3,)
      paddle_vel  (3,)
      reach_err   (3,)
      time        (1,)

    Total: 16 dims
    """
    parts = [
        obs_dict["ball_pos"],
        obs_dict["ball_vel"],
        obs_dict["paddle_pos"],
        obs_dict["paddle_vel"],
        obs_dict["reach_err"],
        obs_dict["time"],
    ]

    arrays = []
    for p in parts:
        arr = np.asarray(p, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        arrays.append(arr)

    return np.concatenate(arrays, axis=-1)


def build_worker_obs(obs_dict, goal, t_in_macro, cfg):
    """
    Worker obs:
      base        : 424
      paddle_vel : 3
      goal_vel   : 3
      phase      : 1
    Total = 431
    """
    base = flatten_myo_obs_worker(obs_dict)
    paddle_vel = obs_dict["paddle_vel"].astype(np.float32)
    goal = np.asarray(goal, dtype=np.float32)

    phase = np.array(
        [t_in_macro / max(1, cfg.high_level_period - 1)],
        dtype=np.float32
    )

    return np.concatenate(
        [base, paddle_vel, goal, phase],
        axis=-1
    ).astype(np.float32)



def worker_reward(obs_dict, hit, contact_force, dv):
    r = 0.0

    if hit:
        r += 10.0
        # optional: bonus if the hit produces bigger impulse (keeps it learnable)
        r += 0.5 * np.tanh(dv)          # 0..~0.5
        r += 0.5 * np.tanh(contact_force) 
        return float(r)

    # pre-contact shaping (CAUSAL): move paddle toward ball
    rel = np.array(obs_dict["ball_pos"], dtype=np.float32) - np.array(obs_dict["paddle_pos"], dtype=np.float32)
    paddle_vel = np.array(obs_dict["paddle_vel"], dtype=np.float32)

    approach = np.dot(paddle_vel, rel) / (np.linalg.norm(rel) + 1e-6)
    if approach > 0:
        r += 0.02 * approach

    # tiny penalty to avoid freezing
    r -= 0.01
    return float(r)

class HitDetector:
    def __init__(self, force_thr=1e-3, dv_thr=0.8):
        self.force_thr = force_thr
        self.dv_thr = dv_thr
        self.prev_ball_vel = None

    def reset(self, obs_dict):
        self.prev_ball_vel = np.array(obs_dict["ball_vel"], dtype=np.float32)

    def step(self, obs_dict):
        # --- 1) contact via touching_info ---
        ti = np.array(obs_dict.get("touching_info", []), dtype=np.float32).reshape(-1)
        contact_force = float(np.sum(np.abs(ti))) if ti.size > 0 else 0.0
        hit_by_force = contact_force > self.force_thr

        # --- 2) hit via ball velocity jump ---
        ball_vel = np.array(obs_dict["ball_vel"], dtype=np.float32)
        dv = 0.0 if self.prev_ball_vel is None else float(np.linalg.norm(ball_vel - self.prev_ball_vel))
        hit_by_dv = dv > self.dv_thr
        self.prev_ball_vel = ball_vel

        hit = hit_by_force or hit_by_dv
        return hit, contact_force, dv



# ================================================================
#  HIERARCHICAL PREDICTOR (for VideoCallback manager videos)
# ================================================================
def make_hierarchical_predictor(cfg, manager_model, worker_model):
    """
    Returns predict_fn(obs, env) for VideoCallback when visualizing the
    full HRL (manager + worker) directly on the raw MyoSuite environment.
    """

    def predict_fn(_ignored_sb3_obs, env_instance):
        # Raw MyoSuite dict
        obs_dict = env_instance.unwrapped.obs_dict

        # 1) Manager picks a goal
        m_obs = flatten_myo_obs_manager(obs_dict).reshape(1, -1)
        goal, _ = manager_model.predict(m_obs, deterministic=True)
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)
        goal = np.clip(goal, -cfg.goal_bound, cfg.goal_bound)

        # 2) Worker executes one low-level step toward that goal
        w_obs = build_worker_obs(
            obs_dict=obs_dict,
            goal=goal,
            t_in_macro=0,
            cfg=cfg,
        ).reshape(1, -1)

        action, _ = worker_model.predict(w_obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    return predict_fn
