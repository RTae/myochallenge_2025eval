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


# ================================================================
#  WORKER OBS WITH GOAL + PHASE (428 dims)
# ================================================================
def build_worker_obs(obs_dict, goal, t_in_macro, cfg):
    """
    Worker observation during training and HRL:
      base   = flatten_myo_obs_worker (424)
      goal   = (3,)
      phase  = (1,)  in [0,1]

    Total: 424 + 3 + 1 = 428
    """
    base = flatten_myo_obs_worker(obs_dict)
    goal = np.asarray(goal, dtype=np.float32)

    # Phase in [0, 1]
    denom = max(1, cfg.high_level_period - 1)
    phase = np.array([t_in_macro / denom], dtype=np.float32)

    return np.concatenate([base, goal, phase], axis=-1).astype(np.float32)


# ================================================================
#  INTRINSIC REWARD FOR WORKER
# ================================================================
def intrinsic_reward(obs_dict, goal):
    """
    Intrinsic reward: how close paddle-ball offset is to the desired goal.
      offset = paddle_pos - ball_pos
      r_int  = - || offset - goal ||
    """
    ball = obs_dict["ball_pos"]
    paddle = obs_dict["paddle_pos"]
    offset = paddle - ball  # current paddle relative to ball

    err = np.linalg.norm(offset - goal)
    return -float(err)


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
