# hrl_utils.py
import numpy as np


# ================================================================
#  FLATTEN WORKER OBSERVATION (Low-level controller)
# ================================================================
def flatten_myo_obs_worker(obs_dict):
    """
    Build a 1D float32 vector for the worker (base obs only).

    Uses the following fields:

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

    Total = 424 dims.

    NOTE: padde_ori_err (4,) is intentionally excluded to keep
    the obs compact and stable.
    """
    parts = [
        obs_dict["time"],          # (1,)
        obs_dict["pelvis_pos"],    # (3,)
        obs_dict["body_qpos"],     # (58,)
        obs_dict["body_qvel"],     # (58,)
        obs_dict["ball_pos"],      # (3,)
        obs_dict["ball_vel"],      # (3,)
        obs_dict["paddle_pos"],    # (3,)
        obs_dict["paddle_vel"],    # (3,)
        obs_dict["paddle_ori"],    # (4,)
        # obs_dict["padde_ori_err"],  # (4,)  <-- intentionally excluded
        obs_dict["reach_err"],     # (3,)
        obs_dict["palm_pos"],      # (3,)
        obs_dict["palm_err"],      # (3,)
        obs_dict["touching_info"], # (6,)
        obs_dict["act"],           # (273,)
    ]

    safe_parts = []
    for p in parts:
        arr = np.array(p, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        safe_parts.append(arr)

    return np.concatenate(safe_parts, axis=-1)


# ================================================================
#  FLATTEN MANAGER OBSERVATION (High-level controller)
# ================================================================
def flatten_myo_obs_manager(obs_dict):
    """
    Compact manager observation:

        ball_pos   (3,)
        ball_vel   (3,)
        paddle_pos (3,)
        paddle_vel (3,)
        reach_err  (3,)
        time       (1,)

    Total = 16 dims.
    """
    parts = [
        obs_dict["ball_pos"],
        obs_dict["ball_vel"],
        obs_dict["paddle_pos"],
        obs_dict["paddle_vel"],
        obs_dict["reach_err"],
        obs_dict["time"],
    ]

    safe_parts = []
    for p in parts:
        arr = np.array(p, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        safe_parts.append(arr)

    return np.concatenate(safe_parts, axis=-1)


# ================================================================
#  WORKER INPUT DURING HRL EXECUTION / TRAINING
# ================================================================
def build_worker_obs(obs_dict, goal, t_in_macro, cfg):
    """
    Build the *goal-conditioned* worker observation:

        worker_obs = [ base_obs (424), goal (goal_dim=3), phase (1) ]

    - base_obs: flattened low-level state
    - goal:     manager's 3D goal vector
    - phase:    scalar in [0,1], position within the macro step

    Total dims = 424 + cfg.goal_dim + 1  (with goal_dim=3 → 428).
    """
    base = flatten_myo_obs_worker(obs_dict)          # (424,)
    goal = np.asarray(goal, dtype=np.float32)        # (goal_dim,)
    phase = np.array(
        [t_in_macro / max(1, cfg.high_level_period - 1)],
        dtype=np.float32
    )                                                # (1,)

    return np.concatenate([base, goal, phase], axis=-1).astype(np.float32)


# ================================================================
#  INTRINSIC REWARD FOR WORKER (optional helper)
# ================================================================
def intrinsic_reward(obs_dict, goal):
    """
    Intrinsic reward for worker:

        r_int = - || (paddle_pos - ball_pos) - goal ||

    You can mix this with env reward if you want:
        r_total = r_env + alpha * r_int
    """
    ball = obs_dict["ball_pos"]
    paddle = obs_dict["paddle_pos"]
    offset = paddle - ball
    err = np.linalg.norm(offset - goal)
    return -float(err)


# ================================================================
#  HIERARCHICAL PREDICTOR (for VideoRecorder)
# ================================================================
def make_hierarchical_predictor(cfg, manager_model, worker_model):
    """
    Returns a predict_fn for VideoRecorder:

        VideoRecorder will call:
            predict_fn(sb3_obs, env)

        We IGNORE sb3_obs and always read `env.unwrapped.obs_dict`
        from MyoSuite, then:

            - Manager gets 16D manager obs → goal (3D)
            - Worker gets 428D goal-conditioned obs (base + goal + phase)

        Phase is set to 0 in videos (start of macro-step).
    """

    def predict_fn(_ignored_sb3_obs, env_instance):
        # 1) Get real MyoSuite dict obs
        obs_dict = env_instance.unwrapped.obs_dict

        # 2) Manager proposes a high-level goal
        m_obs = flatten_myo_obs_manager(obs_dict).reshape(1, -1)
        goal, _ = manager_model.predict(m_obs, deterministic=True)
        goal = goal.astype(np.float32).flatten()  # (goal_dim,)

        # 3) Worker predicts low-level action using goal-conditioned obs
        w_obs = build_worker_obs(
            obs_dict=obs_dict,
            goal=goal,
            t_in_macro=0,  # phase=0 at beginning of macro for video
            cfg=cfg,
        ).reshape(1, -1)

        action, _ = worker_model.predict(w_obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    return predict_fn
