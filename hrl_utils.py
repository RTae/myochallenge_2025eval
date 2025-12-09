# hrl_utils.py
import numpy as np


# ================================================================
#  FLATTEN WORKER OBSERVATION (Low-level controller)
# ================================================================
def flatten_myo_obs_worker(obs_dict):
    """
    Build a 1D float32 vector for the worker.
    Uses only stable keys from obs_dict.
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
    Smaller observation for the manager.
    """
    parts = [
        obs_dict["ball_pos"],    # (3,)
        obs_dict["ball_vel"],    # (3,)
        obs_dict["paddle_pos"],  # (3,)
        obs_dict["paddle_vel"],  # (3,)
        obs_dict["reach_err"],   # (3,)
        obs_dict["time"],        # (1,)
    ]

    safe_parts = []
    for p in parts:
        arr = np.array(p, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        safe_parts.append(arr)

    return np.concatenate(safe_parts, axis=-1)


# ================================================================
#  WORKER INPUT DURING HRL EXECUTION
# ================================================================
def build_worker_obs(obs_dict, goal, t_in_macro, cfg):
    """
    Worker observation:
        [flattened_base_obs (424), goal (3), phase (1)]
    Total dims: 428

    NOTE:
      - For now, phase is always 0, and worker was trained with goal=0, phase=0.
      - Manager can later be upgraded to use real goals once worker is retrained.
    """
    base = flatten_myo_obs_worker(obs_dict)          # (424,)
    goal = np.asarray(goal, dtype=np.float32)        # (goal_dim=3,)
    phase = np.array([0.0], dtype=np.float32)        # keep constant for stability

    return np.concatenate([base, goal, phase], axis=-1).astype(np.float32)


# ================================================================
#  INTRINSIC REWARD FOR WORKER (OPTIONAL, not used yet)
# ================================================================
def intrinsic_reward(obs_dict, goal):
    """
    Worker reward:
        r_int = - || (paddle_pos - ball_pos) - goal ||
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
    Function for VideoCallback.predict_fn.

    VideoCallback will call:
        predict_fn(sb3_obs, env_instance)

    We IGNORE sb3_obs and always use env_instance.unwrapped.obs_dict.

    For now:
      - Manager is still running but worker ignores its goal.
      - We pass zero-goal and phase=0 to the worker, same as in training.
    """

    def predict_fn(_ignored_sb3_obs, env_instance):
        # Real MyoSuite obs dict
        obs_dict = env_instance.unwrapped.obs_dict

        # 1) Manager forward (just for completeness)
        mgr_obs = flatten_myo_obs_manager(obs_dict).reshape(1, -1)
        _goal, _ = manager_model.predict(mgr_obs, deterministic=True)

        # 2) Worker forward: must match training obs exactly
        zero_goal = np.zeros(cfg.goal_dim, dtype=np.float32)
        w_obs = build_worker_obs(
            obs_dict=obs_dict,
            goal=zero_goal,
            t_in_macro=0,
            cfg=cfg,
        ).reshape(1, -1)

        action, _ = worker_model.predict(w_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return action

    return predict_fn
