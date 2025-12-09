# hrl_utils.py
import numpy as np


# ================================================================
#  FLATTEN WORKER OBSERVATION (Low-level controller)
# ================================================================
def flatten_myo_obs_worker(obs_dict):
    """
    Build a 1D float32 vector for the worker.
    Uses full proprioception + task info.
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
        if arr.ndim == 0:  # scalar → (1,)
            arr = arr.reshape(1)
        safe_parts.append(arr)

    return np.concatenate(safe_parts, axis=-1)  # (428,)


# ================================================================
#  FLATTEN MANAGER OBSERVATION (High-level controller)
# ================================================================
def flatten_myo_obs_manager(obs_dict):
    """
    Manager sees a compact, task-focused state.
    """
    parts = [
        obs_dict["ball_pos"],      # (3,)
        obs_dict["ball_vel"],      # (3,)
        obs_dict["paddle_pos"],    # (3,)
        obs_dict["paddle_vel"],    # (3,)
        obs_dict["reach_err"],     # (3,)
        obs_dict["time"],          # (1,)
    ]

    safe_parts = []
    for p in parts:
        arr = np.array(p, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        safe_parts.append(arr)

    return np.concatenate(safe_parts, axis=-1).astype(np.float32)  # (16,)


# ================================================================
#  WORKER INPUT DURING HRL EXECUTION
# ================================================================
def build_worker_obs(obs_dict, goal, t_in_macro, cfg):
    """
    Worker observation during HRL:
        [ flatten_myo_obs_worker (428),
          goal (goal_dim = 3),
          phase (1) ]

    Total dims: 428 + goal_dim(3) + 1 = 432
    """
    base = flatten_myo_obs_worker(obs_dict)                  # (428,)
    goal = np.asarray(goal, dtype=np.float32).reshape(-1)    # (3,)

    denom = max(1, cfg.high_level_period - 1)
    phase = np.array([t_in_macro / denom], dtype=np.float32) # (1,)

    return np.concatenate([base, goal, phase], axis=-1).astype(np.float32)


# ================================================================
#  INTRINSIC REWARD FOR WORKER
# ================================================================
def intrinsic_reward(obs_dict, goal):
    """
    Worker reward:
        r_int = - || (paddle_pos - ball_pos) - goal ||

    goal is a desired (paddle - ball) offset.
    """
    ball = np.array(obs_dict["ball_pos"], dtype=np.float32)
    paddle = np.array(obs_dict["paddle_pos"], dtype=np.float32)

    current_offset = paddle - ball
    goal = np.asarray(goal, dtype=np.float32)

    err = np.linalg.norm(current_offset - goal)
    return -float(err)


# ================================================================
#  HIERARCHICAL PREDICTOR (for VideoRecorder)
# ================================================================
def make_hierarchical_predictor(cfg, manager_model, worker_model):
    """
    predict_fn used by VideoCallback:

        predict_fn(sb3_obs, env_instance) -> action (act_dim,)

    We ignore sb3_obs and directly use env_instance.unwrapped.obs_dict
    from the MyoSuite environment.
    """

    def predict_fn(_ignored_sb3_obs, env_instance):
        # Real MyoSuite dict observation
        obs_dict = env_instance.unwrapped.obs_dict

        # 1) Manager predicts a 3D goal
        m_obs = flatten_myo_obs_manager(obs_dict).reshape(1, -1)
        goal, _ = manager_model.predict(m_obs, deterministic=True)
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)

        # 2) Worker executes 1 low-level step with phase = 0
        w_obs = build_worker_obs(
            obs_dict=obs_dict,
            goal=goal,
            t_in_macro=0,
            cfg=cfg,
        ).reshape(1, -1)

        action, _ = worker_model.predict(w_obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    return predict_fn
