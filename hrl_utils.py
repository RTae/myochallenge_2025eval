# hrl_utils.py
import numpy as np


# ================================================================
#  FLATTEN WORKER OBSERVATION (Low-level controller)
# ================================================================
def flatten_myo_obs_worker(obs_dict):
    """
    Build a 1D float32 vector for the worker.
    This version auto-expands scalars into 1D arrays.
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
        obs_dict["padde_ori_err"],
        obs_dict["reach_err"],
        obs_dict["palm_pos"],
        obs_dict["palm_err"],
        obs_dict["touching_info"],
        obs_dict["act"],
    ]

    safe_parts = []
    for p in parts:
        arr = np.array(p, dtype=np.float32)

        # convert scalar to (1,) array
        if arr.ndim == 0:
            arr = arr.reshape(1)

        safe_parts.append(arr)

    return np.concatenate(safe_parts, axis=-1)

# ================================================================
#  FLATTEN MANAGER OBSERVATION (High-level controller)
# ================================================================
def flatten_myo_obs_manager(obs_dict):
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
#  WORKER INPUT DURING HRL EXECUTION
# ================================================================
def build_worker_obs(obs_dict, goal, t, cfg):
    """
    Worker observation during HRL:
        [flattened_base_obs (429), goal (3), phase (1)]
    Total dims: 433
    """
    base = flatten_myo_obs_worker(obs_dict)         # (429,)
    goal = np.asarray(goal, dtype=np.float32)       # (3,)
    phase = np.array([t / cfg.high_level_period], dtype=np.float32)   # (1,)

    return np.concatenate([base, goal, phase], axis=-1).astype(np.float32)


# ================================================================
#  INTRINSIC REWARD FOR WORKER
# ================================================================
def intrinsic_reward(obs_dict, goal):
    """
    Worker reward:
        r_int = - || (paddle_pos - ball_pos) - goal ||
    """
    ball = obs_dict["ball_pos"]
    paddle = obs_dict["paddle_pos"]
    offset = paddle - ball                   # current paddle-to-ball offset
    err = np.linalg.norm(offset - goal)      # difference from desired goal
    return -float(err)


# ================================================================
#  HIERARCHICAL PREDICTOR (for VideoRecorder)
# ================================================================
def make_hierarchical_predictor(cfg, manager_model, worker_model):
    """
    Function for VideoRecorder.predict_fn:
        VideoRecorder calls:
            predict_fn(sb3_obs, video_env)

        BUT:
            sb3_obs is a flattened vector → NOT USEFUL
            video_env.obs_dict contains the REAL MyoSuite dictionary

        So:
            We IGNORE sb3_obs entirely
            And always use env.obs_dict
    """

    def predict_fn(_ignored_sb3_obs, env_instance):
        # -----------------------------------------------------
        # Extract the real MyoSuite observation dictionary
        # -----------------------------------------------------
        obs_dict = env_instance.obs_dict

        # -----------------------------------------------------
        # 1) MANAGER predicts high-level goal (3D)
        # -----------------------------------------------------
        m_obs = flatten_myo_obs_manager(obs_dict).reshape(1, -1)
        goal, _ = manager_model.predict(m_obs, deterministic=True)
        goal = goal.astype(np.float32).flatten()

        # -----------------------------------------------------
        # 2) WORKER predicts muscle activations for 1 step
        # -----------------------------------------------------
        w_obs = build_worker_obs(
            obs_dict=obs_dict,
            goal=goal,
            t=0,              # always first step for video rollout
            cfg=cfg,
        ).reshape(1, -1)

        action, _ = worker_model.predict(w_obs, deterministic=True)
        return action

    return predict_fn