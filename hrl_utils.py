# hrl_utils.py
import numpy as np


# ====== OBS FLATTENING UTILS ======
def flatten_myo_obs_worker(obs_dict):
    """
    Build a 1D float32 vector for the worker (low-level policy).
    Adjust keys/ordering to match your MyoSuite env's obs.
    """
    parts = []

    # ---- CHANGE THESE KEYS IF NEEDED ----
    parts.append(obs_dict["pelvis_pos"])          # (3,)
    parts.append(obs_dict["body_qpos"])           # (Nq,)
    parts.append(obs_dict["body_vel"])            # (Nq,)
    parts.append(obs_dict["ball_pos"])           # (3,)
    parts.append(obs_dict["ball_vel"])           # (3,)
    parts.append(obs_dict["paddle_pos"])         # (3,)
    parts.append(obs_dict["paddle_vel"])         # (3,)
    parts.append(obs_dict["paddle_ori"])         # (3,) e.g. Euler or axis-angle
    parts.append(obs_dict["reach_err"])          # (3,)
    parts.append(obs_dict["muscle_activations"]) # (Nmusc,)

    return np.concatenate(parts, axis=-1).astype(np.float32)


def flatten_myo_obs_manager(obs_dict):
    """
    Build a 1D float32 vector for the manager (high-level policy).
    We keep this smaller than worker obs.
    """
    parts = []

    # ---- CHANGE THESE KEYS IF NEEDED ----
    parts.append(obs_dict["pelvis_pos"])
    parts.append(obs_dict["ball_pos"])
    parts.append(obs_dict["ball_vel"])
    parts.append(obs_dict["paddle_pos"])
    parts.append(obs_dict["paddle_vel"])
    parts.append(obs_dict["paddle_ori"])
    parts.append(obs_dict["reach_err"])

    return np.concatenate(parts, axis=-1).astype(np.float32)


# ====== WORKER OBS + INTRINSIC REWARD ======
def build_worker_obs(obs_dict, goal, t_in_macro, high_level_period):
    """
    Worker observation = [base_worker_obs, goal, phase]
      - goal: (3,) desired ball_to_paddle offset
      - phase: scalar in [0,1] indicating macro-step progress
    """
    base = flatten_myo_obs_worker(obs_dict)

    phase = np.array(
        [t_in_macro / max(1, high_level_period - 1)],
        dtype=np.float32,
    )

    goal = goal.astype(np.float32)
    return np.concatenate([base, goal, phase], axis=-1).astype(np.float32)


def intrinsic_reward(obs_dict, goal):
    """
    Intrinsic reward for worker:
      r_int = - || (paddle_pos - ball_pos) - goal ||
    """
    ball = obs_dict["ball_pos"]
    paddle = obs_dict["paddle_pos"]
    ball_to_paddle = paddle - ball  # current offset

    pos_error = np.linalg.norm(ball_to_paddle - goal)
    return -float(pos_error)

def make_hierarchical_predictor(cfg, manager_model, worker_model):
    """
    Returns a predict_fn used by VideoRecorder.
    """

    def predict_fn(obs, env):
        # 1) Manager gives a goal
        obs_vec = []  # flatten obs manually
        for key in ["pelvis_pos", "ball_pos", "ball_vel", "paddle_pos", "paddle_vel", "paddle_ori", "reach_err"]:
            obs_vec.append(obs[key])
        obs_vec = np.concatenate(obs_vec).astype(np.float32).reshape(1, -1)

        goal, _ = manager_model.predict(obs_vec, deterministic=True)
        goal = goal.astype(np.float32).flatten()

        # 2) Worker executes for 1 low-level step
        worker_obs = build_worker_obs(
            obs_dict=obs,
            goal=goal,
            t_in_macro=0,
            config=cfg,
        ).reshape(1, -1)

        action, _ = worker_model.predict(worker_obs, deterministic=True)
        return action

    return predict_fn