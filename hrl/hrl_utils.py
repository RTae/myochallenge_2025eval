import numpy as np
from typing import Dict

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


def build_worker_obs(obs_dict, goal, t_in_macro, cfg):
    """
    Worker obs MUST include ball information for hitting.
    
    Structure:
      base (proprioception) : qpos, qvel, act  (e.g., 424 dims)
      ball_pos             : 3
      ball_vel             : 3
      paddle_vel           : 3 (redundant if in base, but safe)
      goal                 : 3
      phase                : 1
      
    Total = base + 13 dims
    """
    # 1. Base proprioceptive observations (no ball)
    base = flatten_myo_obs_worker(obs_dict)  # [..., 424]
    
    # 2. CRITICAL: Add ball observations for hitting
    ball_pos = obs_dict["ball_pos"].astype(np.float32)  # [3]
    ball_vel = obs_dict["ball_vel"].astype(np.float32)  # [3]
    
    # 3. Goal and phase info
    paddle_vel = obs_dict["paddle_vel"].astype(np.float32)  # [3]
    goal = np.asarray(goal, dtype=np.float32)  # [3]
    phase = np.array(
        [t_in_macro / max(1, cfg.high_level_period - 1)],
        dtype=np.float32
    )  # [1]

    return np.concatenate(
        [base, ball_pos, ball_vel, paddle_vel, goal, phase],
        axis=-1
    ).astype(np.float32)

# MyoChallenge 2025 Physical Constants (from official specs)
BALL_MASS = 0.0027  # kg
BALL_RADIUS = 0.02  # m
PADDLE_MASS = 0.150  # kg
PADDLE_FACE_RADIUS = 0.093  # m
PADDLE_HANDLE_RADIUS = 0.016  # m
TABLE_HALF_WIDTH = 1.37  # m (each side)
NET_HEIGHT = 0.305  # m

class HitDetector:
    """Detects ball-paddle contact using velocity change."""

    def __init__(self, dv_thr: float = 2.5, ball_mass: float = BALL_MASS):
        self.dv_thr = dv_thr
        self.ball_mass = ball_mass
        self._prev_ball_vel = None

    def reset(self, obs_dict: dict):
        self._prev_ball_vel = np.array(obs_dict["ball_vel"], dtype=np.float32)

    def step(self, obs_dict: dict, dt: float = 0.01):
        ball_vel = np.array(obs_dict["ball_vel"], dtype=np.float32)

        dv = 0.0 if self._prev_ball_vel is None else float(
            np.linalg.norm(ball_vel - self._prev_ball_vel)
        )

        paddle_pos = obs_dict["paddle_pos"]
        ball_pos = obs_dict["ball_pos"]
        near_paddle = np.linalg.norm(ball_pos - paddle_pos) < 1.2 * PADDLE_FACE_RADIUS

        hit = (dv > self.dv_thr) and near_paddle

        if hit:
            contact_force = self.ball_mass * dv / dt
            self._prev_ball_vel = ball_vel.copy()
            return True, contact_force, dv

        self._prev_ball_vel = ball_vel.copy()
        return False, 0.0, dv


class WorkerReward:
    """Contact-conditioned reward with proximity-gated shaping
       and forced-commitment near the ball.
    """

    def __init__(self,
                 hit_bonus: float = 10.0,
                 impulse_bonus_scale: float = 2.0,
                 force_bonus_scale: float = 1.0,
                 sweet_spot_bonus: float = 2.0,
                 approach_scale: float = 0.2,
                 inactivity_penalty: float = -0.05,
                 energy_penalty_coef: float = 0.001,
                 ball_vel_threshold: float = 0.1,
                 paddle_radius: float = PADDLE_FACE_RADIUS,
                 near_thresh: float = 0.6,
                 commit_thresh: float = 0.3):  # NEW

        self.hit_bonus = hit_bonus
        self.impulse_bonus_scale = impulse_bonus_scale
        self.force_bonus_scale = force_bonus_scale
        self.sweet_spot_bonus = sweet_spot_bonus
        self.approach_scale = approach_scale
        self.inactivity_penalty = inactivity_penalty
        self.energy_penalty_coef = energy_penalty_coef
        self.ball_vel_threshold = ball_vel_threshold
        self.paddle_radius = paddle_radius
        self.near_thresh = near_thresh
        self.commit_thresh = commit_thresh  # NEW

        self._last_ball_vel = None

    # --------------------------------------------------
    def reset(self):
        self._last_ball_vel = None

    # --------------------------------------------------
    def _ball_dist(self, obs_dict) -> float:
        return np.linalg.norm(
            np.array(obs_dict["ball_pos"]) - np.array(obs_dict["paddle_pos"])
        )

    def _near_ball(self, obs_dict) -> bool:
        return self._ball_dist(obs_dict) < self.near_thresh

    def _commit_zone(self, obs_dict) -> bool:
        return self._ball_dist(obs_dict) < self.commit_thresh

    # --------------------------------------------------
    def _goal_align(self, obs_dict, goal) -> float:
        pv = np.array(obs_dict["paddle_vel"], dtype=np.float32)
        gv = np.asarray(goal, dtype=np.float32)
        pv /= (np.linalg.norm(pv) + 1e-6)
        gv /= (np.linalg.norm(gv) + 1e-6)
        return float(np.dot(pv, gv))

    # --------------------------------------------------
    def _compute_sweet_spot_bonus(self, obs_dict) -> float:
        return self.sweet_spot_bonus if self._ball_dist(obs_dict) < 0.30 * self.paddle_radius else 0.0

    # --------------------------------------------------
    def _compute_approach_bonus(self, obs_dict) -> float:
        ball_pos = np.array(obs_dict["ball_pos"])
        paddle_pos = np.array(obs_dict["paddle_pos"])
        ball_vel = np.array(obs_dict["ball_vel"])
        paddle_vel = np.array(obs_dict["paddle_vel"])

        rel = ball_pos - paddle_pos

        if np.linalg.norm(ball_vel) < self.ball_vel_threshold:
            return 0.0

        # Ball must be incoming
        if np.dot(ball_vel, rel) > 0:
            return 0.0

        approach = np.dot(paddle_vel, rel) / (np.linalg.norm(rel) + 1e-6)
        if approach > 0:
            speed_factor = np.clip(np.linalg.norm(ball_vel), 0.0, 2.0)
            return self.approach_scale * approach * speed_factor

        return 0.0

    # --------------------------------------------------
    def __call__(self,
                 obs_dict,
                 hit: bool,
                 goal=None,
                 external_dv=None,
                 external_contact_force=None):

        ball_vel = np.array(obs_dict["ball_vel"], dtype=np.float32)

        # --------------------------------------------------
        # Impulse / force (ONLY meaningful if hit)
        # --------------------------------------------------
        if hit and external_dv is not None:
            impulse = external_dv
            contact_force = external_contact_force
        else:
            impulse = 0.0
            contact_force = 0.0

        self._last_ball_vel = ball_vel.copy()

        # --------------------------------------------------
        # Energy penalty
        # --------------------------------------------------
        muscle_act = np.array(obs_dict.get("act", []), dtype=np.float32)
        energy_penalty = -self.energy_penalty_coef * np.sum(muscle_act ** 2)

        reward = 0.0
        components = {
            "hit_bonus": 0.0,
            "impulse_bonus": 0.0,
            "force_bonus": 0.0,
            "sweet_spot_bonus": 0.0,
            "approach_bonus": 0.0,
            "goal_alignment": 0.0,
            "energy_penalty": energy_penalty,
            "commit_penalty": 0.0,
            "inactivity_penalty": 0.0,
        }

        # ==================================================
        # HIT PHASE
        # ==================================================
        if hit:
            reward += self.hit_bonus
            components["hit_bonus"] = self.hit_bonus

            impulse_bonus = self.impulse_bonus_scale * np.tanh(impulse / 5.0)
            force_bonus = self.force_bonus_scale * np.tanh(contact_force / 25.0)
            sweet = self._compute_sweet_spot_bonus(obs_dict)

            reward += impulse_bonus + force_bonus + sweet
            components["impulse_bonus"] = impulse_bonus
            components["force_bonus"] = force_bonus
            components["sweet_spot_bonus"] = sweet

        # ==================================================
        # NO-HIT PHASE (STRICTLY GATED)
        # ==================================================
        else:
            if self._commit_zone(obs_dict):
                # 🔥 CLOSE but no hit → punish hovering
                reward += -0.08
                components["commit_penalty"] = -0.08

            elif self._near_ball(obs_dict):
                approach_bonus = self._compute_approach_bonus(obs_dict)
                reward += approach_bonus
                components["approach_bonus"] = approach_bonus

                if goal is not None:
                    ga = np.clip(self._goal_align(obs_dict, goal), -0.5, 0.5)
                    goal_bonus = 0.02 * ga
                    reward += goal_bonus
                    components["goal_alignment"] = goal_bonus

            else:
                reward += self.inactivity_penalty
                components["inactivity_penalty"] = self.inactivity_penalty

        reward += energy_penalty
        return float(reward), components

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
