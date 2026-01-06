from __future__ import annotations
from typing import Tuple
import numpy as np

EPS = 1e-12

# ---------------------------
# Small helpers
# ---------------------------

def normalize_vec(v: np.ndarray, eps: float = EPS) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

def normalize_quat(q: np.ndarray, eps: float = EPS) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n

# ---------------------------
# Quaternion core (wxyz)
# ---------------------------

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def quat_conj(q):
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def quat_rotate(q, v):
    q = normalize_quat(q)
    v = np.asarray(v, dtype=np.float64)
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]

# ---------------------------
# Geometry: shortest-arc quaternion
# ---------------------------

def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Shortest-arc quaternion rotating a -> b.
    Returns wxyz. Robust at 180°.
    """
    a = normalize_vec(a)
    b = normalize_vec(b)

    # if either was near-zero
    if np.linalg.norm(a) < EPS or np.linalg.norm(b) < EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    c = float(np.dot(a, b))  # cos(theta)

    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if c < -1.0 + 1e-12:
        # 180°: choose an orthogonal axis
        # pick basis axis least aligned with a
        if abs(a[0]) < abs(a[1]) and abs(a[0]) < abs(a[2]):
            ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif abs(a[1]) < abs(a[2]):
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            ortho = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        axis = normalize_vec(np.cross(a, ortho))
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)

    axis = np.cross(a, b)
    q = np.array([1.0 + c, axis[0], axis[1], axis[2]], dtype=np.float64)
    return normalize_quat(q)

# ---------------------------
# Paddle face normal + building goal quat
# ---------------------------

FACE_AXIS_LOCAL = np.array([0.0, 0.0, -1.0], dtype=np.float64)
QUNT_FACE_AXIS_LOCAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)

def get_face_normal(q: np.ndarray) -> np.ndarray:
    return quat_rotate(q, FACE_AXIS_LOCAL)

def normal_to_quat_face_aligned(n_world: np.ndarray) -> np.ndarray:
    """
    Rotate local FACE axis (-Z) -> n_world.
    """
    n_world = normalize_vec(n_world)
    return quat_from_two_vectors(QUNT_FACE_AXIS_LOCAL, n_world)

def flip_quat_180_x(q: np.ndarray) -> np.ndarray:
    """Rotate quaternion 180 degrees around its LOCAL X-axis: q * [0,1,0,0]."""
    q = np.asarray(q, dtype=np.float64)
    r = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # 180° about x
    return normalize_quat(quat_mul(q, r))

# ---------------------------
# Handle-down constraint
# ---------------------------

def flip_around_local_z(q):
    flip = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # 180° about local Z
    return normalize_quat(quat_mul(q, flip))

def ensure_handle_down(q, handle_axis_local=np.array([1.0, 0.0, 0.0], dtype=np.float64)):
    q = normalize_quat(q)
    handle_world = quat_rotate(q, handle_axis_local)
    if handle_world[2] > 0.0:
        q = flip_around_local_z(q)
    return q

# ---------------------------
# Prediction (only small cleanup)
# ---------------------------

def predict_ball_trajectory(
    ball_pos,
    ball_vel,
    paddle_pos,
    gravity=9.81,
    table_z=0.785,
    restitution=0.9,
    net_height=0.95,
    default_target=np.array([-0.9, 0.0, 0.95], dtype=np.float64),
) -> Tuple[np.ndarray, np.ndarray]:

    ball_pos = np.asarray(ball_pos, dtype=np.float64)
    ball_vel = np.asarray(ball_vel, dtype=np.float64)
    paddle_pos = np.asarray(paddle_pos, dtype=np.float64)

    px = float(paddle_pos[0])
    bx = float(ball_pos[0])
    vx = float(ball_vel[0])

    err_x = px - bx
    if err_x <= 0.0 or vx <= 1e-3:
        pred_pos = np.array([px, ball_pos[1], max(table_z, ball_pos[2])], dtype=np.float64)
        raw_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return pred_pos, raw_quat

    dt = float(np.clip(err_x / vx, 0.0, 2.0))

    y0, z0 = float(ball_pos[1]), float(ball_pos[2])
    vy0, vz0 = float(ball_vel[1]), float(ball_vel[2])

    y_pred = y0 + vy0 * dt
    z_pred = z0 + vz0 * dt - 0.5 * gravity * dt * dt
    vz_pred = vz0 - gravity * dt

    disc = vz0 * vz0 + 2.0 * gravity * (z0 - table_z)
    disc = max(disc, 0.0)
    t_hit = (vz0 + np.sqrt(disc)) / gravity
    hit_mask = (t_hit > 0.0) and (t_hit < dt)

    if hit_mask:
        vz_h = vz0 - gravity * t_hit
        vz_a = -restitution * vz_h
        dt2 = dt - t_hit
        y_pred = (y0 + vy0 * t_hit) + vy0 * dt2
        z_pred = table_z + vz_a * dt2 - 0.5 * gravity * dt2 * dt2
        vz_pred = vz_a - gravity * dt2

    z_pred -= 0.05
    pred_pos = np.array([px, y_pred, z_pred], dtype=np.float64)

    vx_est = max(abs(vx), 0.5)
    t_to_net = abs(px) / vx_est
    h_virt_net = net_height + 0.5 * gravity * (t_to_net ** 2)

    target_x = float(default_target[0])
    target_y = float(default_target[1])

    denom = (target_x - px)
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom == 0 else np.sign(denom) * 1e-6

    ratio_net = np.clip((-px) / denom, 1e-3, 1e3)

    target_z_req = z_pred + (h_virt_net - z_pred) / ratio_net
    final_target_z = float(np.clip(max(default_target[2], target_z_req), 0.5, 3.0))

    d_out = normalize_vec(np.array([target_x - px, target_y - y_pred, final_target_z - z_pred], dtype=np.float64))
    v_in  = normalize_vec(np.array([vx, vy0, vz_pred], dtype=np.float64))

    n = normalize_vec(d_out - v_in)
    if np.linalg.norm(n) < EPS:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    raw_quat = normal_to_quat_face_aligned(n)
    raw_quat = ensure_handle_down(raw_quat)

    return pred_pos, raw_quat