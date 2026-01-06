from __future__ import annotations

from typing import Tuple

import numpy as np

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_normalize(q):
    return q / (np.linalg.norm(q) + 1e-12)

def quat_rotate(q, v):
    qv = np.array([0.0, *v])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]

def get_face_normal(q: np.ndarray) -> np.ndarray:
    return quat_rotate(q, np.array([0.0, 0.0, -1.0], dtype=np.float64))

def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Shortest-arc quaternion that rotates unit vector a -> unit vector b.
    Returns wxyz.
    Robust for near-opposite vectors (180°).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    a = a / na
    b = b / nb

    c = np.dot(a, b)  # cos(theta)
    if c > 1.0 - 1e-12:
        # already aligned
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if c < -1.0 + 1e-12:
        # 180°: choose any axis orthogonal to a
        # pick the smallest component axis to avoid near-zero cross
        if abs(a[0]) < abs(a[1]) and abs(a[0]) < abs(a[2]):
            ortho = np.array([1.0, 0.0, 0.0])
        elif abs(a[1]) < abs(a[2]):
            ortho = np.array([0.0, 1.0, 0.0])
        else:
            ortho = np.array([0.0, 0.0, 1.0])

        axis = np.cross(a, ortho)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        # 180° rotation: w=0, xyz=axis
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)

    axis = np.cross(a, b)
    q = np.array([1.0 + c, axis[0], axis[1], axis[2]], dtype=np.float64)
    return quat_normalize(q)

def normal_to_quat_face_aligned(n: np.ndarray) -> np.ndarray:
    """
    Rotate local FACE axis (-Z) -> n_world.
    Returns wxyz.
    """
    n = quat_normalize(n)
    return quat_from_two_vectors(np.array([0.0, 0.0, -1.0], dtype=np.float64), n)

def flip_quat_180_x(q: np.ndarray) -> np.ndarray:
    """Rotate quaternion 180 degrees around its LOCAL X-axis: q * [0,1,0,0]."""
    q = np.asarray(q, dtype=np.float64)
    r = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # 180deg about x
    return quat_normalize(quat_mul(q, r))

def flip_around_local_z(q):
    # 180° rotation around local Z axis
    flip = np.array([0.0, 0.0, 0.0, 1.0])
    return quat_normalize(quat_mul(q, flip))

def ensure_handle_down(q, handle_axis_local=np.array([1.0, 0.0, 0.0])):
    """
    Ensures the HANDLE axis points "down" in world Z.
    handle_axis_local must be the local axis that actually matches the handle direction in Myo.
    """
    q = quat_normalize(q)

    handle_world = quat_rotate(q, handle_axis_local)
    # if handle points upward, flip roll around face normal (local Z)
    if handle_world[2] > 0.0:
        q = flip_around_local_z(q)

    return quat_normalize(q)

# ============================================================
# Physics-only prediction (NO human tilt/roll here)
# ============================================================

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
    """
    Returns:
      pred_pos: (3,)
      raw_quat: (4,) wxyz, aligns local +Z to desired contact normal (roll-neutral)
    """
    ball_pos = np.asarray(ball_pos, dtype=np.float64)
    ball_vel = np.asarray(ball_vel, dtype=np.float64)
    paddle_pos = np.asarray(paddle_pos, dtype=np.float64)

    px = float(paddle_pos[0])
    bx = float(ball_pos[0])
    vx = float(ball_vel[0])

    # time to reach paddle x-plane
    err_x = px - bx
    if err_x <= 0.0 or vx <= 1e-3:
        # fallback
        pred_pos = np.array([px, ball_pos[1], max(table_z, ball_pos[2])], dtype=np.float64)
        raw_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return pred_pos, raw_quat

    dt = np.clip(err_x / vx, 0.0, 2.0)

    y0, z0 = ball_pos[1], ball_pos[2]
    vy0, vz0 = ball_vel[1], ball_vel[2]

    # initial (no bounce) prediction at dt
    y_pred = y0 + vy0 * dt
    z_pred = z0 + vz0 * dt - 0.5 * gravity * dt * dt
    vz_pred = vz0 - gravity * dt

    # bounce time to table plane (z=table_z)
    disc = vz0 * vz0 + 2.0 * gravity * (z0 - table_z)
    disc = max(disc, 0.0)
    t_hit = (vz0 + np.sqrt(disc)) / gravity  # descending root for typical (works ok)
    hit_mask = (t_hit > 0.0) and (t_hit < dt)

    if hit_mask:
        # velocity at impact
        vz_h = vz0 - gravity * t_hit
        vz_a = -restitution * vz_h  # bounce
        dt2 = dt - t_hit
        y_pred = (y0 + vy0 * t_hit) + vy0 * dt2
        z_pred = table_z + vz_a * dt2 - 0.5 * gravity * dt2 * dt2
        vz_pred = vz_a - gravity * dt2

    z_pred = z_pred - 0.05 # To hit at center of paddle height
    pred_pos = np.array([px, y_pred, z_pred], dtype=np.float64)

    # ------------------------------------------------------------
    # Desired outgoing direction: aim ball toward default_target
    # while ensuring it clears the net (virtual net trick).
    # ------------------------------------------------------------
    vx_est = max(abs(vx), 0.5)
    t_to_net = abs(px) / vx_est  # rough time to reach x=0 net plane
    gravity_drop = 0.5 * gravity * (t_to_net ** 2)
    h_virt_net = net_height + gravity_drop

    target_x = float(default_target[0])
    target_y = float(default_target[1])

    denom = (target_x - px)
    if abs(denom) < 1e-6:
        denom = np.sign(denom) * 1e-6 if denom != 0 else 1e-6

    # ratio for linear interpolation at x=0:
    ratio_net = (-px) / denom
    ratio_net = np.clip(ratio_net, 1e-3, 1e3)

    # required target z such that line from contact to target crosses net above h_virt_net
    target_z_req = z_pred + (h_virt_net - z_pred) / ratio_net
    final_target_z = float(np.clip(max(default_target[2], target_z_req), 0.5, 3.0))

    # outgoing direction unit vector
    dx = target_x - px
    dy = target_y - y_pred
    dz = final_target_z - z_pred
    d_out = np.array([dx, dy, dz], dtype=np.float64)
    d_out /= (np.linalg.norm(d_out) + 1e-9)

    # incoming direction unit vector
    v_in = np.array([vx, vy0, vz_pred], dtype=np.float64)
    v_in /= (np.linalg.norm(v_in) + 1e-9)

    # contact normal ~ (d_out - v_in) normalized (simple reflection heuristic)
    n = d_out - v_in
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = n / n_norm

    # quaternion aligning local +Z to n (ROLL NEUTRAL)
    raw_quat = normal_to_quat_face_aligned(n)
    raw_quat = ensure_handle_down(raw_quat)
    
    return pred_pos, raw_quat