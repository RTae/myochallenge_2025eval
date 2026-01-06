from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ============================================================
# Quaternion helpers (w, x, y, z)
# ============================================================

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

def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / n
    s = np.sin(angle_rad / 2.0)
    return quat_normalize(np.array([np.cos(angle_rad / 2.0), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64))


def get_z_normal(q: np.ndarray) -> np.ndarray:
    """World direction of LOCAL +Z under quaternion q."""
    return quat_rotate(q, np.array([0.0, 0.0, 1.0], dtype=np.float64))


def normal_to_quat_z_aligned(n: np.ndarray) -> np.ndarray:
    """
    Shortest-arc quaternion that rotates [0,0,1] -> n.
    Returns wxyz.
    """
    n = np.asarray(n, dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    n = n / nn

    nx, ny, nz = n
    # shortest arc from z=[0,0,1] to n:
    qw = 1.0 + nz
    qx = -ny
    qy = nx
    qz = 0.0

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    qn = np.linalg.norm(q)

    # singularity: n ~ -z
    if qn < 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    return q / qn


def flip_quat_180_x(q: np.ndarray) -> np.ndarray:
    """Rotate quaternion 180 degrees around its LOCAL X-axis: q * [0,1,0,0]."""
    q = np.asarray(q, dtype=np.float64)
    r = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # 180deg about x
    return quat_normalize(quat_mul(q, r))


def roll_about_local_x(q: np.ndarray, angle_rad: float) -> np.ndarray:
    """Post-multiply by rotation around LOCAL X (handle axis)."""
    r = quat_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=np.float64), angle_rad)
    return quat_normalize(quat_mul(q, r))


def tilt_about_local_y(q: np.ndarray, angle_rad: float) -> np.ndarray:
    """Post-multiply by rotation around LOCAL Y."""
    r = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float64), angle_rad)
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
    raw_quat = normal_to_quat_z_aligned(n)
    raw_quat = ensure_handle_down(raw_quat)
    
    return pred_pos, raw_quat


# ============================================================
# Human layer: forehand/backhand + comfort roll/tilt
# ============================================================

@dataclass
class GripStyle:
    # your observation: "in myo forehand is black"
    # we'll map:
    #   forehand -> apply 180deg local-X flip (or not) depending on visuals
    # You can swap these if colors are opposite in your viewer.

    forehand_flip_x: bool = True     # set True if forehand needs flip_quat_180_x
    backhand_flip_x: bool = False

    # handle comfort: negative roll makes handle "down" depending on convention
    forehand_roll_deg: float = -8.0
    backhand_roll_deg: float = -3.0

    # optional: tilt slightly toward agent (makes face less skyward)
    forehand_tilt_y_deg: float = 0.0
    backhand_tilt_y_deg: float = 0.0

    # switch threshold
    y_threshold: float = -0.05


def apply_grip_style(raw_quat: np.ndarray, pelvis_y: float, pred_y: float, style: GripStyle):
    """
    Decides forehand/backhand and applies:
      - optional flip_quat_180_x (your old logic)
      - roll around LOCAL X (handle comfort)  <-- where your "15deg" belongs
      - optional small tilt
    """
    relative_y = pred_y - pelvis_y

    is_backhand = (relative_y <= style.y_threshold)

    if is_backhand:
        q = raw_quat.copy()
        mode = "BACKHAND"
        if style.backhand_flip_x:
            q = flip_quat_180_x(q)
        q = roll_about_local_x(q, np.deg2rad(style.backhand_roll_deg))
        q = tilt_about_local_y(q, np.deg2rad(style.backhand_tilt_y_deg))
    else:
        q = raw_quat.copy()
        mode = "FOREHAND"
        if style.forehand_flip_x:
            q = flip_quat_180_x(q)
        q = roll_about_local_x(q, np.deg2rad(style.forehand_roll_deg))
        q = tilt_about_local_y(q, np.deg2rad(style.forehand_tilt_y_deg))

    return quat_normalize(q), mode