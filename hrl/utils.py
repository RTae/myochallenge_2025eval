import numpy as np

def safe_unit(v, fallback):
    """
    Normalize vector(s) safely.
    v: (..., 3)
    fallback: (3,)
    """
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(
        v,
        n,
        out=np.broadcast_to(fallback, v.shape).copy(),
        where=n > 1e-9,
    )
    
def reflect_normal(d_in, d_out):
    """
    Ideal reflection normal: n ∝ (d_out - d_in)
    """
    return safe_unit(d_out - d_in, np.array([-1.0, 0.0, 0.0]))

def quat_from_two_unit_vecs(a, b):
    """
    Shortest-arc quaternion rotating a -> b.
    a, b: (..., 3) unit vectors
    returns: (..., 4) quaternion [w, x, y, z]
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    dot = np.sum(a * b, axis=-1, keepdims=True)
    v = np.cross(a, b)
    w = 1.0 + dot

    q = np.concatenate([w, v], axis=-1)
    qn = np.linalg.norm(q, axis=-1, keepdims=True)

    q = np.divide(q, qn, out=np.zeros_like(q), where=qn > 1e-12)

    # Handle opposite vectors (180 deg)
    opposite = (dot < -0.999999).squeeze(-1)
    if np.any(opposite):
        q_op = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if q.ndim == 1:
            q = q_op
        else:
            q = q.copy()
            q[opposite] = q_op

    return q

def quat_to_paddle_normal(q: np.ndarray) -> np.ndarray:
    """
    Paddle hitting normal = local +X axis
    Quaternion format: (x, y, z, w)
    """
    q = q.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)

    x, y, z, w = q

    # X axis of rotation matrix
    return np.array([
        1.0 - 2.0 * (y*y + z*z),
        2.0 * (x*y + w*z),
        2.0 * (x*z - w*y),
    ], dtype=np.float32)

def quat_from_two_unit_vecs(a, b):
    """
    Quaternion rotating a -> b (shortest arc)
    Returns [w, x, y, z]
    """
    dot = np.sum(a * b, axis=-1, keepdims=True)
    v = np.cross(a, b)
    w = 1.0 + dot
    q = np.concatenate([w, v], axis=-1)
    q_norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.divide(q, q_norm, out=np.zeros_like(q), where=q_norm > 1e-12)

    opposite = (dot < -0.999999).squeeze(-1)
    if np.any(opposite):
        q_op = np.array([0.0, 0.0, 0.0, 1.0])
        q = q.copy()
        q[opposite] = q_op

    return q

# --------------------------------------------------
# Core prediction
# --------------------------------------------------
def predict_ball_analytic(
    sim,
    id_info,
    ball_pos,
    ball_vel,
    paddle_pos,
    opp_target=np.array([-0.7, 0.0, 0.8], dtype=np.float32),
    max_dt=2.0,
    restitution=None,
):
    """
    Predict ball position at paddle X-plane and compute ideal paddle orientation.

    Parameters
    ----------
    sim : MuJoCo sim (env.unwrapped.sim)
    id_info : env.unwrapped.id_info
    ball_pos : (3,)
    ball_vel : (3,)
    paddle_pos : (3,)
    opp_target : (3,) target point on opponent side
    max_dt : float
    restitution : optional override

    Returns
    -------
    pred_ball_pos : (3,)
    n_ideal : (3,) ideal paddle normal
    paddle_quat_ideal : (4,) quaternion [w, x, y, z]
    """

    ball_pos = np.asarray(ball_pos, dtype=np.float32)
    ball_vel = np.asarray(ball_vel, dtype=np.float32)
    paddle_pos = np.asarray(paddle_pos, dtype=np.float32)

    # --------------------------------------------------
    # Time to paddle X-plane
    # --------------------------------------------------
    err_x = paddle_pos[0] - ball_pos[0]
    vx = ball_vel[0]

    if err_x <= 0.0 or vx <= 1e-3:
        # Degenerate: ball not coming toward paddle
        pred_ball_pos = ball_pos.copy()
        pred_ball_vel = ball_vel.copy()
    else:
        dt = np.clip(err_x / vx, 0.0, max_dt)

        # --------------------------------------------------
        # Physics parameters from MuJoCo
        # --------------------------------------------------
        g = float(-sim.model.opt.gravity[2])
        if not np.isfinite(g) or g <= 0:
            g = 9.81

        if restitution is None:
            restitution = float(getattr(sim, "predict_restitution", 0.9))
        restitution = float(np.clip(restitution, 0.0, 1.0))

        # Table geometry
        own_gid = id_info.own_half_gid
        table_z = (
            sim.data.geom_xpos[own_gid][2]
            + sim.model.geom_size[own_gid][2]
        )
        ball_r = sim.model.geom_size[id_info.ball_gid][0]
        z_contact = table_z + ball_r

        # --------------------------------------------------
        # Ballistic prediction (pre-bounce)
        # --------------------------------------------------
        y0, z0 = ball_pos[1], ball_pos[2]
        vy0, vz0 = ball_vel[1], ball_vel[2]

        y = y0 + vy0 * dt
        z = z0 + vz0 * dt - 0.5 * g * dt * dt
        vz = vz0 - g * dt

        # --------------------------------------------------
        # Single table bounce
        # --------------------------------------------------
        if z < z_contact:
            # Solve z0 + vz0*t - 0.5*g*t^2 = z_contact
            a = -0.5 * g
            b = vz0
            c = z0 - z_contact

            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sqrt_disc = np.sqrt(disc)
                t_hit = (-b - sqrt_disc) / (2.0 * a)

                if 0.0 < t_hit < dt:
                    vz_hit = vz0 - g * t_hit
                    vz_after = -restitution * vz_hit
                    dt2 = dt - t_hit

                    y_hit = y0 + vy0 * t_hit
                    y = y_hit + vy0 * dt2
                    z = z_contact + vz_after * dt2 - 0.5 * g * dt2 * dt2
                    vz = vz_after - g * dt2

        pred_ball_pos = np.array(
            [paddle_pos[0], y, z], dtype=np.float32
        )
        pred_ball_vel = np.array(
            [vx, vy0, vz], dtype=np.float32
        )

    # --------------------------------------------------
    # Reflection-law paddle normal
    # --------------------------------------------------
    d_out = safe_unit(
        opp_target - pred_ball_pos,
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    )
    d_in = safe_unit(
        pred_ball_vel,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    n_ideal = safe_unit(
        d_out - d_in,
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    )

    # Ensure paddle faces ball (−X convention)
    if n_ideal[0] > 0.0:
        n_ideal = -n_ideal

    # --------------------------------------------------
    # Quaternion: [-X] → n_ideal
    # --------------------------------------------------
    ref = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    paddle_quat_ideal = quat_from_two_unit_vecs(
        safe_unit(ref, ref),
        safe_unit(n_ideal, ref),
    )

    return pred_ball_pos, n_ideal, paddle_quat_ideal

def predict_ball_with_sim(env, paddle_x, max_time=2.0):
    """
    Uses env.sim to roll forward and predict ball at x-plane.
    SLOW – use only for debugging or oracle.
    """
    sim = env.sim
    model = sim.model
    data = sim.data

    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    time0 = float(data.time)

    try:
        dt_sim = float(model.opt.timestep)
        steps = int(max_time / dt_sim)

        ball_sid = env.id_info.ball_sid
        prev = data.site_xpos[ball_sid].copy()

        for _ in range(steps):
            sim.step()
            cur = data.site_xpos[ball_sid].copy()
            if prev[0] < paddle_x <= cur[0]:
                vel = env.get_sensor_by_name(model, data, "pingpong_vel_sensor")
                return cur.copy(), vel.copy(), True
            prev = cur

        return prev.copy(), vel.copy(), False

    finally:
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        data.time = time0
        sim.forward()
        