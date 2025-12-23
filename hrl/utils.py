import numpy as np

# ==================================================
# Global forward direction (from logs: +X)
# ==================================================
FWD = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def safe_unit(v, fallback):
    v = np.asarray(v, dtype=np.float32)
    fallback = np.asarray(fallback, dtype=np.float32)

    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(
        v,
        n,
        out=np.broadcast_to(fallback, v.shape).copy(),
        where=n > 1e-9,
    )


def reflect_normal(d_in, d_out):
    return safe_unit(d_out - d_in, FWD)


def quat_from_two_unit_vecs(a, b):
    """
    Shortest-arc quaternion rotating a -> b
    a, b: (..., 3)
    returns: (..., 4) quaternion [w, x, y, z]
    """
    a = safe_unit(a, FWD)
    b = safe_unit(b, FWD)

    dot = np.sum(a * b, axis=-1, keepdims=True)
    v = np.cross(a, b)
    w = 1.0 + dot

    q = np.concatenate([w, v], axis=-1)
    qn = np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.divide(q, qn, out=np.zeros_like(q), where=qn > 1e-12)

    # Handle opposite vectors (180 deg), group-safe
    opposite = dot.squeeze(-1) < -0.999999
    if np.any(opposite):
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        q = q.copy()
        q[opposite] = np.concatenate(
            [
                np.zeros((np.sum(opposite), 1), dtype=np.float32),
                np.tile(axis, (np.sum(opposite), 1)),
            ],
            axis=-1,
        )

    return q


def quat_to_paddle_normal(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)

    x, y, z, w = q
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dtype=np.float32,
    )


# --------------------------------------------------
# Core prediction
# --------------------------------------------------
def predict_ball_analytic(
    sim,
    id_info,
    ball_pos,
    ball_vel,
    paddle_pos,
    opp_target=np.array([0.7, 0.0, 0.8], dtype=np.float32),
    max_dt=2.0,
    restitution=None,
):
    ball_pos = np.asarray(ball_pos, dtype=np.float32)
    ball_vel = np.asarray(ball_vel, dtype=np.float32)
    paddle_pos = np.asarray(paddle_pos, dtype=np.float32)

    # --------------------------------------------------
    # Time to paddle X-plane (ball moves +X)
    # --------------------------------------------------
    err_x = paddle_pos[0] - ball_pos[0]
    vx = ball_vel[0]

    if err_x <= 0.0 or vx <= 1e-3:
        pred_ball_pos = ball_pos.copy()
        pred_ball_vel = ball_vel.copy()
    else:
        dt = np.clip(err_x / vx, 0.0, max_dt)

        # --------------------------------------------------
        # Physics
        # --------------------------------------------------
        g = float(-sim.model.opt.gravity[2])
        if not np.isfinite(g) or g <= 0:
            g = 9.81

        if restitution is None:
            restitution = float(getattr(sim, "predict_restitution", 0.9))
        restitution = float(np.clip(restitution, 0.0, 1.0))

        own_gid = id_info.own_half_gid
        table_z = (
            sim.data.geom_xpos[own_gid][2]
            + sim.model.geom_size[own_gid][2]
        )
        ball_r = sim.model.geom_size[id_info.ball_gid][0]
        z_contact = table_z + ball_r

        y0, z0 = ball_pos[1], ball_pos[2]
        vy0, vz0 = ball_vel[1], ball_vel[2]

        y = y0 + vy0 * dt
        z = z0 + vz0 * dt - 0.5 * g * dt * dt
        vz = vz0 - g * dt

        if z < z_contact:
            a = -0.5 * g
            b = vz0
            c = z0 - z_contact

            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                t_hit = (-b - np.sqrt(disc)) / (2.0 * a)
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
    d_in = safe_unit(pred_ball_vel, FWD)
    d_out = safe_unit(
        opp_target - pred_ball_pos,
        FWD,
    )

    n_ideal = safe_unit(d_out - d_in, FWD)

    # Ensure paddle faces +X (group-safe)
    if n_ideal[0] < 0.0:
        n_ideal = -n_ideal

    # --------------------------------------------------
    # Quaternion: +X → n_ideal
    # --------------------------------------------------
    ref = FWD
    paddle_quat_ideal = quat_from_two_unit_vecs(ref, n_ideal)

    return pred_ball_pos, n_ideal, paddle_quat_ideal


# --------------------------------------------------
# Oracle sim predictor (unchanged)
# --------------------------------------------------
def predict_ball_with_sim(env, paddle_x, max_time=2.0):
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
                vel = env.get_sensor_by_name(
                    model, data, "pingpong_vel_sensor"
                )
                return cur.copy(), vel.copy(), True
            prev = cur

        return prev.copy(), vel.copy(), False

    finally:
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        data.time = time0
        sim.forward()