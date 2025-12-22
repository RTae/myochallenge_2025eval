import numpy as np

def safe_unit(v, fallback):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.broadcast_to(fallback, v.shape).copy(), where=n > 1e-9)

def reflect_normal(d_in, d_out):
    """
    Ideal reflection normal: n ∝ (d_out - d_in)
    """
    return safe_unit(d_out - d_in, np.array([-1.0, 0.0, 0.0]))

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

def predict_ball_analytic(
    ball_pos,
    ball_vel,
    paddle_x,
    gravity=9.81,
    table_z=0.0,
    restitution=0.9,
):
    """
    Predict ball (pos, vel) at paddle X-plane using ballistic model
    (with at most one table bounce)
    """

    err_x = paddle_x - ball_pos[0]
    vx = ball_vel[0]

    if err_x <= 0.0 or vx <= 1e-3:
        return ball_pos.copy(), ball_vel.copy()

    dt = err_x / vx

    y0, z0 = ball_pos[1], ball_pos[2]
    vy0, vz0 = ball_vel[1], ball_vel[2]

    y = y0 + vy0 * dt
    z = z0 + vz0 * dt - 0.5 * gravity * dt * dt
    vz = vz0 - gravity * dt

    # Simple single bounce check
    if z < table_z:
        vz = -restitution * vz
        z = table_z

    pos = np.array([paddle_x, y, z], dtype=np.float32)
    vel = np.array([vx, vy0, vz], dtype=np.float32)
    return pos, vel


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
        