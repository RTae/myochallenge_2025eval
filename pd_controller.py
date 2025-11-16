import numpy as np


class MorphologyAwareController:
    def __init__(self, env, kp=8.0, kd=1.5):
        self.env = env
        self.kp = kp
        self.kd = kd

    def compute_action(self, q_target):
        sim = self.env.unwrapped.sim
        q = sim.data.qpos.copy()
        qd = sim.data.qvel.copy()

        n = min(len(q_target), len(q), len(qd))
        q_target = q_target[:n]
        q = q[:n]
        qd = qd[:n]

        e = q_target - q
        u_raw = self.kp * e - self.kd * qd
        u = 1.0 / (1.0 + np.exp(-u_raw))

        act = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        act[:len(u)] = np.clip(u, 0.0, 1.0)
        return act
