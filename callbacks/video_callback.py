import os
import numpy as np
import skvideo.io
from loguru import logger
from myosuite.utils import gym


class VideoCallback:
    def __init__(self, env_id, seed, logdir, video_freq=50000,
                 eval_episodes=1, video_frames=400, camera_id=1,
                 video_w=640, video_h=420):

        self.env_id = env_id
        self.seed = seed
        self.logdir = logdir
        self.video_freq = video_freq
        self.video_frames = video_frames
        self.camera_id = camera_id
        self.video_w = video_w
        self.video_h = video_h

        self.predict_fn = None
        self.last_record = 0

        self.video_dir = os.path.join(logdir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

    def attach_predictor(self, fn):
        self.predict_fn = fn

    def step(self, step):
        if step - self.last_record >= self.video_freq:
            self.last_record = step
            path = os.path.join(self.video_dir, f"step_{step}.mp4")
            self._record(path)

    def _record(self, filename):

        os.environ["MUJOCO_GL"] = "egl"
        os.environ.pop("DISPLAY", None)

        env = gym.make(self.env_id)
        obs, _ = env.reset(seed=self.seed + 123)

        frames = []
        logger.info(f"🎥 Recording → {filename}")

        _ = env.sim.renderer.render_offscreen(
            width=self.video_w, height=self.video_h, camera_id=self.camera_id
        )

        for _ in range(self.video_frames):
            frame = env.sim.renderer.render_offscreen(
                width=self.video_w, height=self.video_h, camera_id=self.camera_id
            )
            frames.append(frame)

            if self.predict_fn is not None:
                act = self.predict_fn(obs, env)
            else:
                act = np.zeros(env.action_space.shape[0])

            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                obs, _ = env.reset(seed=self.seed + 123)

        env.close()
        skvideo.io.vwrite(filename, np.asarray(frames),
                          outputdict={"-pix_fmt": "yuv420p"})
        logger.info(f"Saved video {filename}")
