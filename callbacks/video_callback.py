import os
import numpy as np
import skvideo.io
from loguru import logger
from myosuite.utils import gym


class VideoCallback:
    def __init__(
        self,
        env_id: str,
        seed: int,
        logdir: str,
        video_freq: int = 50000,
        eval_episodes: int = 1,
        video_frames: int = 400,
        camera_id: int = 1,
        video_w: int = 640,
        video_h: int = 420,
        verbose: int = 0,
    ):
        self.env_id = env_id
        self.seed = seed
        self.logdir = logdir
        self.video_freq = video_freq
        self.eval_episodes = eval_episodes
        self.video_frames = video_frames
        self.camera_id = camera_id
        self.video_w = video_w
        self.video_h = video_h
        self.verbose = verbose

        self._video_dir = os.path.join(logdir, "videos")
        os.makedirs(self._video_dir, exist_ok=True)

        self._predict_fn = None
        self._last_recorded = 0

    def attach_predictor(self, fn):
        self._predict_fn = fn

    def step(self, step_count: int):
        if (step_count - self._last_recorded) >= self.video_freq:
            path = os.path.join(self._video_dir, f"step_{step_count}.mp4")
            self._record(path)
            self._last_recorded = step_count

    def _record(self, video_path):

        os.environ["MUJOCO_GL"] = "egl"
        os.environ.pop("DISPLAY", None)

        env = gym.make(self.env_id)
        obs, _ = env.reset(seed=self.seed + 123)

        # Warm up renderer
        _ = env.sim.renderer.render_offscreen(
            width=self.video_w,
            height=self.video_h,
            camera_id=self.camera_id,
        )

        frames = []
        logger.info(f"🎥 Recording MyoSuite video → {video_path}")

        for _ in range(self.video_frames):

            frame = env.sim.renderer.render_offscreen(
                width=self.video_w,
                height=self.video_h,
                camera_id=self.camera_id,
            )
            frames.append(frame)

            if self._predict_fn:
                act = self._predict_fn(obs)
            else:
                act = np.zeros(env.action_space.shape[0], dtype=np.float32)

            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                obs, _ = env.reset(seed=self.seed + 123)

        env.close()

        skvideo.io.vwrite(
            video_path,
            np.asarray(frames),
            outputdict={"-pix_fmt": "yuv420p"},
        )

        logger.info(f"✅ Saved video: {video_path}")
