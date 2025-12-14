import os
import numpy as np
import skvideo.io
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from config import Config

class VideoCallback(BaseCallback):
    """
    Video callback for SB3.
    Uses a DEDICATED raw MyoSuite env.
    """

    def __init__(self, env, cfg: Config, predict_fn, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg
        self.env = env
        self.predict_fn = predict_fn

        self.video_dir = os.path.join(cfg.logdir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        self._last_recorded = 0

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step - self._last_recorded >= self.cfg.video_freq:
            path = os.path.join(self.video_dir, f"s{step}.mp4")
            logger.info(f"🎥 Recording video at step {step}")
            self._record(path)
            self._last_recorded = step
        return True

    def _record(self, video_path: str):
        obs, _ = self.env.reset(seed=self.cfg.seed + 123)

        # warmup renderer
        self.env.sim.renderer.render_offscreen(
            width=self.cfg.video_w,
            height=self.cfg.video_h,
            camera_id=self.cfg.camera_id,
        )

        frames = []

        for _ in range(self.cfg.video_frames):
            frame = self.env.sim.renderer.render_offscreen(
                width=self.cfg.video_w,
                height=self.cfg.video_h,
                camera_id=self.cfg.camera_id,
            )
            frames.append(frame)

            if self.predict_fn is not None:
                action = self.predict_fn(obs, self.env)
            else:
                action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)

            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                obs, _ = self.env.reset(seed=self.cfg.seed + 123)

        skvideo.io.vwrite(
            video_path,
            np.asarray(frames),
            outputdict={"-pix_fmt": "yuv420p"},
        )

        logger.info(f"✅ Saved video: {video_path}")