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

    def __init__(self, env_func: callable, cfg: Config, predict_fn, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg
        self.env_func = env_func
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
        env = self.env_func(self.cfg)
        obs, _ = env.reset(seed=self.cfg.seed + 123)

        # warmup renderer
        env.sim.renderer.render_offscreen(
            width=self.cfg.video_w,
            height=self.cfg.video_h,
            camera_id=self.cfg.camera_id,
        )

        frames = []

        for _ in range(self.cfg.video_frames):
            frame = env.sim.renderer.render_offscreen(
                width=self.cfg.video_w,
                height=self.cfg.video_h,
                camera_id=self.cfg.camera_id,
            )
            frames.append(frame)

            action = self.predict_fn(obs, env)

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=self.cfg.seed + 123)

        skvideo.io.vwrite(
            video_path,
            np.asarray(frames),
            outputdict={"-pix_fmt": "yuv420p"},
        )
        
        env.close()

        logger.info(f"✅ Saved video: {video_path}")