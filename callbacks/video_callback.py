import os
import numpy as np
import skvideo.io
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from config import Config
from myosuite.utils import gym as myo_gym


class VideoCallback(BaseCallback):
    """
    Generic video callback for SB3.
    Uses a raw MyoSuite env (not WorkerEnv/ManagerEnv) and a user-provided
    predict_fn(obs, env) to generate actions.
    """

    def __init__(self, cfg: Config, predict_fn, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg
        self.predict_fn = predict_fn

        self.video_dir = os.path.join(cfg.logdir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        self._last_recorded = 0

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step - self._last_recorded >= self.cfg.video_freq:
            path = os.path.join(self.video_dir, f"{self.mode}_step_{step}.mp4")
            logger.info(f"🎥 Triggering video capture at step {step}")
            self._record(path)
            self._last_recorded = step
        return True

    def _record(self, video_path: str):
        # Headless Mujoco
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.pop("DISPLAY", None)

        env = myo_gym.make(self.cfg.env_id)
        obs, _ = env.reset(seed=self.cfg.seed + 123)

        # warmup renderer
        _ = env.sim.renderer.render_offscreen(
            width=self.cfg.video_w,
            height=self.cfg.video_h,
            camera_id=self.cfg.camera_id,
        )

        frames = []
        logger.info(f"🎥 Recording video → {video_path}")

        for _ in range(self.cfg.video_frames):
            frame = env.sim.renderer.render_offscreen(
                width=self.cfg.video_w,
                height=self.cfg.video_h,
                camera_id=self.cfg.camera_id,
            )
            frames.append(frame)

            if self.predict_fn is not None:
                action = self.predict_fn(obs, env)
            else:
                action = np.zeros(env.action_space.shape[0], dtype=np.float32)

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=self.cfg.seed + 123)

        env.close()

        skvideo.io.vwrite(
            video_path,
            np.asarray(frames),
            outputdict={"-pix_fmt": "yuv420p"},
        )

        logger.info(f"✅ Saved video: {video_path}")
