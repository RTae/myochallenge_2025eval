# callbacks/video_callback.py
import os
import numpy as np
import skvideo.io
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from myosuite.utils import gym as myo_gym


class VideoCallback(BaseCallback):
    """
    HRL-aware video callback.

    - Recreates a fresh MyoSuite env for recording.
    - Uses a predict_fn(obs, env_instance) provided externally.
    - Uses Config-only parameters.
    """

    def __init__(self, cfg, mode="worker", predict_fn=None, verbose=0):
        super().__init__(verbose)
        self.cfg = cfg
        self.mode = mode  # "worker" or "manager"
        self.predict_fn = predict_fn

        self.video_dir = os.path.join(cfg.logdir, mode, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        self.last_recorded = 0

    def _on_step(self) -> bool:
        step_count = self.num_timesteps
        if (step_count - self.last_recorded) >= self.cfg.video_freq:
            video_path = os.path.join(
                self.video_dir,
                f"{self.mode}_step_{step_count}.mp4",
            )
            logger.info(f"🎥 Triggering video capture at step {step_count}")
            self._record(video_path)
            self.last_recorded = step_count
        return True

    def _record(self, video_path: str):
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.pop("DISPLAY", None)

        env = myo_gym.make(self.cfg.env_id)
        obs, _ = env.reset(seed=self.cfg.seed + 123)

        # Warm-up
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
                # We ignore obs here and always use env.unwrapped.obs_dict inside predict_fn
                action = self.predict_fn(None, env)
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
