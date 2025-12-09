import os
import numpy as np
import skvideo.io
from loguru import logger
from myosuite.utils import gym

from stable_baselines3.common.callbacks import BaseCallback


class VideoCallback(BaseCallback):
    """
    Unified SB3 video recording callback for MyoSuite HRL.

    - Uses Config for all settings
    - Works for BOTH worker training and manager training
    - Contains full video recorder logic (no extra class needed)
    - Automatically triggers video recording every cfg.video_freq steps
    """

    def __init__(self, cfg, mode="worker", predict_fn=None, verbose=0):
        """
        mode:       "worker" or "manager"
        predict_fn: function(obs, env) → action   (HRL-aware)
        """
        super().__init__(verbose)
        self.cfg = cfg
        self.mode = mode
        self.predict_fn = predict_fn

        # prepare directory
        self.video_dir = os.path.join(cfg.logdir, mode, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

    # -------------------------------------------------------
    # SB3 hook — automatically called every env step
    # -------------------------------------------------------
    def _on_step(self):
        if self.num_timesteps % self.cfg.video_freq == 0:
            video_path = os.path.join(
                self.video_dir, f"{self.mode}_step_{self.num_timesteps}.mp4"
            )
            logger.info(f"🎥 Triggering video capture at step {self.num_timesteps}")
            self._record(video_path)
        return True

    # -------------------------------------------------------
    # Actual video recording logic
    # -------------------------------------------------------
    def _record(self, video_path):

        # export MUJOCO_GL="egl"
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.pop("DISPLAY", None)

        env = gym.make(self.cfg.env_id)
        obs, _ = env.reset(seed=self.cfg.seed + 321)

        # warm-up renderer
        _ = env.sim.renderer.render_offscreen(
            width=self.cfg.video_w,
            height=self.cfg.video_h,
            camera_id=self.cfg.camera_id,
        )

        frames = []
        logger.info(f"🎥 Recording video → {video_path}")

        for _ in range(self.cfg.video_frames):

            # Render
            frame = env.sim.renderer.render_offscreen(
                width=self.cfg.video_w,
                height=self.cfg.video_h,
                camera_id=self.cfg.camera_id,
            )
            frames.append(frame)

            # Predict action (HRL-aware or worker-only)
            if self.predict_fn is not None:
                action = self.predict_fn(obs, env)
            else:
                # default safe fallback
                action = np.zeros(env.action_space.shape[0], dtype=np.float32)

            # step environment
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=self.cfg.seed + 321)

        env.close()

        skvideo.io.vwrite(
            video_path,
            np.asarray(frames),
            outputdict={"-pix_fmt": "yuv420p"},
        )

        logger.info(f"✅ Saved video: {video_path}")
