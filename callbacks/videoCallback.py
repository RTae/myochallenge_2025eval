import os
import numpy as np
import skvideo.io
from loguru import logger
from myosuite.utils import gym


class VideoCallback:
    """
    Standalone video logger for MyoSuite environments.

    Features:
    - Periodically records offscreen RGB videos (headless EGL-safe)
    - Works without Stable-Baselines3
    - Can attach any controller or policy for rendering actions
    """

    def __init__(
        self,
        env_id: str,
        seed: int,
        logdir: str,
        video_freq: int = 50_000,
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

        # Optional: user-attached predictor function
        self._predict_fn = None
        self._last_recorded = 0

    # ------------------------------------------------------------
    def attach_predictor(self, fn):
        """
        Attach an external action function for video rendering.

        Example:
            video_cb.attach_predictor(lambda obs: controller.compute_action(...))
        """
        self._predict_fn = fn
        if self.verbose:
            logger.info("🎯 Attached external action predictor to VideoCallback.")

    # ------------------------------------------------------------
    def step(self, step_count: int):
        """
        Manual trigger for periodic video capture.
        Call this inside your control loop with the current total step.
        """
        if (step_count - self._last_recorded) >= self.video_freq:
            video_path = os.path.join(self._video_dir, f"step_{step_count}.mp4")
            self._record_myo_video(video_path)
            self._last_recorded = step_count

    # ------------------------------------------------------------
    def _record_myo_video(self, video_path: str):
        """Render a MyoSuite offscreen video (headless safe)."""
        env = gym.make(self.env_id)
        obs, _ = env.reset(seed=self.seed + 999)
        frames = []

        logger.info(f"🎥 Recording MyoSuite video → {video_path}")

        for _ in range(self.video_frames):
            # Offscreen render
            frame = env.sim.renderer.render_offscreen(
                width=self.video_w, height=self.video_h, camera_id=self.camera_id
            )
            frames.append(frame)

            # --- Get action ---
            if self._predict_fn is not None:
                act = self._predict_fn(obs)
            else:
                act = np.zeros(env.action_space.shape)

            # Step environment
            step_out = env.step(act)
            if len(step_out) == 5:
                obs, _, term, trunc, _ = step_out
                done = term or trunc
            else:
                obs, _, done, _ = step_out

            if done:
                obs, _ = env.reset(seed=self.seed + 999)

        # Save video
        skvideo.io.vwrite(
            video_path,
            np.asarray(frames),
            outputdict={"-pix_fmt": "yuv420p"},
        )
        env.close()

        logger.info(f"✅ Saved video: {video_path}")
