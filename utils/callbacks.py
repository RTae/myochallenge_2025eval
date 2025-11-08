import os
import numpy as np
import skvideo.io
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from myosuite.utils import gym


class VideoCallback(BaseCallback):
    """
    Combined evaluation + offscreen video recording callback for MyoSuite.

    Features:
    - Logs env-steps/sec
    - Periodically evaluates the policy
    - Records MyoSuite RGB videos (headless EGL safe)
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
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_id = env_id
        self.seed = seed
        self.logdir = logdir
        self.video_freq = video_freq
        self.eval_episodes = eval_episodes
        self.video_frames = video_frames
        self.camera_id = camera_id

        self._video_dir = os.path.join(logdir, "videos")
        os.makedirs(self._video_dir, exist_ok=True)

    # ------------------------------------------------------------
    def _on_step(self) -> bool:
        total_steps = self.model.num_timesteps

        # Periodic evaluation + video
        if (total_steps > 0) and (total_steps % self.video_freq == 0):
            self._record_eval_video_and_metrics()

        return True

    # ------------------------------------------------------------
    def _record_eval_video_and_metrics(self):
        # 1. Evaluate policy
        eval_env = gym.make(self.env_id)
        mean_r, std_r = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=self.eval_episodes,
            deterministic=True,
            warn=False,
            render=False,
        )
        logger.info(f"[Eval] meanR={mean_r:.2f} ± {std_r:.2f}")
        if self.logger:
            self.logger.record("eval/return_mean", float(mean_r))
            self.logger.record("eval/return_std", float(std_r))
        eval_env.close()

        # 2. Record short video with offscreen renderer
        video_path = os.path.join(
            self._video_dir, f"step_{self.model.num_timesteps}.mp4"
        )
        self._record_myo_video(video_path)

    # ------------------------------------------------------------
    def _record_myo_video(self, video_path):
        """Render a MyoSuite offscreen video (headless safe)."""
        env = gym.make(self.env_id)
        obs, _ = env.reset(seed=self.seed + 999)
        frames = []
        for _ in range(self.video_frames):
            frame = env.sim.renderer.render_offscreen(
                width=640, height=420, camera_id=self.camera_id
            )
            frames.append(frame)
            act, _ = self.model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                obs, _ = env.reset()
        env.close()

        skvideo.io.vwrite(video_path, np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
        logger.info(f"Saved video at {video_path}")