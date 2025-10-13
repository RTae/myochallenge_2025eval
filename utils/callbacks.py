import os
import numpy as np
import imageio
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from myosuite.utils import gym


class VideoCallback(BaseCallback):
    """
    Stable-Baselines3-only callback.

    ✅ Safe for headless containers (MUJOCO_GL=osmesa)
    ✨ Features:
      - Periodic evaluation
      - Records .mp4 videos via env.render()
      - Saves best model automatically
    """

    def __init__(
        self,
        eval_env_id: str = "myoChallengeTableTennisP2-v0",
        eval_freq: int = 10_000,
        video_dir: str = "./logs/sb3_videos",
        best_model_dir: str = "./logs/sb3_best",
        n_eval_episodes: int = 3,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env_id = eval_env_id
        self.eval_freq = eval_freq
        self.video_dir = video_dir
        self.best_model_dir = best_model_dir
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.eval_env = None

        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)

        # Disable PyOpenGL destructor warnings (optional)
        try:
            from OpenGL import error as _glerr
            _glerr.ERROR_CHECKING = False
        except Exception:
            pass

        logger.info(
            f"🎬 Initialized MyoVideoCallback (OSMesa)\n"
            f"   • env_id: {self.eval_env_id}\n"
            f"   • eval_freq: {self.eval_freq}\n"
            f"   • video_dir: {self.video_dir}\n"
            f"   • best_model_dir: {self.best_model_dir}"
        )

    # ------------------------------------------------------------
    #   SB3 Lifecycle
    # ------------------------------------------------------------
    def _init_callback(self) -> None:
        """Called once before training starts."""
        if self.eval_env is None:
            self.eval_env = gym.make(self.eval_env_id, render_mode="rgb_array")
            logger.info(f"Initialized eval env: {self.eval_env_id}")

    def _on_step(self) -> bool:
        """Called after each environment step."""
        if (self.n_calls % self.eval_freq) != 0:
            return True
        self._evaluate_and_record(self.model, self.n_calls)
        return True

    # ------------------------------------------------------------
    #   Core Evaluation + Recording
    # ------------------------------------------------------------
    def _evaluate_and_record(self, model, step_count: int):
        """Run evaluation and record .mp4 video."""
        if self.eval_env is None:
            self.eval_env = gym.make(self.eval_env_id, render_mode="rgb_array")

        # --- 1️⃣ numeric evaluation ---
        try:
            mean_reward, std_reward = evaluate_policy(
                model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                render=False,
            )
            logger.info(f"🎯 Step {step_count}: mean_reward={mean_reward:.3f} ± {std_reward:.3f}")
        except Exception as e:
            logger.warning(f"⚠️ evaluate_policy failed: {e}")
            mean_reward = -np.inf

        # --- 2️⃣ best model saving ---
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            best_path = os.path.join(self.best_model_dir, "best_model.zip")
            model.save(best_path)
            logger.info(f"💾 New best model saved: {best_path}")

        # --- 3️⃣ record deterministic rollout ---
        video_path = os.path.abspath(os.path.join(self.video_dir, f"step_{step_count}_r{mean_reward:.2f}"))
        os.makedirs(video_path, exist_ok=True)
        video_file = os.path.join(video_path, f"eval_{step_count}.mp4")

        try:
            writer = imageio.get_writer(video_file, fps=30, codec="libx264", quality=8, macro_block_size=None)

            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            n_frames = 0
            for _ in range(300):
                action, _ = model.predict(obs, deterministic=True)
                step_out = self.eval_env.step(action)
                if len(step_out) == 5:
                    obs, _, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    obs, _, done, _ = step_out

                try:
                    frame = self.eval_env.render()
                    if frame is not None:
                        if frame.dtype != np.uint8:
                            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                        writer.append_data(frame)
                        n_frames += 1
                except Exception as re:
                    logger.warning(f"⚠️ render() failed: {re}")

                if done:
                    obs = self.eval_env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]

            writer.close()
            if n_frames > 0 and os.path.exists(video_file):
                size_mb = os.path.getsize(video_file) / (1024 * 1024)
                logger.info(f"✅ Video saved: {video_file} ({n_frames} frames, {size_mb:.2f} MB)")
            else:
                logger.warning("⚠️ No frames rendered — file may be empty or missing.")

        except Exception as e:
            logger.warning(f"⚠️ Video recording failed at step {step_count}: {e}")
