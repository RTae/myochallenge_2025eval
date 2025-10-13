import os
import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from myosuite.utils import gym


class VideoCallback(BaseCallback):
    """
    SB3-only callback:
      - Periodic evaluation with mean reward logging
      - EGL-safe .mp4 recording
      - Optional best-model saving handled by EvalCallback; here we keep a simple threshold if needed
    """

    def __init__(
        self,
        eval_env_id: str = "myoChallengeTableTennisP2-v0",
        eval_freq: int = 100_000,
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

        logger.info(
            f"🎬 Initialized MyoSB3VideoCallback\n"
            f"   • env_id: {self.eval_env_id}\n"
            f"   • eval_freq: {self.eval_freq}\n"
            f"   • video_dir: {self.video_dir}\n"
            f"   • best_model_dir: {self.best_model_dir}"
        )

    # ============================================================
    #  SB3 CALLBACK METHODS
    # ============================================================
    def _init_callback(self) -> None:
        """Called by SB3 before training begins."""
        if self.eval_env is None:
            self.eval_env = gym.make(self.eval_env_id)
            logger.info(f"Initialized evaluation environment: {self.eval_env_id}")

    def _on_step(self) -> bool:
        """Called by SB3 at each environment step (aggregated across vec envs)."""
        if (self.num_timesteps % self.eval_freq) != 0:
            return True
        self._evaluate_and_record(self.model, self.num_timesteps)
        return True

    # ============================================================
    #  SHARED LOGIC (EVALUATION + VIDEO)
    # ============================================================
    def _evaluate_and_record(self, model, step_count: int):
        """Run evaluation and record an .mp4 video via Mujoco offscreen renderer."""
        if self.eval_env is None:
            self.eval_env = gym.make(self.eval_env_id)

        # --- numeric evaluation ---
        try:
            mean_reward, std_reward = evaluate_policy(
                model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )
            logger.info(f"🎯 Step {step_count}: mean_reward={mean_reward:.3f} ± {std_reward:.3f}")
        except Exception as e:
            logger.warning(f"⚠️ evaluate_policy failed: {e}")
            mean_reward = -np.inf

        # --- video output setup ---
        video_path = os.path.abspath(os.path.join(self.video_dir, f"step_{step_count}_r{float(mean_reward):.2f}"))
        os.makedirs(video_path, exist_ok=True)
        video_file = os.path.join(video_path, f"eval_{step_count}.mp4")

        try:
            import imageio
            writer = imageio.get_writer(video_file, fps=30, codec="libx264", quality=8, macro_block_size=None)

            # 获取底层的 Mujoco 环境
            inner_env = self.eval_env
            while hasattr(inner_env, "env"):
                inner_env = inner_env.env

            sim = getattr(inner_env, "sim", None)
            renderer = getattr(sim, "renderer", None)

            if renderer is None:
                logger.info("Renderer not found in sim, it will be created on first render call.")

            # reset obs (support both old/new gym API)
            obs, _ = (
                self.eval_env.reset(return_info=True)
                if "return_info" in self.eval_env.reset.__code__.co_varnames
                else (self.eval_env.reset(), {})
            )
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
                    frame = self.eval_env.render(mode='rgb_array', width=640, height=480)
                    
                    if frame is None:
                        renderer = getattr(inner_env.sim, "renderer", None)
                        if renderer:
                            frame = renderer.render_offscreen(width=640, height=480)

                    if frame is not None:
                        writer.append_data(frame)
                        n_frames += 1
                except Exception as re:
                    logger.warning(f"⚠️ Render frame failed: {re}")
                    break

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
            import traceback
            logger.warning(traceback.format_exc())