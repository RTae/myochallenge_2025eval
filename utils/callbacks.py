import os
import numpy as np
import imageio
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from myosuite.utils import gym


class MyoUnifiedVideoCallback(BaseCallback, DefaultCallbacks):
    """
    Unified callback for both Stable-Baselines3 and Ray RLlib.

    ✨ Features:
      - Periodic evaluation (SB3) or iteration-based evaluation (RLlib)
      - Headless EGL-safe .mp4 recording
      - Optional best-model saving (SB3)
      - Per-worker EGL GPU pinning (RLlib)
    """

    def __init__(
        self,
        eval_env_id: str = "myoChallengeTableTennisP2-v0",
        eval_freq: int = 10_000,
        video_dir: str = "./logs/rllib_videos",
        best_model_dir: str = "./logs/rllib_best",
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
        self._is_rllib = False

        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)

        logger.info(
            f"🎬 Initialized MyoUnifiedVideoCallback\n"
            f"   • env_id: {self.eval_env_id}\n"
            f"   • eval_freq: {self.eval_freq}\n"
            f"   • video_dir: {self.video_dir}\n"
            f"   • best_model_dir: {self.best_model_dir}"
        )

    # ============================================================
    #  COMMON HELPERS
    # ============================================================
    def _init_env(self):
        """Ensure evaluation env is created."""
        if self.eval_env is None:
            self.eval_env = gym.make(self.eval_env_id)
            logger.info(f"Initialized evaluation environment: {self.eval_env_id}")

    # ============================================================
    #  SB3 CALLBACK METHODS
    # ============================================================
    def _init_callback(self) -> None:
        """Called by SB3 before training begins."""
        self._init_env()

    def _on_step(self) -> bool:
        """Called by SB3 at each environment step."""
        if (self.n_calls % self.eval_freq) != 0:
            return True
        self._evaluate_and_record(self.model, self.n_calls)
        return True

    # ============================================================
    #  RLlib CALLBACK METHODS (override subset only)
    # ============================================================
    def on_algorithm_init(self, *, algorithm, **kwargs):
        """Called once when RLlib algorithm is created."""
        self._is_rllib = True
        self._init_env()
        try:
            if not self.eval_env_id:
                env_obj = algorithm.workers.local_worker().env
                self.eval_env_id = env_obj.spec.id if hasattr(env_obj, "spec") else str(env_obj)
        except Exception:
            pass
        logger.info(f"✅ MyoUnifiedVideoCallback active under RLlib for {self.eval_env_id}")

    def on_sub_environment_created(self, *, worker, sub_environment, env_context, **kwargs):
        """Pin EGL device to worker GPU if available."""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            first_gpu = cuda_visible.split(",")[0].strip()
            if os.environ.get("EGL_DEVICE_ID") != first_gpu:
                os.environ["EGL_DEVICE_ID"] = first_gpu
                logger.info(
                    f"📌 Set EGL_DEVICE_ID={first_gpu} "
                    f"(CUDA_VISIBLE_DEVICES={cuda_visible})"
                )

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Runs after each RLlib iteration."""
        try:
            timesteps = int(result.get("timesteps_total", 0))
            if "evaluation" not in result:
                return

            policy = algorithm.get_policy()
            model = type("ModelStub", (), {})()
            model.predict = lambda obs, deterministic=True: (
                policy.compute_single_action(obs, explore=False)[0],
                None,
            )

            self._evaluate_and_record(model, timesteps)
        except Exception as e:
            logger.warning(f"⚠️ Video callback error in RLlib: {e}")

    # ============================================================
    #  SHARED LOGIC (EVALUATION + VIDEO RECORDING)
    # ============================================================
    def _evaluate_and_record(self, model, step_count: int):
        """Perform evaluation and record an .mp4 video."""
        self._init_env()

        mean_reward, std_reward = (0.0, 0.0)

        # Only run numeric evaluation for SB3
        if evaluate_policy is not None and not self._is_rllib:
            mean_reward, std_reward = evaluate_policy(
                model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )
            logger.info(f"🎯 Step {step_count}: mean_reward={mean_reward:.3f} ± {std_reward:.3f}")

        # Save best model (SB3 only)
        if not self._is_rllib and mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            best_path = os.path.join(self.best_model_dir, "best_model.zip")
            model.save(best_path)
            logger.info(f"💾 New best model saved: {best_path}")

        # --- absolute path fix ---
        video_path = os.path.abspath(os.path.join(self.video_dir, f"step_{step_count}_r{mean_reward:.2f}"))
        os.makedirs(video_path, exist_ok=True)
        video_file = os.path.join(video_path, f"eval_{step_count}.mp4")

        try:
            import imageio

            # ✅ Use legacy writer API with explicit codec
            writer = imageio.get_writer(video_file, fps=30, codec="libx264", quality=8, macro_block_size=None)

            inner_env = self.eval_env
            while hasattr(inner_env, "env"):
                inner_env = inner_env.env

            sim = getattr(inner_env, "sim", None)
            renderer = getattr(sim, "renderer", None)

            if renderer is None:
                logger.warning("⚠️ No Mujoco renderer found in eval_env; skipping video.")
                writer.close()
                return

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
                    frame = renderer.render_offscreen(width=640, height=480)
                    if frame is not None:
                        writer.append_data(frame)
                        n_frames += 1
                except Exception as re:
                    logger.warning(f"⚠️ Render frame failed: {re}")
                    continue

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