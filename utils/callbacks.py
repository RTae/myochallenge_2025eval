import os
import numpy as np
import imageio
import traceback
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from myosuite.utils import gym


class VideoCallback(BaseCallback):
    """
    Stable-Baselines3 Video Callback using MyoSuite's offscreen renderer.
    """

    def __init__(
        self,
        eval_env_id="myoChallengeTableTennisP2-v0",
        eval_freq=10_000,
        video_dir="./logs/sb3_videos",
        best_model_dir="./logs/sb3_best",
        n_eval_episodes=3,
        camera_id=0,
        width=640,
        height=480,
        max_frames=300,
        verbose=1,
        num_worker=1
    ):
        super().__init__(verbose)
        self.eval_env_id = eval_env_id
        self.eval_freq = eval_freq
        self.video_dir = video_dir
        self.best_model_dir = best_model_dir
        self.n_eval_episodes = n_eval_episodes
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.max_frames = max_frames
        self.best_mean_reward = -np.inf
        self.eval_env = None
        self.num_worker = num_worker

        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)

        logger.info(
            f"🎬 Initialized MyoSuite VideoCallback (offscreen)\n"
            f"   • env_id: {self.eval_env_id}\n"
            f"   • eval_freq: {self.eval_freq}\n"
            f"   • camera_id: {self.camera_id}\n"
            f"   • video_dir: {self.video_dir}\n"
            f"   • best_model_dir: {self.best_model_dir}"
        )

    # ------------------------------------------------------------
    # SB3 Lifecycle
    # ------------------------------------------------------------
    def _init_callback(self):
        # not used for env creation now; handled inside _evaluate_and_record
        pass

    def _on_step(self):
        if (self.n_calls % self.eval_freq) != 0:
            return True
        self._evaluate_and_record(self.model, self.n_calls)
        return True

    # ------------------------------------------------------------
    # Evaluation + Offscreen Video Recording
    # ------------------------------------------------------------
    def _evaluate_and_record(self, model, step_count: int):
        # --- Always create a fresh evaluation env in this process ---
        try:
            if self.eval_env is not None:
                self.eval_env.close()
            self.eval_env = gym.make(self.eval_env_id)
            logger.info(f"♻️  Recreated eval env (PID={os.getpid()}) at step {step_count}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create eval env: {e}")
            return True

        # --- Numeric evaluation ---
        try:
            mean_reward, std_reward = evaluate_policy(
                model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                render=False,
            )
            logger.info(f"🎯 Step {int(step_count*self.num_worker)}: mean={mean_reward:.3f} ± {std_reward:.3f}")
        except Exception as e:
            logger.warning(f"⚠️ evaluate_policy failed: {e}")
            mean_reward = -np.inf

        # --- Save best model ---
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            best_path = os.path.join(self.best_model_dir, "best_model.zip")
            model.save(best_path)
            logger.info(f"💾 New best model saved: {best_path}")

        # --- Video recording ---
        video_file = os.path.join(self.video_dir, f"eval_step{int(step_count*self.num_worker)}_r{mean_reward:.2f}.mp4")

        try:
            logger.info("🎥 Starting video recording...")
            writer = imageio.get_writer(
                video_file, fps=30, codec="libx264", quality=8, macro_block_size=None
            )

            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            n_frames = 0
            terminated = False
            truncated = False

            while not (terminated or truncated) and n_frames < self.max_frames:
                action, _ = model.predict(obs, deterministic=True)
                step_out = self.eval_env.step(action)

                if len(step_out) == 5:
                    obs, _, terminated, truncated, _ = step_out
                else:
                    obs, _, done, _ = step_out
                    terminated = done
                    truncated = False

                # --- Offscreen render (headless-safe) ---
                try:
                    frame = self.eval_env.unwrapped.sim.renderer.render_offscreen(
                        width=self.width, height=self.height, camera_id=self.camera_id
                    )
                    if frame is not None:
                        if frame.dtype != np.uint8:
                            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                        writer.append_data(frame)
                        n_frames += 1
                        if n_frames % 50 == 0:
                            logger.info(f"   • Recorded {n_frames} frames...")
                except Exception:
                    logger.warning(f"⚠️ render_offscreen() failed:\n{traceback.format_exc()}")
                    break

                if terminated or truncated:
                    obs = self.eval_env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]

            writer.close()
            if n_frames > 0 and os.path.exists(video_file):
                size_mb = os.path.getsize(video_file) / (1024 * 1024)
                logger.info(f"✅ Saved video: {video_file} ({n_frames} frames, {size_mb:.2f} MB)")
            else:
                logger.warning("⚠️ No frames rendered — empty or missing file.")

        except Exception as e:
            logger.warning(
                f"⚠️ Video recording failed at step {step_count}: {e}\n{traceback.format_exc()}"
            )

class MetricCallback(BaseCallback):
    """
    Custom metrics for MyoSuite environments:
      - episodic return (dense reward sum)
      - success rate (based on 'solved' flag)
      - actor loss, critic loss, entropy loss
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = {}
        self.episode_successes = {}
        self.last_actor_loss = None
        self.last_critic_loss = None

    def _on_rollout_end(self):
        """Capture PPO losses after each rollout."""
        try:
            if hasattr(self.model.logger, "name_to_value"):
                self.last_actor_loss = self.model.logger.name_to_value.get("train/actor_loss")
                self.last_critic_loss = self.model.logger.name_to_value.get("train/critic_loss")
        except Exception as e:
            logger.warning(f"Failed to get losses: {e}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for idx, info in enumerate(infos):
            if "rwd_dense" not in info:
                continue

            # accumulate episodic reward per env
            self.episode_rewards[idx] = self.episode_rewards.get(idx, 0.0) + info["rwd_dense"]

            # when an episode ends
            if info.get("done", False):
                ep_ret = self.episode_rewards.pop(idx, 0.0)
                solved = float(info.get("solved", 0.0))
                self.episode_successes[idx] = solved

                # log custom metrics
                self.model.logger.record("custom/return_vs_timestep", ep_ret)
                self.model.logger.record("custom/success_rate_vs_step", solved)

        # log PPO losses
        if self.last_actor_loss is not None:
            self.model.logger.record("custom/actor_loss", self.last_actor_loss)
        if self.last_critic_loss is not None:
            self.model.logger.record("custom/critic_loss", self.last_critic_loss)

        return True