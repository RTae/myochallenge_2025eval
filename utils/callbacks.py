import os
import numpy as np
import imageio
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from loguru import logger
from myosuite.utils import gym


class VideoEvalCallback(BaseCallback):
    """Periodic evaluation + .mp4 recording with persistent EGL renderer."""

    def __init__(self, eval_env_id, eval_freq, video_dir, best_model_dir, n_eval_episodes=3, verbose=1):
        super().__init__(verbose)
        self.eval_env_id = eval_env_id
        self.eval_freq = eval_freq
        self.video_dir = video_dir
        self.best_model_dir = best_model_dir
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.eval_env = None
        self.renderer = None  # <- keep persistent

    def _init_callback(self) -> None:
        self.eval_env = gym.make(self.eval_env_id)
        # get inner mujoco env
        inner_env = self.eval_env
        while hasattr(inner_env, "env"):
            inner_env = inner_env.env
        self.sim = inner_env.sim
        try:
            self.renderer = self.sim.renderer  # persistent EGL renderer
            logger.info(f"✅ Persistent EGL renderer initialized for {self.eval_env_id}")
        except Exception as e:
            logger.warning(f"⚠️ Could not create persistent renderer: {e}")
        logger.info(f"Initialized evaluation environment: {self.eval_env_id}")

    def _on_step(self) -> bool:
        if (self.n_calls % self.eval_freq) != 0:
            return True

        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
        )
        logger.info(f"🎯 Step {self.n_calls}: mean_reward={mean_reward:.3f} ± {std_reward:.3f}")

        # Save best model
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            best_model_path = os.path.join(self.best_model_dir, "best_model.zip")
            self.model.save(best_model_path)
            logger.info(f"💾 New best model saved: {best_model_path}")

        # Record video safely
        video_path = os.path.join(self.video_dir, f"step_{self.n_calls}_r{mean_reward:.2f}")
        os.makedirs(video_path, exist_ok=True)
        video_file = os.path.join(video_path, f"eval_{self.n_calls}.mp4")

        try:
            logger.info(f"🎥 Recording video to {video_file}")
            writer = imageio.get_writer(video_file, fps=30)

            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            for _ in range(300):
                action, _ = self.model.predict(obs, deterministic=True)
                step_out = self.eval_env.step(action)
                if len(step_out) == 5:
                    obs, _, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    obs, _, done, _ = step_out

                # use persistent renderer
                frame = self.renderer.render_offscreen(width=640, height=480)
                writer.append_data(frame)

                if done:
                    obs = self.eval_env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]

            writer.close()
            logger.info(f"✅ Video saved successfully: {video_file}")

        except Exception as e:
            logger.warning(f"⚠️ Skipping video at step {self.n_calls} due to render error: {e}")

        return True
