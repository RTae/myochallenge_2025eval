import os
from myosuite.utils import gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder
from loguru import logger

class VideoEvalCallback(BaseCallback):
    """
    Custom callback for periodic video evaluation.
    Records a short evaluation episode as an .mp4 file and saves the best model.
    """

    def __init__(self, eval_env_id, eval_freq, video_dir, best_model_dir, n_eval_episodes=3, verbose=1):
        super().__init__(verbose)
        self.eval_env_id = eval_env_id
        self.eval_freq = eval_freq
        self.video_dir = video_dir
        self.best_model_dir = best_model_dir
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        self.eval_env = gym.make(self.eval_env_id)
        if self.verbose:
            logger.info(f"Initialized evaluation environment: {self.eval_env_id}")

    def _on_step(self) -> bool:
        # Run every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )

            logger.info(f"\n🎯 Step {self.n_calls}: mean_reward={mean_reward:.3f} ± {std_reward:.3f}")

            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = os.path.join(self.best_model_dir, "best_model.zip")
                self.model.save(best_model_path)
                logger.info(f"💾 New best model saved: {best_model_path} (mean_reward={mean_reward:.3f})")

            # Record short video
            video_path = os.path.join(self.video_dir, f"step_{self.n_calls}_r{mean_reward:.2f}")
            record_env = VecVideoRecorder(
                make_vec_env(self.eval_env_id, n_envs=1),
                video_folder=video_path,
                record_video_trigger=lambda step: step == 0,
                video_length=300,   # about 10 seconds
                name_prefix=f"eval_{self.n_calls}",
            )

            obs = record_env.reset()
            for _ in range(300):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = record_env.step(action)
                if done.any():
                    obs = record_env.reset()
            record_env.close()
            logger.info(f"🎥 Video recorded and saved to: {video_path}")
        return True
