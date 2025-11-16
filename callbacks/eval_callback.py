import os
import numpy as np
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from myosuite.utils import gym


class EvalCallback:

    def __init__(self, env_id, seed, eval_freq=5000, eval_episodes=3, logdir="./logs"):
        self.env_id = env_id
        self.seed = seed
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes

        self.total_steps = 0
        self.last_eval = 0

        self.predict_fn = None
        self.writer = SummaryWriter(os.path.join(logdir, "tb"))

    def attach_predictor(self, fn):
        self.predict_fn = fn

    def _init_callback(self, model=None):
        self.eval_env = gym.make(self.env_id)
        self.eval_env.reset(seed=self.seed)
        logger.info("EvalCallback initialized.")

    def _episode_success(self, obs_dict):
        return float(obs_dict["touching_info"][0] > 0.5)

    def run_evaluation(self):

        rewards = []
        successes = []

        for _ in range(self.eval_episodes):

            obs, info = self.eval_env.reset(seed=self.seed)
            ep_reward = 0
            success = 0

            while True:
                obs_dict = self.eval_env.unwrapped.get_obs_dict(self.eval_env.unwrapped.sim)

                act = self.predict_fn(obs, self.eval_env)
                obs, rew, terminated, truncated, info = self.eval_env.step(act)

                ep_reward += rew
                success = max(success, self._episode_success(obs_dict))

                if terminated or truncated:
                    break

            rewards.append(ep_reward)
            successes.append(success)

        mean_r = float(np.mean(rewards))
        success_rate = float(np.mean(successes))

        logger.info(f"[Eval] step {self.total_steps} | mean={mean_r:.2f} | success={success_rate:.2f}")

        self.writer.add_scalar("eval/reward", mean_r, self.total_steps)
        self.writer.add_scalar("eval/success", success_rate, self.total_steps)

    def _on_step(self):
        self.total_steps += 1

        if self.total_steps - self.last_eval >= self.eval_freq:
            self.last_eval = self.total_steps
            self.run_evaluation()

    def _on_training_end(self):
        logger.info("Evaluation finished.")
        self.eval_env.close()
        self.writer.close()
