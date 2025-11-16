import os
import numpy as np
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from myosuite.utils import gym


class EvalCallback:
    def __init__(
        self,
        env_id,
        seed,
        eval_freq=5000,
        eval_episodes=3,
        logdir="./logs"
    ):
        self.env_id = env_id
        self.seed = seed
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes

        self.total_steps = 0
        self.last_eval_step = 0

        self.writer = SummaryWriter(os.path.join(logdir, "tb"))
        self.eval_env = None
        self.predict_fn = None

    def attach_predictor(self, fn):
        self.predict_fn = fn

    def _init_callback(self, model=None):
        self.eval_env = gym.make(self.env_id)
        self.eval_env.reset(seed=self.seed)
        logger.info("EvalCallback initialized.")

    def _episode_success(self, obs_dict):
        touching = obs_dict["touching_info"]
        return float(touching[0] > 0.5)

    def run_evaluation(self):
        rewards = []
        successes = []

        for _ in range(self.eval_episodes):
            obs, info = self.eval_env.reset(seed=self.seed)
            ep_reward = 0.0
            ep_success = 0.0

            while True:
                sim = self.eval_env.unwrapped.sim
                obs_dict = self.eval_env.unwrapped.get_obs_dict(sim)

                if self.predict_fn is not None:
                    act = self.predict_fn(obs, self.eval_env)
                else:
                    act = np.zeros(self.eval_env.action_space.shape[0], dtype=np.float32)

                obs, rew, terminated, truncated, info = self.eval_env.step(act)
                ep_reward += rew
                ep_success = max(ep_success, self._episode_success(obs_dict))

                if terminated or truncated:
                    break

            rewards.append(ep_reward)
            successes.append(ep_success)

        mean_reward = float(np.mean(rewards))
        success_rate = float(np.mean(successes))

        logger.info(
            f"[Eval] Step {self.total_steps} | Mean={mean_reward:.3f} | Success={success_rate*100:.1f}%"
        )

        self.writer.add_scalar("eval/mean_reward", mean_reward, self.total_steps)
        self.writer.add_scalar("eval/success_rate", success_rate, self.total_steps)
        self.writer.flush()

    def _on_step(self):
        self.total_steps += 1
        if self.total_steps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.total_steps
            self.run_evaluation()

    def _on_training_end(self):
        logger.info("Evaluation finished.")
        if self.eval_env is not None:
            self.eval_env.close()
        self.writer.close()
