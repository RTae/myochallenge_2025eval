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
        self.logdir = logdir

        self.last_eval_step = 0
        self.total_steps = 0

        self.writer = SummaryWriter(os.path.join(logdir, "tb"))
        self.eval_env = None

        self.predict_fn = None

    def attach_predictor(self, fn):
        self.predict_fn = fn

    def _init_callback(self, model=None):
        self.eval_env = gym.make(self.env_id)
        self.eval_env.reset(seed=self.seed)
        logger.info("EvalCallback initialized.")

    def _episode_success(self, obs, info):
        for k in ["success", "is_success", "done_success", "task_success"]:
            if k in info:
                return bool(info[k])
        return False

    def run_evaluation(self):

        rewards = []
        successes = []

        for _ in range(self.eval_episodes):

            obs, info = self.eval_env.reset(seed=self.seed)
            ep_reward = 0
            ep_success = False

            while True:

                action = self.predict_fn(obs, self.eval_env)

                obs, rew, terminated, truncated, info = self.eval_env.step(action)
                ep_reward += rew

                if self._episode_success(obs, info):
                    ep_success = True

                if terminated or truncated:
                    break

            rewards.append(ep_reward)
            successes.append(1 if ep_success else 0)

        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards))
        succ_r = float(np.mean(successes))

        logger.info(
            f"[Eval] Step {self.total_steps} | Mean={mean_r:.3f} | Success={succ_r*100:.1f}%"
        )

        self.writer.add_scalar("eval/mean_reward", mean_r, self.total_steps)
        self.writer.add_scalar("eval/std_reward",  std_r, self.total_steps)
        self.writer.add_scalar("eval/success_rate", succ_r, self.total_steps)
        self.writer.flush()

    def _on_step(self):
        self.total_steps += 1
        if self.total_steps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.total_steps
            self.run_evaluation()

    def _on_training_end(self):
        logger.info("Evaluation finished.")
        if self.eval_env:
            self.eval_env.close()
        self.writer.close()
