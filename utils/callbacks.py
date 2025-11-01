import os
import traceback
import imageio
from tensorboardX import SummaryWriter
from config import Config

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from loguru import logger
from myosuite.utils import gym

# =====================================================
#  TENSORBOARD + VIDEO LOGGER
# =====================================================
class JaxVideoMetricLogger:
    def __init__(self, cfg: Config):
        os.makedirs(cfg.logdir, exist_ok=True)
        self.writer = SummaryWriter(cfg.logdir)
        self.cfg = cfg
        self.video_dir = os.path.join(cfg.logdir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.global_step = 0
        self.best_reward = -np.inf

    def log_scalar(self, tag, val):
        self.writer.add_scalar(tag, val, self.global_step)

    def record_eval_video(self, env_id, policy_net, policy_params, manager_net, manager_params):
        if (self.global_step % self.cfg.video_freq) != 0:
            return
        logger.info(f"🎬 Recording evaluation video at step {self.global_step}")
        try:
            eval_env = gym.make(env_id)
            frames, ep_rews = [], []
            for ep in range(self.cfg.eval_episodes):
                obs, _ = eval_env.reset()
                done, trunc = False, False
                total_r = 0.0
                skill_id = select_skill(manager_params, manager_net, obs, self.cfg)
                skill_oh = one_hot(skill_id, self.cfg.skills)
                n_frames = 0
                while not (done or trunc) and n_frames < self.cfg.max_eval_frames:
                    obs_skill = np.concatenate([obs.astype(np.float32), skill_oh], axis=-1)
                    key = random.PRNGKey((n_frames + 99) % 2**31)
                    act, _, _ = policy_net.apply(policy_params, jnp.expand_dims(jnp.asarray(obs_skill), 0), key)
                    step_out = eval_env.step(np.asarray(act[0]))
                    obs, r, done, trunc, _ = step_out
                    total_r += float(r)
                    try:
                        frame = eval_env.unwrapped.sim.renderer.render_offscreen(
                            width=self.cfg.video_w, height=self.cfg.video_h, camera_id=self.cfg.camera_id
                        )
                        if frame is not None:
                            if frame.dtype != np.uint8:
                                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                            frames.append(frame)
                            n_frames += 1
                    except Exception:
                        logger.warning(traceback.format_exc())
                        break
                ep_rews.append(total_r)
            mean_r = float(np.mean(ep_rews)) if len(ep_rews) else -np.inf
            std_r = float(np.std(ep_rews)) if len(ep_rews) else 0.0
            video_path = os.path.join(self.video_dir, f"step{self.global_step}_r{mean_r:.2f}.mp4")
            if len(frames) > 0:
                imageio.mimsave(video_path, frames, fps=30)
                logger.info(f"✅ Saved video: {video_path} ({len(frames)} frames)")
            else:
                logger.warning("⚠️ No frames captured during evaluation.")
            self.log_scalar("eval/mean_reward", mean_r)
            self.log_scalar("eval/std_reward", std_r)
            if mean_r > self.best_reward:
                self.best_reward = mean_r
                # Save best policy params (np.savez with flattened dict leaves)
                flat, treedef = jax.tree_util.tree_flatten(policy_params)
                np.savez(os.path.join(self.cfg.logdir, "best_policy_params.npz"),
                         **{f"arr_{i}": np.asarray(x) for i, x in enumerate(flat)})
                with open(os.path.join(self.cfg.logdir, "best_policy_treedef.txt"), "w") as f:
                    f.write(str(treedef))
                logger.info("💾 New best policy checkpoint saved.")
            eval_env.close()
        except Exception as e:
            logger.warning(f"⚠️ Video recording failed: {e}")

    def close(self):
        self.writer.close()