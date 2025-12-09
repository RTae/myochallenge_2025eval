# train_manager.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from manager_env import ManagerEnv
from config import Config

def make_env():
    return ManagerEnv(Config(), worker_model_path="worker.zip")

def main():
    cfg = Config()
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_lambda,
        learning_rate=cfg.ppo_lr,
        clip_range=cfg.ppo_clip,
        n_epochs=cfg.ppo_epochs,
    )

    model.learn(total_timesteps=cfg.total_timesteps)
    model.save("manager.zip")
    env.close()

if __name__ == "__main__":
    main()
