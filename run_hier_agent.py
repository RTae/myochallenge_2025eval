# run_hier_agent.py
from stable_baselines3 import PPO
from manager_env import ManagerEnv
from config import Config

def main():
    cfg = Config()
    env = ManagerEnv(cfg, "worker.zip")
    manager = PPO.load("manager.zip")

    for ep in range(5):
        obs, _ = env.reset()
        done = False
        ep_ret = 0
        while not done:
            a, _ = manager.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(a)
            ep_ret += r
        print(f"Episode {ep}: return={ep_ret:.3f}")

if __name__ == "__main__":
    main()
