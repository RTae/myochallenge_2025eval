import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from myosuite.utils import gym

from config import Config
from hrl.worker_env import TableTennisWorker
from hrl.manager_env import TableTennisManager

def check_worker_basic():
    print("\n[CHECK 1] Worker basic reset & step")

    cfg = Config()
    env = TableTennisWorker(cfg)

    obs, info = env.reset()
    print("Worker obs shape:", obs.shape)
    assert obs.shape == (18,), "Worker obs must be 18D"

    # Random action (same dim as muscle controller)
    act = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(act)
    print("Reward:", reward)
    print("Terminated:", terminated, "Truncated:", truncated)

    assert np.isfinite(obs).all(), "Obs contains NaNs"
    assert np.isfinite(reward), "Reward is NaN"

    print("✔ Worker basic check passed")
    
def check_worker_goal():
    print("\n[CHECK 2] Worker goal sanity")

    cfg = Config()
    env = TableTennisWorker(cfg)
    obs, info = env.reset()

    g = env.current_goal
    print("Current goal (phys):", g)

    assert g.shape == (6,)
    assert g[5] >= env.dt_min and g[5] <= env.dt_max, "dt out of range"
    assert g[3] <= 0.0, "nx should face -X"

    print("✔ Worker goal sanity passed")
    
def check_worker_prediction():
    print("\n[CHECK 3] Worker prediction sanity")

    cfg = Config()
    env = TableTennisWorker(cfg)
    obs, info = env.reset()

    obs_dict = env.env.unwrapped.obs_dict

    goal_norm = env.predict_goal_from_state(obs_dict)
    print("Predicted goal (norm):", goal_norm)

    assert goal_norm.shape == (6,)
    assert np.all(goal_norm >= -1.01) and np.all(goal_norm <= 1.01)

    print("✔ Worker prediction sanity passed")
    

def check_manager_basic():
    print("\n[CHECK 4] Manager basic wiring")

    cfg = Config()

    # Worker vec env
    worker_env = DummyVecEnv([lambda: TableTennisWorker(cfg)])

    # Dummy frozen worker policy (random for now)
    worker_model = PPO(
        "MlpPolicy",
        worker_env,
        n_steps=16,
        batch_size=16,
        learning_rate=1e-4,
        verbose=0,
    )

    manager_env = TableTennisManager(
        worker_env=worker_env,
        worker_model=worker_model,
        config=cfg,
        decision_interval=5,
        max_episode_steps=50,
    )

    obs, info = manager_env.reset()
    print("Manager obs shape:", obs.shape)
    assert obs.shape == (19,), "Manager obs must be 19D"

    action = np.zeros(6, dtype=np.float32)
    obs, reward, terminated, truncated, info = manager_env.step(action)

    print("Reward:", reward)
    print("Info:", info)

    assert np.isfinite(reward)
    assert "goal_delta_norm" in info

    print("✔ Manager basic wiring passed")
    
def check_short_rollout():
    print("\n[CHECK 5] Short HRL rollout")

    cfg = Config()
    worker_env = DummyVecEnv([lambda: TableTennisWorker(cfg)])

    worker_model = PPO(
        "MlpPolicy",
        worker_env,
        n_steps=32,
        batch_size=32,
        learning_rate=1e-4,
        verbose=0,
    )

    manager_env = TableTennisManager(
        worker_env=worker_env,
        worker_model=worker_model,
        config=cfg,
        decision_interval=5,
        max_episode_steps=100,
    )

    obs, _ = manager_env.reset()

    for i in range(10):
        action = manager_env.action_space.sample()
        obs, reward, term, trunc, info = manager_env.step(action)
        print(f"Step {i}: reward={reward:.3f}")

        if term or trunc:
            break

    print("✔ Short rollout passed")

if __name__ == "__main__":
    check_worker_basic()
    check_worker_goal()
    check_worker_prediction()
    check_manager_basic()
    check_short_rollout()

    print("\n🎉 ALL SANITY CHECKS PASSED — SAFE TO TRAIN 🎉")