import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import Config
from hrl.worker_env import TableTennisWorker
from hrl.manager_env import TableTennisManager


# ============================================================
# CHECK 1: Worker basic reset & step
# ============================================================
def check_worker_basic():
    print("\n[CHECK 1] Worker basic reset & step")

    cfg = Config()
    env = TableTennisWorker(cfg)

    obs, info = env.reset()
    print("Worker obs shape:", obs.shape)

    assert obs.shape == (24,), "Worker obs must be 24D (18 state + 6 goal)"

    act = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(act)

    assert np.isfinite(obs).all(), "Obs contains NaNs"
    assert np.isfinite(reward), "Reward is NaN"

    print("Reward:", reward)
    print("✔ Worker basic check passed")


# ============================================================
# CHECK 2: Worker goal sanity
# ============================================================
def check_worker_goal():
    print("\n[CHECK 2] Worker goal sanity")

    cfg = Config()
    env = TableTennisWorker(cfg)
    env.reset()

    g = env.current_goal
    print("Current goal (phys):", g)

    assert g.shape == (6,), "Goal must be 6D"

    # dt sanity (based on goal normalization ranges)
    dt_center = env.goal_center[5]
    dt_half = env.goal_half_range[5]
    assert dt_center - dt_half <= g[5] <= dt_center + dt_half, "dt out of range"

    # Paddle normal X should face -X
    assert g[3] <= 0.0, "nx should face -X"

    print("✔ Worker goal sanity passed")


# ============================================================
# CHECK 3: Worker prediction sanity
# ============================================================
def check_worker_prediction():
    print("\n[CHECK 3] Worker prediction sanity")

    cfg = Config()
    env = TableTennisWorker(cfg)
    env.reset()

    obs_dict = env.env.unwrapped.obs_dict
    goal_norm = env.predict_goal_from_state(obs_dict)

    print("Predicted goal (norm):", goal_norm)

    assert goal_norm.shape == (6,)
    assert np.all(goal_norm >= -1.01) and np.all(goal_norm <= 1.01), \
        "Predicted goal not normalized"

    print("✔ Worker prediction sanity passed")


# ============================================================
# CHECK 4: Manager basic wiring
# ============================================================
def check_manager_basic():
    print("\n[CHECK 4] Manager basic wiring")

    cfg = Config()

    worker_env = DummyVecEnv([lambda: TableTennisWorker(cfg)])

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

    expected_dim = worker_env.observation_space.shape[0] + 1
    assert obs.shape == (expected_dim,), \
        f"Manager obs must be {expected_dim}D"

    action = np.zeros(6, dtype=np.float32)
    obs, reward, terminated, truncated, info = manager_env.step(action)

    assert np.isfinite(reward)
    assert "goal_delta_norm" in info

    print("Reward:", reward)
    print("✔ Manager basic wiring passed")


# ============================================================
# CHECK 5: Short HRL rollout
# ============================================================
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


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    check_worker_basic()
    check_worker_goal()
    check_worker_prediction()
    check_manager_basic()
    check_short_rollout()

    print("\n🎉 ALL SANITY CHECKS PASSED — SAFE TO TRAIN 🎉")