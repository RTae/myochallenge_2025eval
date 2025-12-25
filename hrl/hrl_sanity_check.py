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

    # dt bounds (from normalization)
    dt_center = env.goal_center[5]
    dt_half = env.goal_half_range[5]
    assert dt_center - dt_half <= g[5] <= dt_center + dt_half

    # paddle normal must face -X
    assert g[3] <= 0.0, "nx must face -X"

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

    assert goal_norm.shape == (6,)
    assert np.all(goal_norm >= -1.01)
    assert np.all(goal_norm <= 1.01)

    print("✔ Worker prediction sanity passed")


# ============================================================
# CHECK 4: Manager → Worker wiring (READ-ONLY)
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

    obs, _ = manager_env.reset()
    expected_dim = worker_env.observation_space.shape[0] + 1

    assert obs.shape == (expected_dim,), "Manager obs shape mismatch"

    # zero delta
    obs, reward, terminated, truncated, info = manager_env.step(
        np.zeros(6, dtype=np.float32)
    )

    assert np.isfinite(reward)
    assert "goal_delta_norm" in info

    print("✔ Manager basic wiring passed")


# ============================================================
# CHECK 5: Progress is READ-ONLY in manager
# ============================================================
def check_manager_progress_readonly():
    print("\n[CHECK 5] Manager progress is read-only")

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
    )

    manager_env.reset()

    progress = worker_env.env_method("get_progress")
    print("Worker progress:", progress)

    assert isinstance(progress, list)
    assert len(progress) == 1
    assert 0.0 <= progress[0] <= 1.0

    print("✔ Progress read-only check passed")


# ============================================================
# CHECK 6: Short HRL rollout
# ============================================================
def check_short_rollout():
    print("\n[CHECK 6] Short HRL rollout")

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
    check_manager_progress_readonly()
    check_short_rollout()

    print("\n🎉 ALL SANITY CHECKS PASSED — SAFE TO TRAIN 🎉")