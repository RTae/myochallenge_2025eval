import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, Any, List

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from config import Config
from env_factory import build_manager_vec, build_worker_vec
from hrl.worker_env import TableTennisWorker
from lattice.ppo.policies import LatticeActorCriticPolicy

# Your helpers
from utils import resume_vecnormalize_on_training_env


# ==================================================
# Loaders (same as your train code)
# ==================================================
def load_worker_model(path: str):
    return PPO.load(path, device="cuda", policy=LatticeActorCriticPolicy)


def load_worker_vecnormalize(path: str, venv: TableTennisWorker) -> VecNormalize:
    env = DummyVecEnv([lambda: venv])
    vecnorm = VecNormalize.load(path, env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


# ==================================================
# Generic episode runner for VecEnv (num_envs=1 assumed)
# ==================================================
def run_episodes(model: PPO, env, n_episodes: int, deterministic: bool = True) -> Dict[str, Any]:
    """
    Runs exactly n_episodes and aggregates episode reward, success rate, effort.
    Assumes env is a VecEnv with num_envs=1 (like your eval usage).
    """
    ep_rewards: List[float] = []
    efforts: List[float] = []
    success_count = 0

    obs = env.reset()

    for _ in tqdm(range(n_episodes), desc="Evaluating"):
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)

            done = bool(dones[0])
            info = infos[0]

            if done:
                # SB3 VecMonitor-style episode info
                if "episode" in info and isinstance(info["episode"], dict) and "r" in info["episode"]:
                    ep_rewards.append(float(info["episode"]["r"]))
                else:
                    # fallback: if your env doesn’t provide "episode"
                    # you can accumulate rewards manually, but keeping simple here
                    ep_rewards.append(0.0)

                # optional custom keys
                efforts.append(float(info.get("effort", 0.0)))
                if bool(info.get("is_success", False)):
                    success_count += 1
                break

    ep_rewards = ep_rewards if len(ep_rewards) > 0 else [0.0]
    efforts = efforts if len(efforts) > 0 else [0.0]

    return {
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "success_rate": 100.0 * float(success_count) / float(n_episodes),
        "mean_effort": float(np.mean(efforts)),
        "std_effort": float(np.std(efforts)),
        "episodes": n_episodes,
    }


# ==================================================
# Build eval worker env (frozen VecNormalize)
# ==================================================
def make_eval_worker_env(cfg: Config, worker_vecnorm_path: Optional[str]):

    env = build_worker_vec(cfg=cfg, num_envs=1)

    # If you saved vecnormalize.pkl during training, load it onto eval env
    if worker_vecnorm_path and os.path.exists(worker_vecnorm_path):
        env = resume_vecnormalize_on_training_env(
            env,
            worker_vecnorm_path,
            training=False,
            norm_reward=False,
        )

    if isinstance(env, VecNormalize):
        env.training = False
        env.norm_reward = False

    return env


# ==================================================
# Build eval manager env (frozen worker inside)
# ==================================================
def make_eval_manager_env(
    cfg: Config,
    worker_model_path: str,
    worker_vecnorm_path: str,
    decision_interval: int,
    max_episode_steps: int,
):

    def worker_env_loader(path: str):
        return load_worker_vecnormalize(path, TableTennisWorker(cfg))

    env = build_manager_vec(
        cfg=cfg,
        num_envs=1,
        worker_model_loader=load_worker_model,
        worker_env_loader=worker_env_loader,
        worker_model_path=worker_model_path,
        worker_env_path=worker_vecnorm_path,
        decision_interval=decision_interval,
        max_episode_steps=max_episode_steps,
    )
    return env


# ==================================================
# Evaluate one HRL checkpoint folder
# Expected structure (like your train code):
#   <exp>/worker/worker_model.pkl
#   <exp>/worker/vecnormalize.pkl
#   <exp>/manager/manager_model.pkl
# OR “best” folders if you want:
#   <exp>/worker/best/best_model.zip
#   <exp>/manager/best/best_model.zip
# ==================================================
def evaluate_one_experiment(
    exp_dir: str,
    n_episodes: int,
    use_best: bool = False,
    eval_worker_too: bool = False,
) -> Dict[str, Any]:

    cfg = Config()  # keep consistent config creation
    cfg.logdir = exp_dir  # not required, but convenient

    worker_dir = os.path.join(exp_dir, "worker")
    manager_dir = os.path.join(exp_dir, "manager")

    if use_best:
        worker_model_path = os.path.join(worker_dir, "best", "best_model.zip")
        manager_model_path = os.path.join(manager_dir, "best", "best_model.zip")
    else:
        worker_model_path = os.path.join(worker_dir, "worker_model.pkl")
        manager_model_path = os.path.join(manager_dir, "manager_model.pkl")

    worker_vecnorm_path = os.path.join(worker_dir, "vecnormalize.pkl")

    # sanity checks
    for p in [worker_model_path, manager_model_path, worker_vecnorm_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    # ----- MANAGER eval -----
    manager_env = make_eval_manager_env(
        cfg=cfg,
        worker_model_path=worker_model_path,
        worker_vecnorm_path=worker_vecnorm_path,
        decision_interval=5,
        max_episode_steps=cfg.episode_len,
    )
    manager_model = PPO.load(manager_model_path, env=manager_env, device="cpu")
    mgr_stats = run_episodes(manager_model, manager_env, n_episodes=n_episodes, deterministic=True)
    manager_env.close()

    out: Dict[str, Any] = {
        "Experiment": os.path.basename(os.path.normpath(exp_dir)),
        "Manager Mean Reward": mgr_stats["mean_reward"],
        "Manager Std Reward": mgr_stats["std_reward"],
        "Manager Success Rate (%)": mgr_stats["success_rate"],
        "Manager Mean Effort": mgr_stats["mean_effort"],
        "Manager Std Effort": mgr_stats["std_effort"],
    }

    # ----- optional WORKER-only eval -----
    if eval_worker_too:
        worker_env = make_eval_worker_env(cfg=cfg, worker_vecnorm_path=worker_vecnorm_path)
        worker_model = PPO.load(worker_model_path, env=worker_env, device="cuda", policy=LatticeActorCriticPolicy)
        w_stats = run_episodes(worker_model, worker_env, n_episodes=n_episodes, deterministic=True)
        worker_env.close()

        out.update({
            "Worker Mean Reward": w_stats["mean_reward"],
            "Worker Std Reward": w_stats["std_reward"],
            "Worker Success Rate (%)": w_stats["success_rate"],
            "Worker Mean Effort": w_stats["mean_effort"],
            "Worker Std Effort": w_stats["std_effort"],
        })

    return out


def evaluate_many(
    exp_glob: str,
    n_episodes: int = 100,
    use_best: bool = False,
    eval_worker_too: bool = False,
):
    exp_dirs = sorted(glob.glob(exp_glob))
    if not exp_dirs:
        print(f"No experiment dirs found for glob: {exp_glob}")
        return

    rows = []
    print(f"Found {len(exp_dirs)} experiment dirs")

    for d in exp_dirs:
        try:
            row = evaluate_one_experiment(
                exp_dir=d,
                n_episodes=n_episodes,
                use_best=use_best,
                eval_worker_too=eval_worker_too,
            )
            rows.append(row)
            print(
                f"[OK] {row['Experiment']} | "
                f"MgrR={row['Manager Mean Reward']:.2f} | "
                f"MgrS={row['Manager Success Rate (%)']:.1f}%"
            )
        except Exception as e:
            print(f"[FAIL] {d}: {e}")
            raise

    df = pd.DataFrame(rows)

    # aggregated summary (manager)
    mgr_r_mean = df["Manager Mean Reward"].mean()
    mgr_r_std = df["Manager Mean Reward"].std()
    mgr_s_mean = df["Manager Success Rate (%)"].mean()
    mgr_s_std = df["Manager Success Rate (%)"].std()

    print("\n" + "=" * 90)
    print("FINAL REPORT (Manager)")
    print("=" * 90)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)
    print(df.to_string(index=False))
    print("-" * 90)
    print(f"Manager Reward:       {mgr_r_mean:.2f} ± {mgr_r_std:.2f}")
    print(f"Manager Success Rate: {mgr_s_mean:.2f}% ± {mgr_s_std:.2f}")
    print("=" * 90)

    return df


if __name__ == "__main__":
    # Example:
    #   python eval_hrl.py
    # Evaluate folders like: ./logs/exp_*/  (adjust to your structure)
    evaluate_many(
        exp_glob="./logs/*/",
        n_episodes=200,
        use_best=False,        # set True to use best/best_model.zip
        eval_worker_too=True,  # also evaluate worker alone
    )