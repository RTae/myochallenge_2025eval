import os
import sys
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from config import Config
from env_factory import build_manager_vec, build_worker_vec
from hrl.worker_env import TableTennisWorker
from lattice.ppo.policies import LatticeActorCriticPolicy
from utils import resume_vecnormalize_on_training_env


# =============================================================================
# Small helpers
# =============================================================================

def freeze_vecnormalize(env) -> None:
    """If env is VecNormalize, freeze it for eval."""
    if isinstance(env, VecNormalize):
        env.training = False
        env.norm_reward = False


def load_worker_model(model_path: str, device: str = "cuda") -> PPO:
    return PPO.load(model_path, device=device, policy=LatticeActorCriticPolicy)


def load_worker_vecnormalize(vecnorm_path: str, base_env: TableTennisWorker) -> VecNormalize:
    """
    Loads VecNormalize stats onto a DummyVecEnv wrapping the base_env.
    Used when manager needs a worker env instance with saved normalization.
    """
    env = DummyVecEnv([lambda: base_env])
    vecnorm = VecNormalize.load(vecnorm_path, env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


# =============================================================================
# Episode runner (VecEnv num_envs=1)
# =============================================================================

def run_vecenv_episodes(
    model: PPO,
    env,
    trials: int,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Evaluate `trials` episodes on a VecEnv with num_envs=1.

    Collects:
      - episode reward (from info["episode"]["r"] if exists)
      - effort (info.get("effort", 0.0))
      - success (bool(info.get("is_success", False)))
    """
    ep_rewards: List[float] = []
    efforts: List[float] = []
    success_count = 0

    obs = env.reset()

    for _ in tqdm(range(trials), desc="Evaluating"):
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)

            done = bool(dones[0])
            info = infos[0]

            if done:
                # reward
                if isinstance(info.get("episode"), dict) and "r" in info["episode"]:
                    ep_rewards.append(float(info["episode"]["r"]))
                else:
                    # fallback (avoid crashing)
                    ep_rewards.append(0.0)

                # effort + success
                efforts.append(float(info.get("effort", 0.0)))
                if bool(info.get("is_success", False)):
                    success_count += 1
                break

    if not ep_rewards:
        ep_rewards = [0.0]
    if not efforts:
        efforts = [0.0]

    return {
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "success_rate": 100.0 * float(success_count) / float(trials),
        "mean_effort": float(np.mean(efforts)),
        "std_effort": float(np.std(efforts)),
    }


# =============================================================================
# HRL Eval Config + Path resolver
# =============================================================================

@dataclass(frozen=True)
class HRLEvalPaths:
    worker_model: str
    worker_vecnorm: str
    manager_model: str


def resolve_hrl_paths(exp_dir: str, use_best: bool) -> HRLEvalPaths:
    worker_dir = os.path.join(exp_dir, "worker")
    manager_dir = os.path.join(exp_dir, "manager")

    if use_best:
        worker_model = os.path.join(worker_dir, "best", "best_model.zip")
        manager_model = os.path.join(manager_dir, "best", "best_model.zip")
    else:
        worker_model = os.path.join(worker_dir, "worker_model.pkl")
        manager_model = os.path.join(manager_dir, "manager_model.pkl")

    worker_vecnorm = os.path.join(worker_dir, "vecnormalize.pkl")

    for p in [worker_model, manager_model, worker_vecnorm]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    return HRLEvalPaths(
        worker_model=worker_model,
        worker_vecnorm=worker_vecnorm,
        manager_model=manager_model,
    )


# =============================================================================
# Env builders (eval-frozen)
# =============================================================================

def make_eval_worker_env(cfg: Config, worker_vecnorm_path: Optional[str]):

    env = build_worker_vec(cfg=cfg, num_envs=1)

    if worker_vecnorm_path and os.path.exists(worker_vecnorm_path):
        env = resume_vecnormalize_on_training_env(
            env,
            worker_vecnorm_path,
            training=False,
            norm_reward=False,
        )

    freeze_vecnormalize(env)
    return env


def make_eval_manager_env(
    cfg: Config,
    worker_model_path: str,
    worker_vecnorm_path: str,
    decision_interval: int,
    max_episode_steps: int,
):
    def worker_env_loader(vecnorm_path: str):
        # manager expects a callable that returns env with loaded vecnorm
        return load_worker_vecnormalize(vecnorm_path, TableTennisWorker(cfg))

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

    freeze_vecnormalize(env)
    return env


# =============================================================================
# Evaluate one experiment folder (HRL)
# =============================================================================

def evaluate_one_hrl_experiment(
    exp_dir: str,
    trials: int,
    use_best: bool = False,
    eval_worker_too: bool = False,
    deterministic: bool = True,
) -> Dict[str, Any]:

    cfg = Config()
    cfg.logdir = exp_dir  # optional convenience

    paths = resolve_hrl_paths(exp_dir, use_best=use_best)

    # ---- manager eval ----
    manager_env = make_eval_manager_env(
        cfg=cfg,
        worker_model_path=paths.worker_model,
        worker_vecnorm_path=paths.worker_vecnorm,
        decision_interval=5,
        max_episode_steps=cfg.episode_len,
    )
    manager_model = PPO.load(paths.manager_model, env=manager_env, device="cpu")
    mgr = run_vecenv_episodes(manager_model, manager_env, trials=trials, deterministic=deterministic)
    manager_env.close()

    row: Dict[str, Any] = {
        "Experiment": os.path.basename(os.path.normpath(exp_dir)),
        "High Level Policy Mean Reward": mgr["mean_reward"],
        "High Level Policy Std Reward": mgr["std_reward"],
        "High Level Policy Success Rate (%)": mgr["success_rate"],
        "High Level Policy Mean Effort": mgr["mean_effort"],
        "High Level Policy Std Effort": mgr["std_effort"],
    }

    # ---- optional worker-only eval ----
    if eval_worker_too:
        worker_env = make_eval_worker_env(cfg=cfg, worker_vecnorm_path=paths.worker_vecnorm)
        worker_model = PPO.load(
            paths.worker_model,
            env=worker_env,
            device="cuda",
            policy=LatticeActorCriticPolicy,
        )
        w = run_vecenv_episodes(worker_model, worker_env, trials=trials, deterministic=deterministic)
        worker_env.close()

        row.update({
            "Low Level Policy Mean Reward": w["mean_reward"],
            "Low Level Policy Std Reward": w["std_reward"],
            "Low Level Policy Success Rate (%)": w["success_rate"],
            "Low Level Policy Mean Effort": w["mean_effort"],
            "Low Level Policy Std Effort": w["std_effort"],
        })

    return row


# =============================================================================
# Evaluate many folders (HRL) - styled like your single-model script
# =============================================================================

def evaluate(
    folders: List[str],
    trials: int = 200,
    use_best: bool = False,
    eval_worker_too: bool = False,
) -> Optional[pd.DataFrame]:

    if not folders:
        print("No experiment folders found.")
        return None

    print(f"Found {len(folders)} experiment folders. Beginning evaluation...\n")
    print("-" * 80)

    rows: List[Dict[str, Any]] = []

    for exp_dir in folders:
        name = os.path.basename(os.path.normpath(exp_dir))
        try:
            row = evaluate_one_hrl_experiment(
                exp_dir=exp_dir,
                trials=trials,
                use_best=use_best,
                eval_worker_too=eval_worker_too,
                deterministic=True,
            )
            rows.append(row)
            print(
                f"Finished {name}: "
                f"HighLevelR={row['High Level Policy Mean Reward']:.2f}, "
                f"HighLevelS={row['High Level Policy Success Rate (%)']:.1f}%"
            )
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            raise

    df = pd.DataFrame(rows)

    # ---- aggregated summary ----
    mgr_r_mean = df["High Level Policy Mean Reward"].mean()
    mgr_r_std = df["High Level Policy Mean Reward"].std()
    mgr_s_mean = df["High Level Policy Success Rate (%)"].mean()
    mgr_s_std = df["High Level Policy Success Rate (%)"].std()

    print("\n" + "=" * 90)
    print("FINAL AGGREGATED REPORT (High Level Policy)")
    print("=" * 90)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)
    print(df.to_string(index=False))
    print("-" * 90)
    print(f"High Level Policy Reward:       {mgr_r_mean:.2f} ± {mgr_r_std:.2f}")
    print(f"High Level Policy Success Rate: {mgr_s_mean:.2f}% ± {mgr_s_std:.2f}")
    print("=" * 90)

    # ---- if worker columns exist, print worker summary too ----
    if eval_worker_too and "Low Level Policy Mean Reward" in df.columns:
        w_r_mean = df["Low Level Policy Mean Reward"].mean()
        w_r_std = df["Low Level Policy Mean Reward"].std()
        w_s_mean = df["Low Level Policy Success Rate (%)"].mean()
        w_s_std = df["Low Level Policy Success Rate (%)"].std()
        print("\n" + "=" * 90)
        print("FINAL AGGREGATED REPORT (Low Level Policy)")
        print("=" * 90)
        print(f"Low Level Policy Reward:        {w_r_mean:.2f} ± {w_r_std:.2f}")
        print(f"Low Level Policy Success Rate:  {w_s_mean:.2f}% ± {w_s_std:.2f}")
        print("=" * 90)

    return df

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=str, default="./logs", help="Root folder containing experiment subfolders")
    p.add_argument("--glob", type=str, default="*/", help="Glob under logs root to select experiments")
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--use-best", action="store_true", help="Use best/best_model.zip instead of *_model.pkl")
    p.add_argument("--eval-worker", action="store_true", help="Also evaluate worker policy alone")
    return p.parse_args()

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    args = _parse_args()
    folders = sorted(glob.glob(os.path.join(args.logs, args.glob)))
    evaluate(
        folders=folders,
        trials=args.trials,
        use_best=args.use_best,
        eval_worker_too=args.eval_worker,
    )