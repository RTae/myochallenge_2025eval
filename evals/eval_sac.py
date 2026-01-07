"""
SAC EVAL script (single-model) in the SAME “shape” as your HRL eval scripts:

- freeze_vecnormalize
- run_vecenv_episodes
- resolve_single_paths
- make_eval_env
- evaluate_one_experiment
- evaluate_folders
- CLI: --logs/--glob/--trials/--use-best

Assumed training layout (from your SAC train code):
  <exp_dir>/
    model.pkl
    best_model/best_model.zip   (optional, from EvalCallback)

Update the glob default to match your folder names.
"""

import os
import sys
import glob
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

# Fix path to allow importing from root (keep same style)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from config import Config
from env_factory import create_default_env


# =============================================================================
# Helpers
# =============================================================================

def freeze_vecnormalize(env) -> None:
    if isinstance(env, VecNormalize):
        env.training = False
        env.norm_reward = False


def run_vecenv_episodes(
    model: SAC,
    env,
    trials: int,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    VecEnv num_envs=1 evaluation for SAC (non-recurrent).
    Metrics match your other eval scripts:
      - episode reward from info["episode"]["r"] if available
      - effort from info.get("effort", 0.0)
      - success from info.get("is_success", False)
    """
    ep_rewards: List[float] = []
    efforts: List[float] = []
    success_count = 0

    obs = env.reset()

    for _ in tqdm(range(trials), desc="Evaluating"):
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, dones, infos = env.step(action)

            done = bool(dones[0])
            info = infos[0]

            if done:
                if isinstance(info.get("episode"), dict) and "r" in info["episode"]:
                    ep_rewards.append(float(info["episode"]["r"]))
                else:
                    ep_rewards.append(0.0)

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
# Paths
# =============================================================================

@dataclass(frozen=True)
class SACEvalPaths:
    model_path: str


def resolve_sac_paths(exp_dir: str, use_best: bool) -> SACEvalPaths:
    """
    If use_best:
        <exp_dir>/best_model/best_model.zip
    Else:
        <exp_dir>/model.pkl
    """
    if use_best:
        model_path = os.path.join(exp_dir, "best_model", "best_model.zip")
    else:
        model_path = os.path.join(exp_dir, "model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing required file: {model_path}")

    return SACEvalPaths(model_path=model_path)


# =============================================================================
# Env builder
# =============================================================================

def make_eval_env(cfg: Config):
    env = create_default_env(cfg, num_envs=1)
    freeze_vecnormalize(env)
    return env


# =============================================================================
# Evaluate one experiment
# =============================================================================

def evaluate_one_experiment(
    exp_dir: str,
    trials: int,
    use_best: bool = False,
    deterministic: bool = True,
) -> Dict[str, Any]:
    cfg = Config()
    cfg.logdir = exp_dir  # convenience only

    paths = resolve_sac_paths(exp_dir, use_best=use_best)

    env = make_eval_env(cfg)
    try:
        model = SAC.load(paths.model_path, env=env)
        stats = run_vecenv_episodes(model, env, trials=trials, deterministic=deterministic)
    finally:
        env.close()

    return {
        "Experiment": os.path.basename(os.path.normpath(exp_dir)),
        "Mean Reward": stats["mean_reward"],
        "Std Reward": stats["std_reward"],
        "Success Rate (%)": stats["success_rate"],
        "Mean Effort": stats["mean_effort"],
        "Std Effort": stats["std_effort"],
    }


# =============================================================================
# Evaluate many folders
# =============================================================================

def evaluate_folders(
    folders: List[str],
    trials: int = 200,
    use_best: bool = False,
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
            row = evaluate_one_experiment(
                exp_dir=exp_dir,
                trials=trials,
                use_best=use_best,
                deterministic=True,
            )
            rows.append(row)
            print(
                f"Finished {name}: "
                f"Mean Reward={row['Mean Reward']:.2f}, "
                f"Success Rate (%)={row['Success Rate (%)']:.1f}%"
            )
        except FileNotFoundError as e:
            print(f"Skipping {name}: {e}")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            raise

    if not rows:
        print("No models evaluated.")
        return None

    df = pd.DataFrame(rows)

    r_mean = df["Mean Reward"].mean()
    r_std = df["Mean Reward"].std()
    s_mean = df["Success Rate (%)"].mean()
    s_std = df["Success Rate (%)"].std()
    e_mean = df["Mean Effort"].mean()
    e_std = df["Mean Effort"].std()

    print("\n" + "=" * 90)
    print("FINAL AGGREGATED REPORT")
    print("=" * 90)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)
    print(df.to_string(index=False))
    print("-" * 90)
    print(f"Reward:       {r_mean:.2f} ± {r_std:.2f}")
    print(f"Success Rate: {s_mean:.2f}% ± {s_std:.2f}")
    print(f"Effort:       {e_mean:.4f} ± {e_std:.4f}")
    print("=" * 90)

    return df


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=str, default="./logs", help="Root folder containing experiment subfolders")
    p.add_argument("--glob", type=str, default="sac*/", help="Glob under logs root to select experiments")
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--use-best", action="store_true", help="Use best_model/best_model.zip instead of model.pkl")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    folders = sorted(glob.glob(os.path.join(args.logs, args.glob)))
    evaluate_folders(
        folders=folders,
        trials=args.trials,
        use_best=args.use_best,
    )