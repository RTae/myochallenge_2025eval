import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3.common.vec_env import VecNormalize

# --- keep your imports (same algorithm) ---
from sb3_contrib import RecurrentPPO
from lattice.ppo.policies import LatticeActorCriticPolicy

from config import Config
from env_factory import create_default_env


# =============================================================================
# Small helpers (same behavior, cleaner)
# =============================================================================

def freeze_vecnormalize(env) -> None:
    """If env is VecNormalize, freeze it for eval."""
    if isinstance(env, VecNormalize):
        env.training = False
        env.norm_reward = False


@dataclass(frozen=True)
class EvalResult:
    seed: str
    mean_reward: float
    success_rate: float
    mean_effort: float


def resolve_model_path(folder_path: str) -> Optional[str]:
    """
    Keep your layout:
      <folder>/best_model/best_model.zip
    Return None if missing.
    """
    model_path = os.path.join(folder_path, "best_model", "best_model.zip")
    return model_path if os.path.exists(model_path) else None


def load_recurrent_model(model_path: str, env) -> RecurrentPPO:
    """
    Centralize loading. You used custom_objects; keep it.
    """
    return RecurrentPPO.load(
        model_path,
        env=env,
        custom_objects={"LatticeActorCriticPolicy": LatticeActorCriticPolicy},
    )


def evaluate_single_model(model_path: str, env, trials: int = 1000) -> Tuple[float, float, float]:
    """
    Runs evaluation for a single RecurrentPPO model instance on a VecEnv num_envs=1.
    Keeps your LSTM state handling and metrics identical.
    """
    model = load_recurrent_model(model_path, env)

    obs = env.reset()

    # Recurrent policies need hidden states + episode_starts markers
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)  # num_envs=1

    all_rewards: List[float] = []
    efforts: List[float] = []
    success_count = 0

    for _ in tqdm(range(trials), desc=f"Evaluating {os.path.basename(model_path)}"):
        while True:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )

            obs, _, dones, infos = env.step(action)

            # Critical: reset LSTM on episode boundaries
            episode_starts = dones

            done = bool(dones[0])
            info = infos[0]

            if done:
                # same assumption as your original code (episode info exists)
                all_rewards.append(float(info["episode"]["r"]))
                efforts.append(float(info.get("effort", 0.0)))
                if bool(info.get("is_success", False)):
                    success_count += 1
                break

    mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    success_rate = 100.0 * float(success_count) / float(trials)
    mean_effort = float(np.mean(efforts)) if efforts else 0.0

    return mean_reward, success_rate, mean_effort


def evaluate(folders: List[str], trials: int = 1000):
    cfg = Config()
    results: List[EvalResult] = []

    print(f"Found {len(folders)} experiment folders. Beginning evaluation...\n")
    print("-" * 60)

    for folder_path in folders:
        seed_name = os.path.basename(os.path.normpath(folder_path))

        model_path = resolve_model_path(folder_path)
        if model_path is None:
            print(f"Skipping {seed_name}: best_model/best_model.zip not found.")
            continue

        env = None
        try:
            # Fresh env per run (num_envs=1 to match LSTM logic)
            env = create_default_env(cfg, num_envs=1)
            freeze_vecnormalize(env)

            mean_r, success_r, mean_e = evaluate_single_model(model_path, env, trials=trials)

            results.append(EvalResult(seed=seed_name, mean_reward=mean_r, success_rate=success_r, mean_effort=mean_e))
            print(f"Finished {seed_name}: R={mean_r:.2f}, S={success_r:.1f}%, E={mean_e:.4f}")

        except Exception as e:
            print(f"Error evaluating {seed_name}: {e}")
            # raise  # uncomment if you want full stack trace
        finally:
            if env is not None:
                env.close()

    if not results:
        print("No models evaluated.")
        return

    # Report Results (same format)
    df = pd.DataFrame([{
        "Seed": r.seed,
        "Mean Reward": r.mean_reward,
        "Success Rate": r.success_rate,
        "Mean Effort": r.mean_effort,
    } for r in results])

    avg_reward = df["Mean Reward"].mean()
    std_reward = df["Mean Reward"].std()
    avg_success = df["Success Rate"].mean()
    std_success = df["Success Rate"].std()
    avg_effort = df["Mean Effort"].mean()
    std_effort = df["Mean Effort"].std()

    print("\n" + "=" * 80)
    print("FINAL AGGREGATED REPORT")
    print("=" * 80)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(df.to_string(index=False))
    print("-" * 80)
    print(f"Average Reward:       {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Success Rate: {avg_success:.2f}% ± {std_success:.2f}")
    print(f"Average Effort:       {avg_effort:.4f} ± {std_effort:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate(
        folders=sorted(glob.glob("./logs/ppo_lattice_seed*/")),
        trials=1000,
    )