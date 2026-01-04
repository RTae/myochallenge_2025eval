import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Import your custom modules
from config import Config
from env_factory import create_default_env

def evaluate_single_model(model_path, env, trials=1000):
    """
    Runs evaluation for a single model instance.
    """
    # Load model
    model = PPO.load(model_path, env=env)
    obs = env.reset()
    
    all_rewards = []
    efforts = []
    success_count = 0
        
    for _ in tqdm(range(trials), desc=f"Evaluating {model_path}"):    
        while True:
            action, _ = model.predict(obs, deterministic=True)
            
            obs, _, dones, infos = env.step(action)
            
            done = dones[0]
            info = infos[0]
            
            if done:
                all_rewards.append(info["episode"]["r"])
                efforts.append(info.get('effort', 0.0)) # Safety get in case effort is missing
                if info["is_success"]:
                    success_count += 1
                    
                break
        
    mean_reward = np.mean(all_rewards) if all_rewards else 0.0
    success_rate = (success_count / trials) * 100
    efforts_mean = np.mean(efforts) if efforts else 0.0
    return mean_reward, success_rate, efforts_mean

def evaluate(folders, trials=1000):
    
    # 1. Setup the Standardized Environment
    cfg = Config()
    results = []

    print(f"Found {len(folders)} experiment folders. Beginning evaluation...\n")
    print("-" * 60)

    for folder_path in folders:
        # --- FIX IS HERE ---
        # os.path.normpath removes the trailing slash (e.g. "seed42/" -> "seed42")
        # so basename can correctly grab the folder name.
        seed_name = os.path.basename(os.path.normpath(folder_path))
        
        # Construct Path
        model_path = os.path.join(folder_path, "best_model", "best_model.zip")
        
        if not os.path.exists(model_path):
            print(f"Skipping {seed_name}: {model_path} not found.")
            continue

        # Run Evaluation
        try:
            env = create_default_env(cfg, num_envs=1)
            
            if isinstance(env, VecNormalize):
                env.training = False
                env.norm_reward = False
                
            mean_r, success_r, efforts_mean = evaluate_single_model(model_path, env, trials=trials)
            
            results.append({
                "Seed": seed_name,
                "Mean Reward": mean_r,
                "Success Rate": success_r,
                "Mean Effort": efforts_mean
            })
            
            print(f"Finished {seed_name}: R={mean_r:.2f}, S={success_r:.1f}%, E={efforts_mean:.4f}")
            
            env.close()
            
        except Exception as e:
            # Better error printing so we see WHICH seed failed
            print(f"Error evaluating {seed_name}: {e}")
            raise e

    # 3. Report Results
    if not results:
        print("No models evaluated.")
        return

    df = pd.DataFrame(results)
    
    # Calculate stats
    avg_reward = df["Mean Reward"].mean()
    std_reward = df["Mean Reward"].std()
    avg_success = df["Success Rate"].mean()
    std_success = df["Success Rate"].std()
    avg_effort = df["Mean Effort"].mean()
    std_effort = df["Mean Effort"].std()

    print("\n" + "="*80)
    print("FINAL AGGREGATED REPORT")
    print("="*80)
    # Ensure pandas prints all columns and doesn't truncate
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(df.to_string(index=False))
    print("-" * 80)
    print(f"Average Reward:       {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Success Rate: {avg_success:.2f}% ± {std_success:.2f}")
    print(f"Average Effort:       {avg_effort:.4f} ± {std_effort:.4f}")
    print("="*80)

if __name__ == "__main__":
    
    evaluate(
        folders=sorted(glob.glob("./logs/ppo_seed*/")),
        trials=1000 # Change back to 10000 when ready
    )