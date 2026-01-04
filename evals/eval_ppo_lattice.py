import sys
import os

# Fix path to allow importing from root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3.common.vec_env import VecNormalize

# --- CHANGED: Import RecurrentPPO and Lattice Policy ---
from sb3_contrib import RecurrentPPO
from lattice.ppo.policies import LatticeActorCriticPolicy

# Import your custom modules
from config import Config
from env_factory import create_default_env

def evaluate_single_model(model_path, env, trials=1000):
    """
    Runs evaluation for a single RecurrentPPO model instance.
    """
    # We pass custom_objects just in case, though usually importing the class is enough
    model = RecurrentPPO.load(model_path, env=env, custom_objects={"LatticeActorCriticPolicy": LatticeActorCriticPolicy})
    
    obs = env.reset()
    
    # Recurrent policies need hidden states (lstm_states) and episode_starts markers
    lstm_states = None
    num_envs = 1
    episode_starts = np.ones((num_envs,), dtype=bool)
    
    all_rewards = []
    efforts = []
    success_count = 0
        
    for _ in tqdm(range(trials), desc=f"Evaluating {os.path.basename(model_path)}"):    
        while True:
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )
            
            obs, _, dones, infos = env.step(action)
            
            # Update episode_starts for the next step (critical for LSTM to reset on done)
            episode_starts = dones
            
            done = dones[0]
            info = infos[0]
            
            if done:
                all_rewards.append(info["episode"]["r"])
                efforts.append(info.get('effort', 0.0))
                if info.get("is_success", False):
                    success_count += 1
                    
                break
        
    mean_reward = np.mean(all_rewards) if all_rewards else 0.0
    success_rate = (success_count / trials) * 100
    efforts_mean = np.mean(efforts) if efforts else 0.0
    
    return mean_reward, success_rate, efforts_mean

def evaluate(folders, trials=1000):
    
    cfg = Config()
    results = []

    print(f"Found {len(folders)} experiment folders. Beginning evaluation...\n")
    print("-" * 60)

    for folder_path in folders:
        seed_name = os.path.basename(os.path.normpath(folder_path))
        
        # Construct Path
        model_path = os.path.join(folder_path, "best_model", "best_model.zip")
        
        if not os.path.exists(model_path):
            print(f"Skipping {seed_name}: {model_path} not found.")
            continue

        # Run Evaluation
        try:
            # Create fresh env (num_envs=1 for simplicity in logic)
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
            print(f"Error evaluating {seed_name}: {e}")
            # raise e # Uncomment if you want to see full stack trace

    # Report Results
    if not results:
        print("No models evaluated.")
        return

    df = pd.DataFrame(results)
    
    avg_reward = df["Mean Reward"].mean()
    std_reward = df["Mean Reward"].std()
    avg_success = df["Success Rate"].mean()
    std_success = df["Success Rate"].std()
    avg_effort = df["Mean Effort"].mean()
    std_effort = df["Mean Effort"].std()

    print("\n" + "="*80)
    print("FINAL AGGREGATED REPORT")
    print("="*80)
    
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
        folders=sorted(glob.glob("./logs/ppo_lattice_seed*/")), 
        trials=1000 
    )