import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# ppo = 240.92, 222.10, 260.26
# ppo-hrl = 
# ppo-lattice=
# ppo-lattice-hrl=

# sac = 342.62, 368.32, 454.22
# sac-hrl = 3.37
# sac-lattice = 204.20

def load_tb_scalar(logdir, tag):
    """
    Load (step, value) pairs from TensorBoard logs.
    """
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        raise ValueError(f"Tag '{tag}' not found. Available: {ea.Tags()['scalars']}")

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values

def normalize_auc(auc, optimal_reward=1.0, max_steps=1e6):
    """Normalize AUC by optimal performance and time"""
    return auc / (optimal_reward * max_steps)

def compute_auc(steps, values):
    """
    Compute Area Under Curve using trapezoidal rule.
    """
    duration = steps[-1] - steps[0]
    return np.trapz(values, steps) / duration


if __name__ == "__main__":
    logdir = "./logs/PPO_LATTICE/ppo_lattice44/RecurrentPPO_1"
    tag = "eval/mean_reward"
    
    steps, values = load_tb_scalar(logdir, tag)
    
    # Compute AUC
    auc = compute_auc(steps, values)
    
    # Optional normalization
    # norm_auc = normalize_auc(auc, optimal_reward=1000, max_steps=steps[-1])
    
    print(f"AUC for tag '{tag}': {auc:.2f}")
    # print(f"Normalized AUC: {norm_auc:.4f}")