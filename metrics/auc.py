import numpy as np
from tensorboard.backend.event_processing import event_accumulator

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

def smooth_values(values, window=10):
    """Apply moving average smoothing"""
    if window <= 1:
        return values
    return np.convolve(values, np.ones(window)/window, mode='same')

def normalize_auc(auc, optimal_reward=1.0, max_steps=1e6):
    """Normalize AUC by optimal performance and time"""
    return auc / (optimal_reward * max_steps)

def compute_auc(steps, values):
    """
    Compute Area Under Curve using trapezoidal rule.
    """
    return np.trapz(values, steps)


if __name__ == "__main__":
    logdir = "./logs/exp"
    tag = "rollout/ep_rew_mean"
    
    steps, values = load_tb_scalar(logdir, tag)
    
    # Apply smoothing
    values_smooth = smooth_values(values, window=10)
    
    # Compute AUC
    auc = compute_auc(steps, values_smooth)
    
    # Optional normalization
    # norm_auc = normalize_auc(auc, optimal_reward=1000, max_steps=steps[-1])
    
    print(f"AUC for tag '{tag}': {auc:.2f}")
    # print(f"Normalized AUC: {norm_auc:.4f}")