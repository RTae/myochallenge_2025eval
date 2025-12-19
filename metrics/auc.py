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


def compute_auc(steps, values):
    """
    Compute Area Under Curve using trapezoidal rule.
    """
    return np.trapz(values, steps)


if __name__ == "__main__":
    logdir = "./logs/exp"
    tag = "rollout/ep_rew_mean"

    steps, values = load_tb_scalar(logdir, tag)
    auc = compute_auc(steps, values)
    print(f"AUC for tag '{tag}': {auc}")