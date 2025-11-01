import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from myosuite.utils import gym
from typing import Callable


# =====================================================
#  EVOLUTION STRATEGIES HELPERS
# =====================================================
def tree_add(a, b):
    """Elementwise add two PyTree structures."""
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)

def tree_scale(a, s):
    """Scale all leaves of a PyTree by scalar s."""
    return jax.tree_util.tree_map(lambda x: x * s, a)

def sample_noise_like(key, tree):
    """
    Sample Gaussian noise with the same structure and shape as a given PyTree.
    Used for Evolution Strategies parameter perturbation.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = random.split(key, len(leaves))
    noisy = [random.normal(k, shape=x.shape) for k, x in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(treedef, noisy)

# =====================================================
#  SKILL SELECTION + ONE-HOT ENCODING
# =====================================================
def one_hot(indices, n_classes):
    """
    One-hot encode a single scalar or an array of indices.

    Args:
        indices (int or np.ndarray): scalar or 1D array of skill IDs.
        n_classes (int): total number of skills.

    Returns:
        np.ndarray: one-hot encoded vector(s) of shape (n_classes,) or (batch, n_classes)
    """
    indices = np.atleast_1d(indices).astype(int)
    one_hot = np.zeros((indices.shape[0], n_classes), dtype=np.float32)
    one_hot[np.arange(indices.shape[0]), indices] = 1.0
    if one_hot.shape[0] == 1:
        return one_hot[0]
    return one_hot

def select_skill(manager_params, manager_net, obs, nskills):
    """
    Select skills (single or batch) using the manager network.

    Args:
        manager_params: parameters of the Manager network
        manager_net: Flax model (Manager)
        obs: numpy array of observations (shape [obs_dim] or [batch, obs_dim])
        nskills: total number of skills

    Returns:
        np.ndarray or int: skill indices
    """
    obs_jax = jnp.asarray(obs)
    if obs_jax.ndim == 1:
        logits = manager_net.apply(manager_params, jnp.expand_dims(obs_jax, 0))
        return int(jnp.argmax(logits, axis=-1)[0])
    else:
        logits = manager_net.apply(manager_params, obs_jax)
        return np.asarray(jnp.argmax(logits, axis=-1))

# =====================================================
#  CHECKPOINT / PARAMETER UTILITIES
# =====================================================
def flatten_params(params):
    """
    Flatten a PyTree of parameters into a dict of arrays for saving.
    Returns (flat_dict, treedef).
    """
    leaves, treedef = jax.tree_util.tree_flatten(params)
    flat = {f"arr_{i}": np.asarray(x) for i, x in enumerate(leaves)}
    return flat, treedef

def save_tree(params, path_prefix: str):
    """
    Save parameters and tree structure to disk (.npz + _treedef.txt).
    """
    flat, treedef = flatten_params(params)
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    np.savez(path_prefix + ".npz", **flat)
    with open(path_prefix + "_treedef.txt", "w") as f:
        f.write(str(treedef))

def split_keys(key, n):
    """Convenience wrapper around jax.random.split()."""
    return random.split(key, n)


# =====================================================
#  Minimal Video Helper (gym.wrappers.RecordVideo)
# =====================================================
def make_video_env(env_id: str, seed: int, video_dir: str, episode_trigger: Callable[[int], bool]):
    """
    Creates a single env wrapped with RecordVideo. SB3 uses monitor for stats; we add it here too.
    """
    os.makedirs(video_dir, exist_ok=True)

    def _make():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(env, video_dir=video_dir, episode_trigger=episode_trigger)
        env.reset(seed=seed)
        return env

    return _make()