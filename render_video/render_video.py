import os
import sys
import glob
import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import skvideo.io

from stable_baselines3 import PPO
from custom_env import CustomEnv

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from config import Config

def _reset_any(env, seed: Optional[int] = None):
    """
    Supports both:
      - Gymnasium raw env: obs, info = env.reset(seed=...)
      - VecEnv: obs = env.reset()
    """
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()

    # VecEnv returns obs (np.ndarray)
    if isinstance(out, tuple) and len(out) == 2:
        obs, _info = out
        return obs
    return out


def _step_any(env, action):
    """
    Supports both:
      - Gymnasium raw env: obs, r, terminated, truncated, info
      - VecEnv: obs, r, dones, infos
    Returns: obs, done(bool), info(dict)
    """
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, _r, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, done, info
    elif isinstance(out, tuple) and len(out) == 4:
        obs, _r, dones, infos = out
        done = bool(dones[0])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        return obs, done, info
    else:
        raise RuntimeError(f"Unknown env.step() return format: {type(out)} / len={len(out) if hasattr(out,'__len__') else 'NA'}")


def render_offscreen_frame(env, width: int, height: int, camera_id: int):
    """
    Matches your VideoCallback usage.
    """
    return env.sim.renderer.render_offscreen(width=width, height=height, camera_id=camera_id)


def write_video_mp4(path: str, frames: List[np.ndarray], fps: int = 30):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    skvideo.io.vwrite(
        path,
        np.asarray(frames),
        outputdict={
            "-pix_fmt": "yuv420p",
            "-r": str(fps),
        },
    )


# =============================================================================
# Paths
# =============================================================================

@dataclass(frozen=True)
class VideoPaths:
    model_path: str
    out_path: str


def resolve_video_paths(exp_dir: str, out_dir: Optional[str], out_name: str) -> VideoPaths:
    model_path = os.path.join(exp_dir, "best_model", "best_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing required file: {model_path}")

    if out_dir is None:
        out_dir = os.path.join(exp_dir, "videos")

    out_path = os.path.join(out_dir, out_name)
    return VideoPaths(model_path=model_path, out_path=out_path)


# =============================================================================
# Core rendering
# =============================================================================

def record_video_one_experiment(
    exp_dir: str,
    frames: int,
    deterministic: bool = True,
    fps: int = 30,
    seed_offset: int = 123,
    out_dir: Optional[str] = None,
    out_name: str = "eval.mp4",
):
    cfg = Config()
    cfg.logdir = exp_dir  # convenience (like eval scripts)

    paths = resolve_video_paths(exp_dir, out_dir=out_dir, out_name=out_name)

    # Create env (num_envs=1 to keep it simple and consistent with eval scripts)
    env = CustomEnv(cfg)

    try:
        model = PPO.load(paths.model_path, env=env)

        # Reset
        obs = _reset_any(env, seed=cfg.seed + seed_offset)

        video_frames: List[np.ndarray] = []

        for _ in range(frames):
            frame = render_offscreen_frame(env, width=cfg.video_w, height=cfg.video_h, camera_id=-1)
            video_frames.append(frame)

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, done, _info = _step_any(env, action)

            if done:
                obs = _reset_any(env, seed=cfg.seed + seed_offset)

        write_video_mp4(paths.out_path, video_frames, fps=fps)
        print(f"✅ Saved video: {paths.out_path}")

    finally:
        env.close()


# =============================================================================
# CLI utilities (same “logs + glob” pattern)
# =============================================================================

def list_experiments(logs_root: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(logs_root, pattern)))


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=str, default="./logs", help="Root folder containing experiment subfolders")
    p.add_argument("--glob", type=str, default="ppo*/", help="Glob under logs root to select experiments")
    p.add_argument("--exp", type=str, default="", help="Exact experiment folder name to render (overrides --idx)")
    p.add_argument("--idx", type=int, default=0, help="Index into matched folders if --exp not provided")
    p.add_argument("--frames", type=int, default=600, help="Number of frames to record")
    p.add_argument("--fps", type=int, default=30, help="Output video FPS")
    p.add_argument("--deterministic", action="store_true", help="Deterministic policy actions")
    p.add_argument("--out-dir", type=str, default="", help="Override output dir (default: <exp>/videos)")
    p.add_argument("--out-name", type=str, default="eval.mp4", help="Output file name (mp4)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    folders = list_experiments(args.logs, args.glob)
    if not folders:
        raise SystemExit(f"No experiments found under {args.logs} with glob {args.glob}")

    if args.exp:
        exp_dir = os.path.join(args.logs, args.exp)
        if not os.path.isdir(exp_dir):
            raise SystemExit(f"--exp not found or not a folder: {exp_dir}")
    else:
        if args.idx < 0 or args.idx >= len(folders):
            raise SystemExit(f"--idx out of range: {args.idx} (found {len(folders)} folders)")
        exp_dir = folders[args.idx]

    out_dir = args.out_dir if args.out_dir.strip() else None

    print(f"🎥 Rendering experiment: {exp_dir}")
    record_video_one_experiment(
        exp_dir=exp_dir,
        frames=args.frames,
        deterministic=bool(args.deterministic),
        fps=args.fps,
        out_dir=out_dir,
        out_name=args.out_name,
    )