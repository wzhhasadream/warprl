import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import tyro

from nnxrl.agents import RainbowSACAgent
from nnxrl.env.isaaclab import make_isaaclab_env
from nnxrl.utils import record_video


@dataclasses.dataclass
class Args:
    run_dir: str
    checkpoint_path: str | None = None
    output_path: str | None = None
    env_id: str | None = None
    seed: int | None = None
    headless: bool = True
    num_envs: int = 1
    num_episodes: int = 1
    video_length: int = 1000
    fps: int = 30
    format: str = "mp4"


def _load_config(run_dir: Path) -> SimpleNamespace:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return SimpleNamespace(**config)


def _resolve_checkpoint_path(run_dir: Path, checkpoint_path: str | None) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path)

    model_dir = run_dir / "model"
    checkpoints = sorted(
        model_dir.glob("*_ckpt"),
        key=lambda path: int(path.name.removesuffix("_ckpt")),
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found under {model_dir}. Pass --checkpoint-path explicitly."
        )
    return checkpoints[-1]


def _default_output_path(run_dir: Path, checkpoint_path: Path, fmt: str) -> Path:
    step_name = checkpoint_path.name.removesuffix("_ckpt")
    return run_dir / "video" / f"step_{step_name}" / f"isaaclab_record.{fmt}"


def _save_video(videos: np.ndarray, output_path: Path, fps: int, fmt: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = videos[0]
    suffix = output_path.suffix.removeprefix(".").lower()
    fmt = suffix or fmt.lower()

    if fmt == "gif":
        duration = int(round(1000 / fps))
        imageio.mimsave(output_path, frames, duration=duration)
    elif fmt == "mp4":
        imageio.mimsave(output_path, frames, fps=fps)
    else:
        raise ValueError(f"Unsupported video format: {fmt}")


def main() -> None:
    args = tyro.cli(Args)

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = _load_config(run_dir)
    checkpoint_path = _resolve_checkpoint_path(run_dir, args.checkpoint_path)

    cfg.env_id = args.env_id or cfg.env_id
    cfg.seed = args.seed if args.seed is not None else cfg.seed
    cfg.eval_episode = args.num_episodes
    cfg.buffer_size = max(args.num_envs, int(getattr(cfg, "n_step", 1)))
    cfg.buffer_device = "cpu"

    envs = make_isaaclab_env(
        env_name=cfg.env_id,
        num_envs=args.num_envs,
        seed=cfg.seed,
        headless=args.headless,
        render_mode="rgb_array",
    )
    agent = RainbowSACAgent(envs, cfg)
    agent.load(str(checkpoint_path))

    def policy(obs: np.ndarray) -> np.ndarray:
        return agent.get_action(obs)

    try:
        videos = record_video(
            policy,
            envs,
            env_type="isaaclab",
            num_episodes=args.num_episodes,
            video_length=args.video_length,
        )
        output_path = (
            Path(args.output_path).expanduser().resolve()
            if args.output_path is not None
            else _default_output_path(run_dir, checkpoint_path, args.format)
        )
        _save_video(videos, output_path, args.fps, args.format)
        print(f"Saved IsaacLab video to: {output_path}")
    finally:
        envs.close()


if __name__ == "__main__":
    main()
