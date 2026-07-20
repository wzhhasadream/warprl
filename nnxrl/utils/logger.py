import atexit
import csv
import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union
import time
import numpy as np

_current_run = None
run_dir = None


def _tile_videos(videos: np.ndarray) -> np.ndarray:
    """Arrange a batch of videos into a single row-major grid video."""
    batch_size, num_frames, height, width, channels = videos.shape
    num_cols = int(np.ceil(np.sqrt(batch_size)))
    num_rows = int(np.ceil(batch_size / num_cols))
    num_slots = num_rows * num_cols

    if num_slots != batch_size:
        padding = np.zeros(
            (num_slots - batch_size, num_frames, height, width, channels),
            dtype=videos.dtype,
        )
        videos = np.concatenate((videos, padding), axis=0)

    return videos.reshape(
        num_rows, num_cols, num_frames, height, width, channels
    ).transpose(2, 0, 3, 1, 4, 5).reshape(
        num_frames, num_rows * height, num_cols * width, channels
    )


def _to_python_scalar(value: Any, round_floats: bool = True) -> Any:
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 4) if round_floats else value
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, float):
            return round(value, 4) if round_floats else value
        return value
    if hasattr(value, "shape"):
        arr = np.asarray(value)
        if arr.ndim == 0:
            scalar = arr.item()
            if isinstance(scalar, float):
                return round(scalar, 4) if round_floats else scalar
            return scalar
        return arr.tolist()
    return value


def _normalize_config(value: Any, round_floats: bool = False) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_config(value[key], round_floats=round_floats)
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_config(item, round_floats=round_floats) for item in value]
    return _to_python_scalar(value, round_floats=round_floats)


def _config_hash(config: Dict, exclude_keys: tuple[str, ...] = ("seed",)) -> str:
    config_for_hash = {
        key: value for key, value in config.items() if str(key) not in exclude_keys
    }
    normalized = _normalize_config(config_for_hash)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


class Run:
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        dir: str = "Results",
        flush_every: int = 100,
        flush_interval: float = 10.0,
    ):
        self.project = project
        self.name = name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.step_count = 0
        self.flush_every = flush_every
        self.flush_interval = flush_interval
        self._finished = False
        self.start_time = time.time()

        cfg = config or {}
        self.seed = cfg.get("seed")
        self.config_hash = _config_hash(cfg)

        os.makedirs(dir, exist_ok=True)
        run_parent_dir = os.path.join(
            dir, project, self.name, f"cfg_{self.config_hash}")

        if self.seed is not None:
            run_parent_dir = os.path.join(run_parent_dir, f"seed_{self.seed}")

        self.run_dir = run_parent_dir
        os.makedirs(self.run_dir, exist_ok=True)
        global run_dir
        run_dir = self.run_dir

        self.metrics_jsonl_file = os.path.join(self.run_dir, "metrics.jsonl")
        self.metrics_file = os.path.join(self.run_dir, "metrics.csv")
        self.config_file = os.path.join(self.run_dir, "config.json")

        self.config = cfg
        if self.config:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(_normalize_config(self.config), f, indent=2)

        self._metrics_fp = open(
            self.metrics_jsonl_file,
            "w",
            encoding="utf-8",
            buffering=1,
        )

    def log(
        self,
        data: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        if self._finished:
            raise RuntimeError("Run has already been finished.")
        wall_time = time.time() - self.start_time
        data["wall_time"] = wall_time
        if step is None:
            step = self.step_count

        row = {"step": step}
        for key, value in data.items():
            row[key] = _to_python_scalar(value)

        self._metrics_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._metrics_fp.flush()

        if commit:
            self.step_count = max(self.step_count, step + 1)

    def flush(self):
        if self._finished:
            return

        self._metrics_fp.flush()

    def _convert_jsonl_to_csv(self):
        rows_by_step: dict[int, dict[str, Any]] = {}
        all_keys: set[str] = set()

        with open(self.metrics_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                step = row["step"]
                merged_row = rows_by_step.setdefault(step, {"step": step})
                for key, value in row.items():
                    if key == "step":
                        continue
                    merged_row[key] = value
                    all_keys.add(key)

        columns = ["step"] + sorted(all_keys)

        with open(self.metrics_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for step in sorted(rows_by_step):
                row = rows_by_step[step]
                writer.writerow({col: row.get(col, "") for col in columns})

    def finish(self):
        if self._finished:
            return

        self.flush()
        self._metrics_fp.close()
        self._convert_jsonl_to_csv()
        self._finished = True
        global _current_run
        if _current_run is self:
            _current_run = None
        print(f"Run finished. Data saved to: {self.run_dir}")


    def video(self, videos: np.ndarray, step: int, fps: int = 30, fmt: str = 'mp4'):
        import imageio.v2 as imageio
        fmt = fmt.removeprefix(".").lower()
        self.video_path = os.path.join(self.run_dir, 'video', f'step_{step}')
        os.makedirs(self.video_path, exist_ok=True)

        def save_frames(frames: np.ndarray, name: str) -> None:
            path = os.path.join(self.video_path, f"{name}.{fmt}")
            if fmt == "gif":
                duration = int(round(1000 / fps))
                imageio.mimsave(path, frames, duration=duration)
            elif fmt == "mp4":
                imageio.mimsave(path, frames, fps=fps)
            else:
                raise ValueError(f"Unsupported video format: {fmt}")

        if videos.ndim == 5:
            # [B, T, H, W, C] -> [T, rows * H, cols * W, C].
            save_frames(_tile_videos(videos), "video")
        elif videos.ndim == 4:
            # [T, H, W, C] -> save one video.
            save_frames(videos, "video")
        else:
            raise ValueError(f"Expected video array with 4 or 5 dims, got shape {videos.shape}")

    def save_agent(self, agent, step: int):
        model_path = os.path.join(self.run_dir, "model")
        os.makedirs(model_path, exist_ok=True)
        agent.save(os.path.join(model_path, f"{step}_step"))

    def save_onnx(self, agent, step: int):
        onnx_path = os.path.join(self.run_dir, f"onnx")
        os.makedirs(onnx_path, exist_ok=True)
        agent.save_onnx(os.path.join(onnx_path, f"{step}_step"))


def init(
    project: str,
    name: Optional[str] = None,
    config: Optional[Dict] = None,
    dir: str = "Results",
    flush_every: int = 100,
    flush_interval: float = 10.0,
) -> Run:
    """
    Create a new run and return it.
    run_path = dir/project/name
    """

    global _current_run
    _current_run = Run(
        project=project,
        name=name,
        config=config,
        dir=dir,
        flush_every=flush_every,
        flush_interval=flush_interval,
    )
    return _current_run


def log(data: Dict[str, Union[float, int]], step: Optional[int] = None, commit: bool = True):
    if _current_run is None:
        raise RuntimeError("No active run. Call init() first.")
    _current_run.log(data, step, commit)


def video(videos: np.ndarray, step: int, fps: int = 30, fmt: str = "mp4"):
    if _current_run is None:
        raise RuntimeError("No active run. Call init() first.")
    _current_run.video(videos, step, fps, fmt)


def save_agent(agent, step: int):
    if _current_run is None:
        raise RuntimeError("No active run. Call init() first.")
    _current_run.save_agent(agent, step)


def save_onnx(agent, step: int):
    if _current_run is None:
        raise RuntimeError("No active run. Call init() first.")
    _current_run.save_onnx(agent, step)

def finish():
    if _current_run is not None:
        _current_run.finish()


atexit.register(finish)
