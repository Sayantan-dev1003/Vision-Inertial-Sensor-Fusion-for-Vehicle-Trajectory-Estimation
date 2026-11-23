from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict
import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VideoMetadata:
    video_path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    timestamp_source: str

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["fps"] = round(out["fps"], 4)
        out["duration_s"] = round(out["duration_s"], 3)
        return out


class VideoLoader:
    """Inspects dashcam video metadata and builds frame timestamp tables."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

    def inspect_metadata(self) -> VideoMetadata:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

        if fps <= 0.0 or frame_count <= 0:
            raise ValueError(
                f"Invalid video metadata (fps={fps}, frame_count={frame_count}) for {self.video_path}"
            )

        return VideoMetadata(
            video_path=str(self.video_path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_s=frame_count / fps,
            timestamp_source="frame_index_over_fps",
        )

    def build_frame_table(self, meta: VideoMetadata | None = None) -> pd.DataFrame:
        meta = meta or self.inspect_metadata()
        indices = np.arange(meta.frame_count, dtype=np.int64)
        return pd.DataFrame(
            {
                "frame_idx": indices,
                "t_video_s": indices / meta.fps,
                "timestamp_source": meta.timestamp_source,
            }
        )