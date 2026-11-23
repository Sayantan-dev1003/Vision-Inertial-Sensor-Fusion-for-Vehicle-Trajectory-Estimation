from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import pandas as pd


@dataclass(frozen=True)
class VisualMeasurementPacket:
    frame_idx: int
    t_video_s: float
    t_sync_s: float
    dt_s: float
    delta_yaw_rad: float
    yaw_rate_rad_s: float
    delta_forward: float
    forward_speed: float
    inlier_ratio: float
    inlier_count: int
    median_flow_px: float
    is_stationary: bool
    is_valid: bool
    is_metric: bool
    scale_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VisualMeasurementBuilder:
    """Converts per-step visual odometry packets into structured DataFrames for EKF & comparison."""

    @staticmethod
    def to_dataframe(packets: List[VisualMeasurementPacket]) -> pd.DataFrame:
        if not packets:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in packets])