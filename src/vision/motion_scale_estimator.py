from dataclasses import dataclass
import numpy as np

from .camera_calibration import CameraExtrinsics, CameraIntrinsics
from .optical_flow import OpticalFlowResult
from .visual_odometry import VisualOdometryEstimate


@dataclass(frozen=True)
class ScaleEstimate:
    delta_forward_distance: float   # Metres if metric_calibrated, else relative units
    forward_speed: float            # m/s if metric_calibrated, else relative units/s
    scale_factor: float
    scale_status: str               # "relative_uncalibrated" | "metric_planar_scaled"
    is_metric: bool


class MotionScaleEstimator:
    """Converts scale-ambiguous monocular motion into either explicit relative units or planar-calibrated metres."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        scale_mode: str = "relative_unit",
    ) -> None:
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.scale_mode = str(scale_mode)

    def estimate(
        self,
        vo_est: VisualOdometryEstimate,
        flow_res: OpticalFlowResult,
        dt_s: float,
        img_height: int,
    ) -> ScaleEstimate:
        if dt_s <= 1e-4 or not vo_est.is_valid or vo_est.is_stationary:
            status = (
                "metric_planar_scaled"
                if (self.scale_mode == "planar_ground" and self.intrinsics.is_calibrated)
                else "relative_uncalibrated"
            )
            return ScaleEstimate(
                delta_forward_distance=0.0,
                forward_speed=0.0,
                scale_factor=1.0,
                scale_status=status,
                is_metric=(status == "metric_planar_scaled"),
            )

        if self.scale_mode == "planar_ground" and self.intrinsics.is_calibrated:
            dist_m = self._estimate_planar_ground_displacement(flow_res, img_height)
            return ScaleEstimate(
                delta_forward_distance=dist_m,
                forward_speed=dist_m / dt_s,
                scale_factor=dist_m / max(1e-4, vo_est.forward_unit_step),
                scale_status="metric_planar_scaled",
                is_metric=True,
            )

        # Default transparent Track B baseline: pure relative visual motion (no GPS/IMU leakage)
        rel_step = float(vo_est.forward_unit_step)
        return ScaleEstimate(
            delta_forward_distance=rel_step,
            forward_speed=rel_step / dt_s,
            scale_factor=1.0,
            scale_status="relative_uncalibrated",
            is_metric=False,
        )

    def _estimate_planar_ground_displacement(
        self,
        flow_res: OpticalFlowResult,
        img_height: int,
    ) -> float:
        """Estimate forward displacement from lower-road-plane vertical optical flow."""
        if len(flow_res.pts_prev) == 0:
            return 0.0

        # Select points on the lower road patch (y > 0.62 * H)
        road_mask = flow_res.pts_prev[:, 1] > (0.62 * img_height)
        if np.sum(road_mask) < 10:
            return 0.0

        y0 = flow_res.pts_prev[road_mask, 1]
        dy = flow_res.flow_vectors[road_mask, 1]
        # Ground plane projection: Z = h * fy / (y - cy), dZ ≈ Z^2 * dy / (h * fy)
        y_rel = np.maximum(15.0, y0 - self.intrinsics.cy)
        z_depth = (self.extrinsics.camera_height_m * self.intrinsics.fy) / y_rel
        dz = (z_depth**2) * np.maximum(0.0, dy) / (self.extrinsics.camera_height_m * self.intrinsics.fy)
        return float(np.clip(np.median(dz), 0.0, 4.0))