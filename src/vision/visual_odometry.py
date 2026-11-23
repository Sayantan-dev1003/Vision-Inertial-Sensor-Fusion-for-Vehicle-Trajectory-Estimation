from dataclasses import dataclass
import cv2
import numpy as np

from .camera_calibration import CameraIntrinsics
from .optical_flow import OpticalFlowResult


@dataclass(frozen=True)
class VisualOdometryEstimate:
    delta_yaw_rad: float          # CCW-positive vehicle heading change over dt
    yaw_rate_rad_s: float
    forward_unit_step: float      # Scale-ambiguous forward expansion proxy (>= 0)
    inlier_ratio: float
    inlier_count: int
    median_flow_px: float
    is_stationary: bool
    is_valid: bool


class MonocularVisualOdometry:
    """Estimates ego-motion (CCW-positive yaw rate & radial forward motion) from optical flow."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        ransac_prob: float = 0.999,
        ransac_thresh_px: float = 1.5,
        min_inliers: int = 25,
        stationary_flow_px: float = 1.45,
        max_yaw_rate_rad_s: float = 0.85,
    ) -> None:
        self.intrinsics = intrinsics
        self.K = intrinsics.matrix()
        self.ransac_prob = float(ransac_prob)
        self.ransac_thresh_px = float(ransac_thresh_px)
        self.min_inliers = int(min_inliers)
        self.stationary_flow_px = float(stationary_flow_px)
        self.max_yaw_rate_rad_s = float(max_yaw_rate_rad_s)

    def _compute_radial_forward_motion(
        self,
        pts_prev: np.ndarray,
        flow_vectors: np.ndarray,
        delta_yaw_rad: float,
    ) -> float:
        """
        Remove rotational horizontal shift (fx * delta_yaw) and measure radial expansion
        away from principal point (cx, cy). Immune to recoverPose sign flips.
        """
        if len(pts_prev) == 0:
            return 0.0

        dx_rot = self.intrinsics.fx * delta_yaw_rad
        dx_trans = flow_vectors[:, 0] - dx_rot
        dy_trans = flow_vectors[:, 1]

        rx = pts_prev[:, 0] - self.intrinsics.cx
        ry = pts_prev[:, 1] - self.intrinsics.cy
        r_norm = np.hypot(rx, ry)

        valid_r = r_norm > 25.0
        if np.sum(valid_r) < 10:
            return 0.0

        radial_flow = (rx[valid_r] * dx_trans[valid_r] + ry[valid_r] * dy_trans[valid_r]) / r_norm[valid_r]
        # Normalize by focal length so output is dimensionless angular expansion
        norm_expansion = float(np.median(radial_flow)) / self.intrinsics.fx
        return float(np.clip(norm_expansion * 100.0, 0.0, 2.5))

    def estimate_step(
        self,
        flow_res: OpticalFlowResult,
        dt_s: float,
    ) -> VisualOdometryEstimate:
        if dt_s <= 1e-4 or flow_res.tracked_count < self.min_inliers:
            return VisualOdometryEstimate(
                delta_yaw_rad=0.0,
                yaw_rate_rad_s=0.0,
                forward_unit_step=0.0,
                inlier_ratio=0.0,
                inlier_count=flow_res.tracked_count,
                median_flow_px=flow_res.median_flow_mag_px,
                is_stationary=flow_res.median_flow_mag_px < self.stationary_flow_px,
                is_valid=False,
            )

        if flow_res.median_flow_mag_px < self.stationary_flow_px:
            return VisualOdometryEstimate(
                delta_yaw_rad=0.0,
                yaw_rate_rad_s=0.0,
                forward_unit_step=0.0,
                inlier_ratio=1.0,
                inlier_count=flow_res.tracked_count,
                median_flow_px=flow_res.median_flow_mag_px,
                is_stationary=True,
                is_valid=True,
            )

        E, mask = cv2.findEssentialMat(
            flow_res.pts_prev,
            flow_res.pts_curr,
            self.K,
            method=cv2.RANSAC,
            prob=self.ransac_prob,
            threshold=self.ransac_thresh_px,
        )

        if E is None or mask is None or E.shape != (3, 3):
            return self._fallback_flow_motion(flow_res, dt_s)

        inlier_mask = mask.ravel() > 0
        inlier_count = int(np.sum(inlier_mask))
        inlier_ratio = float(inlier_count / max(1, flow_res.tracked_count))

        if inlier_count < self.min_inliers:
            return self._fallback_flow_motion(flow_res, dt_s)

        _, R, _, _ = cv2.recoverPose(
            E,
            flow_res.pts_prev[inlier_mask],
            flow_res.pts_curr[inlier_mask],
            self.K,
        )

        # Left turn (CCW in ENU) shifts background right (+dx) -> R[0, 2] is positive
        vehicle_delta_yaw_rad = float(np.arctan2(R[0, 2], R[2, 2]))
        yaw_rate = vehicle_delta_yaw_rad / dt_s

        if abs(yaw_rate) > self.max_yaw_rate_rad_s:
            return self._fallback_flow_motion(flow_res, dt_s)

        fwd_step = self._compute_radial_forward_motion(
            flow_res.pts_prev[inlier_mask],
            flow_res.flow_vectors[inlier_mask],
            vehicle_delta_yaw_rad,
        )

        return VisualOdometryEstimate(
            delta_yaw_rad=vehicle_delta_yaw_rad,
            yaw_rate_rad_s=yaw_rate,
            forward_unit_step=fwd_step,
            inlier_ratio=inlier_ratio,
            inlier_count=inlier_count,
            median_flow_px=flow_res.median_flow_mag_px,
            is_stationary=False,
            is_valid=True,
        )

    def _fallback_flow_motion(
        self,
        flow_res: OpticalFlowResult,
        dt_s: float,
    ) -> VisualOdometryEstimate:
        dx_med = float(np.median(flow_res.flow_vectors[:, 0]))
        delta_yaw = float(np.arctan2(dx_med, self.intrinsics.fx))
        yaw_rate = float(np.clip(delta_yaw / dt_s, -self.max_yaw_rate_rad_s, self.max_yaw_rate_rad_s))
        delta_yaw = yaw_rate * dt_s
        fwd_step = self._compute_radial_forward_motion(
            flow_res.pts_prev, flow_res.flow_vectors, delta_yaw
        )
        return VisualOdometryEstimate(
            delta_yaw_rad=delta_yaw,
            yaw_rate_rad_s=yaw_rate,
            forward_unit_step=fwd_step,
            inlier_ratio=0.40,
            inlier_count=flow_res.tracked_count,
            median_flow_px=flow_res.median_flow_mag_px,
            is_stationary=False,
            is_valid=True,
        )