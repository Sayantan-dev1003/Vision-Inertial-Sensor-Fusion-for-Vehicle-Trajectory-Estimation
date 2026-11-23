from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence
import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: tuple[float, ...]
    is_calibrated: bool

    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def scaled(self, scale: float) -> "CameraIntrinsics":
        return CameraIntrinsics(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
            dist_coeffs=self.dist_coeffs,
            is_calibrated=self.is_calibrated,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CameraExtrinsics:
    camera_height_m: float
    pitch_rad: float
    yaw_offset_rad: float


class CameraCalibrator:
    """Manages camera intrinsics, image resizing/undistortion, and mounting extrinsics."""

    def __init__(self, camera_cfg: Dict[str, Any]) -> None:
        self.cfg = camera_cfg
        self.calibrated = bool(camera_cfg.get("calibrated", False))
        width = int(camera_cfg.get("image_width", 1920))
        height = int(camera_cfg.get("image_height", 1080))

        intr_cfg = camera_cfg.get("intrinsics", {})
        if self.calibrated:
            fx = float(intr_cfg["fx"])
            fy = float(intr_cfg["fy"])
            cx = float(intr_cfg["cx"])
            cy = float(intr_cfg["cy"])
            dist = tuple(float(d) for d in intr_cfg.get("dist_coeffs", [0.0] * 5))
        else:
            # Nominal pinhole approximation when lens calibration is unavailable
            fx = 0.85 * float(width)
            fy = 0.85 * float(width)
            cx = 0.5 * float(width)
            cy = 0.5 * float(height)
            dist = (0.0, 0.0, 0.0, 0.0, 0.0)

        self.base_intrinsics = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            dist_coeffs=dist,
            is_calibrated=self.calibrated,
        )

        ext_cfg = camera_cfg.get("extrinsics", {})
        self.extrinsics = CameraExtrinsics(
            camera_height_m=float(ext_cfg.get("camera_height_m", 1.25)),
            pitch_rad=float(np.deg2rad(ext_cfg.get("pitch_deg", 0.0))),
            yaw_offset_rad=float(np.deg2rad(ext_cfg.get("yaw_offset_deg", 0.0))),
        )

    def get_scaled_intrinsics(self, scale: float = 1.0) -> CameraIntrinsics:
        return self.base_intrinsics.scaled(scale)

    def prepare_gray_frame(self, bgr_frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            new_w = int(round(gray.shape[1] * scale))
            new_h = int(round(gray.shape[0] * scale))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if self.calibrated and any(abs(d) > 1e-9 for d in self.base_intrinsics.dist_coeffs):
            intr = self.get_scaled_intrinsics(scale)
            gray = cv2.undistort(gray, intr.matrix(), np.array(intr.dist_coeffs, dtype=float))
        return gray

    @staticmethod
    def normalize_points(pts_px: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
        """Convert 2D pixel coordinates (N, 2) to normalized camera rays (N, 2)."""
        pts = np.asarray(pts_px, dtype=float).reshape(-1, 2)
        x_norm = (pts[:, 0] - intrinsics.cx) / intrinsics.fx
        y_norm = (pts[:, 1] - intrinsics.cy) / intrinsics.fy
        return np.column_stack([x_norm, y_norm])