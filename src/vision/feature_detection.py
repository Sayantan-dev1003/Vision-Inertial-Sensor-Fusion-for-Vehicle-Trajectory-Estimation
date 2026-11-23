import cv2
import numpy as np


class FeatureDetector:
    """Detects Shi-Tomasi background keypoints within a road/environment ROI mask."""

    def __init__(
        self,
        max_corners: int = 400,
        quality_level: float = 0.01,
        min_distance_px: int = 12,
        roi_top_ratio: float = 0.35,
        roi_bottom_ratio: float = 0.88,
    ) -> None:
        self.max_corners = int(max_corners)
        self.quality_level = float(quality_level)
        self.min_distance_px = int(min_distance_px)
        self.roi_top_ratio = float(roi_top_ratio)
        self.roi_bottom_ratio = float(roi_bottom_ratio)
        self._cached_mask: np.ndarray | None = None
        self._cached_shape: tuple[int, int] | None = None

    def build_roi_mask(self, height: int, width: int) -> np.ndarray:
        if self._cached_mask is not None and self._cached_shape == (height, width):
            return self._cached_mask

        mask = np.zeros((height, width), dtype=np.uint8)
        y_top = int(np.clip(round(height * self.roi_top_ratio), 0, height - 1))
        y_bot = int(np.clip(round(height * self.roi_bottom_ratio), y_top + 1, height))
        x_margin = int(round(width * 0.03))
        mask[y_top:y_bot, x_margin : width - x_margin] = 255

        self._cached_mask = mask
        self._cached_shape = (height, width)
        return mask

    def detect(self, gray_frame: np.ndarray) -> np.ndarray:
        h, w = gray_frame.shape[:2]
        mask = self.build_roi_mask(h, w)
        corners = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance_px,
            mask=mask,
            blockSize=7,
        )
        if corners is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return corners.astype(np.float32)