from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class OpticalFlowResult:
    pts_prev: np.ndarray          # (N, 2) matched points in frame k-1
    pts_curr: np.ndarray          # (N, 2) matched points in frame k
    flow_vectors: np.ndarray      # (N, 2) pixel displacement (dx, dy)
    median_flow_mag_px: float
    tracked_count: int


class LucasKanadeOpticalFlowTracker:
    """Pyramidal Lucas-Kanade tracker with forward-backward consistency filtering."""

    def __init__(
        self,
        win_size: tuple[int, int] = (21, 21),
        max_level: int = 3,
        fb_error_thresh_px: float = 1.5,
    ) -> None:
        self.win_size = tuple(win_size)
        self.max_level = int(max_level)
        self.fb_error_thresh_px = float(fb_error_thresh_px)
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        )

    def track(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray,
    ) -> OpticalFlowResult:
        if prev_pts is None or len(prev_pts) == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return OpticalFlowResult(empty, empty, empty, 0.0, 0)

        curr_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria,
        )
        if curr_pts is None or st_fwd is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return OpticalFlowResult(empty, empty, empty, 0.0, 0)

        # Forward-backward check to eliminate drifted or occluded tracks
        back_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray,
            prev_gray,
            curr_pts,
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria,
        )
        if back_pts is None or st_bwd is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return OpticalFlowResult(empty, empty, empty, 0.0, 0)

        p0 = prev_pts.reshape(-1, 2)
        p1 = curr_pts.reshape(-1, 2)
        p0_back = back_pts.reshape(-1, 2)

        fb_err = np.linalg.norm(p0 - p0_back, axis=1)
        valid = (
            (st_fwd.reshape(-1) == 1)
            & (st_bwd.reshape(-1) == 1)
            & (fb_err <= self.fb_error_thresh_px)
        )

        p0_valid = p0[valid]
        p1_valid = p1[valid]
        flow = p1_valid - p0_valid
        med_mag = float(np.median(np.linalg.norm(flow, axis=1))) if len(flow) > 0 else 0.0

        return OpticalFlowResult(
            pts_prev=p0_valid,
            pts_curr=p1_valid,
            flow_vectors=flow,
            median_flow_mag_px=med_mag,
            tracked_count=int(len(p0_valid)),
        )