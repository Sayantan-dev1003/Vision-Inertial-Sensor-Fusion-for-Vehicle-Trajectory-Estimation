import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ekf import VisualOnlyEKF
from src.preprocessing import RecordingManifest, TimestampSynchronizer
from src.vision import (
    CameraCalibrator,
    FeatureDetector,
    LucasKanadeOpticalFlowTracker,
    MonocularVisualOdometry,
    MotionScaleEstimator,
    VisualMeasurementBuilder,
    VisualMeasurementPacket,
)


class OpticalFlowOnlyPipeline:
    """Runs Track B (Optical-Flow-Only Visual Odometry + Kalman Filtering)."""

    def __init__(
        self,
        manifest: RecordingManifest,
        recording_id: str,
        camera_yaml: str | Path = "configs/camera.yaml",
    ) -> None:
        self.rec_cfg = manifest.get_recording(recording_id)
        self.cam_yaml = RecordingManifest._read_yaml(Path(camera_yaml))
        self.cam_cfg = self.cam_yaml["camera"]
        self.flow_cfg = self.cam_yaml["optical_flow"]
        self.vis_ekf_cfg = self.cam_yaml["visual_ekf"]

    def extract_visual_measurements(self, synced_frames: pd.DataFrame) -> List[VisualMeasurementPacket]:
        calibrator = CameraCalibrator(self.cam_cfg)
        scale = float(self.flow_cfg.get("process_scale", 0.5))
        stride = max(1, int(self.flow_cfg.get("frame_stride", 2)))
        intrinsics = calibrator.get_scaled_intrinsics(scale)

        detector = FeatureDetector(
            max_corners=int(self.flow_cfg["max_corners"]),
            quality_level=float(self.flow_cfg["quality_level"]),
            min_distance_px=int(self.flow_cfg["min_distance_px"]),
            roi_top_ratio=float(self.flow_cfg["roi_top_ratio"]),
            roi_bottom_ratio=float(self.flow_cfg["roi_bottom_ratio"]),
        )
        tracker = LucasKanadeOpticalFlowTracker(
            fb_error_thresh_px=float(self.flow_cfg["fb_error_thresh_px"]),
        )
        vo = MonocularVisualOdometry(
            intrinsics=intrinsics,
            ransac_prob=float(self.flow_cfg["ransac_prob"]),
            ransac_thresh_px=float(self.flow_cfg["ransac_thresh_px"]),
            min_inliers=int(self.flow_cfg["min_inliers"]),
            stationary_flow_px=float(self.flow_cfg["stationary_flow_px"]),
        )
        scale_est = MotionScaleEstimator(
            intrinsics=intrinsics,
            extrinsics=calibrator.extrinsics,
            scale_mode=str(self.flow_cfg.get("scale_mode", "relative_unit")),
        )

        cap = cv2.VideoCapture(str(self.rec_cfg.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.rec_cfg.video_path}")

        t_video_arr = synced_frames["t_video_s"].to_numpy(dtype=float)
        t_sync_arr = synced_frames["t_sync_s"].to_numpy(dtype=float)
        total_frames = len(synced_frames)

        packets: List[VisualMeasurementPacket] = []
        prev_gray: np.ndarray | None = None
        prev_pts: np.ndarray | None = None
        prev_t: float = 0.0
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= total_frames:
                    break

                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue

                curr_gray = calibrator.prepare_gray_frame(frame, scale=scale)
                t_vid = float(t_video_arr[frame_idx])
                t_syn = float(t_sync_arr[frame_idx])

                if prev_gray is None:
                    prev_gray = curr_gray
                    prev_pts = detector.detect(curr_gray)
                    prev_t = t_syn
                    packets.append(
                        VisualMeasurementPacket(
                            frame_idx=frame_idx,
                            t_video_s=t_vid,
                            t_sync_s=t_syn,
                            dt_s=0.0,
                            delta_yaw_rad=0.0,
                            yaw_rate_rad_s=0.0,
                            delta_forward=0.0,
                            forward_speed=0.0,
                            inlier_ratio=1.0,
                            inlier_count=len(prev_pts),
                            median_flow_px=0.0,
                            is_stationary=True,
                            is_valid=True,
                            is_metric=(scale_est.scale_mode == "planar_ground" and intrinsics.is_calibrated),
                            scale_status=(
                                "metric_planar_scaled"
                                if (scale_est.scale_mode == "planar_ground" and intrinsics.is_calibrated)
                                else "relative_uncalibrated"
                            ),
                        )
                    )
                else:
                    dt = t_syn - prev_t
                    if prev_pts is None or len(prev_pts) < int(self.flow_cfg["min_inliers"]):
                        prev_pts = detector.detect(prev_gray)

                    flow_res = tracker.track(prev_gray, curr_gray, prev_pts)
                    vo_res = vo.estimate_step(flow_res, dt)
                    sc_res = scale_est.estimate(vo_res, flow_res, dt, curr_gray.shape[0])

                    packets.append(
                        VisualMeasurementPacket(
                            frame_idx=frame_idx,
                            t_video_s=t_vid,
                            t_sync_s=t_syn,
                            dt_s=dt,
                            delta_yaw_rad=vo_res.delta_yaw_rad,
                            yaw_rate_rad_s=vo_res.yaw_rate_rad_s,
                            delta_forward=sc_res.delta_forward_distance,
                            forward_speed=sc_res.forward_speed,
                            inlier_ratio=vo_res.inlier_ratio,
                            inlier_count=vo_res.inlier_count,
                            median_flow_px=vo_res.median_flow_px,
                            is_stationary=vo_res.is_stationary,
                            is_valid=vo_res.is_valid,
                            is_metric=sc_res.is_metric,
                            scale_status=sc_res.scale_status,
                        )
                    )

                    # Refresh corner detections for next pair
                    prev_gray = curr_gray
                    prev_pts = detector.detect(curr_gray)
                    prev_t = t_syn

                if frame_idx > 0 and frame_idx % 6000 == 0:
                    print(f"[Track B] Processed {frame_idx}/{total_frames} frames...")

                frame_idx += 1
        finally:
            cap.release()

        return packets

    def run(self) -> Dict[str, Any]:
        synced_frames = pd.read_csv(self.rec_cfg.processed_dir / "synchronized_frames.csv")
        gyro_df = pd.read_csv(self.rec_cfg.interim_dir / "gyro_clean.csv")

        packets = self.extract_visual_measurements(synced_frames)
        raw_vo_df = VisualMeasurementBuilder.to_dataframe(packets)

        # Cross-validate video-to-sensor offset against gyroscope yaw rate
        est_offset_s, offset_corr = TimestampSynchronizer.estimate_video_sensor_offset(
            t_video_s=raw_vo_df["t_video_s"].to_numpy(dtype=float),
            visual_rot_rate=raw_vo_df["yaw_rate_rad_s"].to_numpy(dtype=float),
            t_gyro_s=gyro_df.loc[gyro_df["is_valid"], "t_sec"].to_numpy(dtype=float),
            gyro_yaw_rate=gyro_df.loc[gyro_df["is_valid"], "yaw_rate_rad_s"].to_numpy(dtype=float),
            search_range_s=(0.0, 40.0),
            step_s=0.1,
        )

        vis_ekf = VisualOnlyEKF(self.vis_ekf_cfg)
        vis_traj = vis_ekf.run(packets)

        out_dir = self.rec_cfg.output_dir / "optical_flow_only"
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_vo_df.to_csv(out_dir / "visual_odometry_raw.csv", index=False)

        # Align Track B estimates onto all synchronized video frames
        frame_aligned = pd.merge_asof(
            synced_frames[["frame_idx", "t_video_s", "t_sync_s"]].sort_values("t_sync_s"),
            vis_traj.drop(columns=["frame_idx", "t_video_s"]).sort_values("t_sync_s"),
            on="t_sync_s",
            direction="nearest",
            tolerance=0.25,
        )
        frame_aligned.to_csv(out_dir / "optical_flow_frame_aligned.csv", index=False)

        summary: Dict[str, Any] = {
            "recording_id": self.rec_cfg.recording_id,
            "track": "B_optical_flow_only",
            "processed_vo_steps": int(len(raw_vo_df)),
            "valid_vo_pct": round(float(raw_vo_df["is_valid"].mean() * 100.0), 2),
            "stationary_vo_pct": round(float(raw_vo_df["is_stationary"].mean() * 100.0), 2),
            "mean_inlier_ratio": round(float(raw_vo_df["inlier_ratio"].mean()), 4),
            "scale_status": str(raw_vo_df["scale_status"].iloc[0]),
            "configured_offset_s": self.rec_cfg.video_to_sensor_offset_s,
            "visual_gyro_estimated_offset_s": est_offset_s,
            "visual_gyro_offset_correlation": offset_corr,
            "frame_aligned_rows": int(len(frame_aligned)),
        }
        with open(out_dir / "optical_flow_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Track B Optical-Flow-Only Pipeline")
    parser.add_argument("--recording", default="drive_20")
    parser.add_argument("--default-cfg", default="configs/default.yaml")
    parser.add_argument("--recordings-cfg", default="configs/recordings.yaml")
    parser.add_argument("--camera-cfg", default="configs/camera.yaml")
    args = parser.parse_args()

    manifest = RecordingManifest(args.default_cfg, args.recordings_cfg)
    pipeline = OpticalFlowOnlyPipeline(manifest, args.recording, args.camera_cfg)
    summary = pipeline.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()