import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ekf import FusionEKF
from src.preprocessing import RecordingManifest
from src.sensors import (
    AccelerometerProcessor,
    GPSProcessor,
    GyroscopeProcessor,
    IMUPreprocessor,
    SensorTimelineBuilder,
)
from src.vision import VisualMeasurementPacket


class SensorFusionPipeline:
    """Runs Track C (Sensor + Optical Flow Fusion EKF) and aligns output onto video frames."""

    def __init__(
        self,
        manifest: RecordingManifest,
        recording_id: str,
        fusion_ekf_yaml: str | Path = "configs/fusion_ekf.yaml",
    ) -> None:
        self.rec_cfg = manifest.get_recording(recording_id)
        self.prep_cfg = manifest.preprocessing_params
        self.fusion_cfg = RecordingManifest._read_yaml(Path(fusion_ekf_yaml))["fusion_ekf"]

    @staticmethod
    def _load_visual_packets(vo_csv_path: Path) -> List[VisualMeasurementPacket]:
        if not vo_csv_path.exists():
            raise FileNotFoundError(
                f"Visual odometry output not found: {vo_csv_path}. Run Phase 3 (run_optical_flow_only.py) first."
            )
        df = pd.read_csv(vo_csv_path)
        # Smooth visual yaw rate and forward speed lightly to suppress frame-pair noise
        df["yaw_rate_smooth"] = (
            df["yaw_rate_rad_s"].rolling(window=5, min_periods=1, center=True).median()
        )
        df["fwd_speed_smooth"] = (
            df["forward_speed"].rolling(window=5, min_periods=1, center=True).median()
        )

        packets: List[VisualMeasurementPacket] = []
        for row in df.itertuples(index=False):
            packets.append(
                VisualMeasurementPacket(
                    frame_idx=int(row.frame_idx),
                    t_video_s=float(row.t_video_s),
                    t_sync_s=float(row.t_sync_s),
                    dt_s=float(row.dt_s),
                    delta_yaw_rad=float(row.delta_yaw_rad),
                    yaw_rate_rad_s=float(row.yaw_rate_smooth),
                    delta_forward=float(row.delta_forward),
                    forward_speed=float(row.fwd_speed_smooth),
                    inlier_ratio=float(row.inlier_ratio),
                    inlier_count=int(row.inlier_count),
                    median_flow_px=float(row.median_flow_px),
                    is_stationary=bool(row.is_stationary),
                    is_valid=bool(row.is_valid),
                    is_metric=bool(row.is_metric),
                    scale_status=str(row.scale_status),
                )
            )
        return packets

    def run(self) -> Dict[str, Any]:
        acc_df = pd.read_csv(self.rec_cfg.interim_dir / "accr_clean.csv")
        gyro_df = pd.read_csv(self.rec_cfg.interim_dir / "gyro_clean.csv")
        gps_df = pd.read_csv(self.rec_cfg.interim_dir / "gps_clean.csv")
        synced_frames = pd.read_csv(self.rec_cfg.processed_dir / "synchronized_frames.csv")

        vo_raw_path = self.rec_cfg.output_dir / "optical_flow_only" / "visual_odometry_raw.csv"
        visual_packets = self._load_visual_packets(vo_raw_path)

        zupt_cfg = self.fusion_cfg["zupt"]
        imu_prep = IMUPreprocessor(
            acc_processor=AccelerometerProcessor(smoothing_window=5),
            gyro_processor=GyroscopeProcessor(smoothing_window=3),
            zupt_gyro_thresh_rad_s=float(zupt_cfg["gyro_thresh_rad_s"]),
            zupt_accel_thresh_m_s2=float(zupt_cfg["accel_thresh_m_s2"]),
            zupt_accel_std_thresh_m_s2=float(zupt_cfg["accel_rolling_std_thresh_m_s2"]),
            zupt_min_duration_s=float(zupt_cfg["min_duration_s"]),
        )
        imu_timeline = imu_prep.build_imu_timeline(acc_df, gyro_df)

        gps_proc = GPSProcessor(
            base_pos_std_m=float(self.fusion_cfg["measurement_noise"]["gps_pos_std_m"]),
            max_speed_m_s=float(self.prep_cfg["gps_max_speed_m_s"]),
            holdout_intervals_s=self.fusion_cfg.get("holdout_intervals_s", []),
        )
        gps_processed = gps_proc.process(gps_df)
        sensor_packets = SensorTimelineBuilder.build_packets(imu_timeline, gps_processed)

        fusion_estimator = FusionEKF(self.fusion_cfg)
        full_traj = fusion_estimator.run(sensor_packets, visual_packets)

        out_dir = self.rec_cfg.output_dir / "sensor_fusion"
        out_dir.mkdir(parents=True, exist_ok=True)
        full_traj.to_csv(out_dir / "sensor_fusion_full_rate.csv", index=False)

        frame_aligned = self._resample_to_frames(full_traj, synced_frames)
        frame_aligned.to_csv(out_dir / "sensor_fusion_frame_aligned.csv", index=False)

        gps_events = full_traj[full_traj["event_type"] == "gps"]
        summary: Dict[str, Any] = {
            "recording_id": self.rec_cfg.recording_id,
            "track": "C_sensor_vision_fusion_ekf",
            "total_ekf_steps": int(len(full_traj)),
            "gps_updates_attempted": int(len(gps_events)),
            "gps_updates_accepted": int(gps_events["gps_accepted"].sum()),
            "visual_yaw_updates_accepted": int(full_traj["vis_yaw_accepted"].sum()),
            "visual_outage_speed_updates": int(full_traj["vis_speed_applied"].sum()),
            "total_zupt_steps": int(full_traj["zupt_applied"].sum()),
            "final_visual_scale_m_per_unit": round(float(full_traj["visual_scale_est"].iloc[-1]), 4),
            "total_path_length_m": round(
                float(
                    np.sum(
                        np.hypot(
                            np.diff(full_traj["px_fused_m"]),
                            np.diff(full_traj["py_fused_m"]),
                        )
                    )
                ),
                2,
            ),
            "max_speed_m_s": round(float(full_traj["speed_fused_m_s"].max()), 2),
            "mean_pos_std_m": round(
                float(np.hypot(full_traj["std_px_fused_m"], full_traj["std_py_fused_m"]).mean()),
                2,
            ),
            "frame_aligned_rows": int(len(frame_aligned)),
        }
        with open(out_dir / "sensor_fusion_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    @staticmethod
    def _resample_to_frames(
        full_traj: pd.DataFrame,
        synced_frames: pd.DataFrame,
    ) -> pd.DataFrame:
        dedup = full_traj.drop_duplicates(subset=["t_sec"], keep="last").sort_values("t_sec")
        aligned = pd.merge_asof(
            synced_frames[["frame_idx", "t_video_s", "t_sync_s"]].sort_values("t_sync_s"),
            dedup[
                [
                    "t_sec",
                    "px_fused_m",
                    "py_fused_m",
                    "speed_fused_m_s",
                    "heading_fused_rad",
                    "yaw_rate_fused_rad_s",
                    "acc_bias_fused_m_s2",
                    "gyro_bias_fused_rad_s",
                    "visual_scale_est",
                    "std_px_fused_m",
                    "std_py_fused_m",
                    "std_speed_fused_m_s",
                    "std_heading_fused_rad",
                ]
            ].rename(columns={"t_sec": "t_fused_s"}),
            left_on="t_sync_s",
            right_on="t_fused_s",
            direction="nearest",
            tolerance=0.25,
        )
        aligned["track_c_valid"] = aligned["px_fused_m"].notna()
        return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Track C Sensor-Vision Fusion EKF")
    parser.add_argument("--recording", default="drive_20")
    parser.add_argument("--default-cfg", default="configs/default.yaml")
    parser.add_argument("--recordings-cfg", default="configs/recordings.yaml")
    parser.add_argument("--fusion-ekf-cfg", default="configs/fusion_ekf.yaml")
    args = parser.parse_args()

    manifest = RecordingManifest(args.default_cfg, args.recordings_cfg)
    pipeline = SensorFusionPipeline(manifest, args.recording, args.fusion_ekf_cfg)
    summary = pipeline.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()