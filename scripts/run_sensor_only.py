import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ekf import SensorOnlyEKF
from src.preprocessing import RecordingManifest
from src.sensors import (
    AccelerometerProcessor,
    GPSProcessor,
    GyroscopeProcessor,
    IMUPreprocessor,
    SensorTimelineBuilder,
)


class SensorOnlyPipeline:
    """Runs Track A (Sensor-Only EKF) and aligns output onto common video frame timestamps."""

    def __init__(
        self,
        manifest: RecordingManifest,
        recording_id: str,
        sensor_ekf_yaml: str | Path = "configs/sensor_ekf.yaml",
    ) -> None:
        self.rec_cfg = manifest.get_recording(recording_id)
        self.prep_cfg = manifest.preprocessing_params
        self.ekf_cfg = RecordingManifest._read_yaml(Path(sensor_ekf_yaml))["sensor_ekf"]

    def run(self) -> Dict[str, Any]:
        acc_df = pd.read_csv(self.rec_cfg.interim_dir / "accr_clean.csv")
        gyro_df = pd.read_csv(self.rec_cfg.interim_dir / "gyro_clean.csv")
        gps_df = pd.read_csv(self.rec_cfg.interim_dir / "gps_clean.csv")
        synced_frames = pd.read_csv(self.rec_cfg.processed_dir / "synchronized_frames.csv")

        zupt_cfg = self.ekf_cfg["zupt"]
        imu_prep = IMUPreprocessor(
            acc_processor=AccelerometerProcessor(smoothing_window=3),
            gyro_processor=GyroscopeProcessor(smoothing_window=3),
            zupt_gyro_thresh_rad_s=float(zupt_cfg["gyro_thresh_rad_s"]),
            zupt_accel_thresh_m_s2=float(zupt_cfg["accel_thresh_m_s2"]),
            zupt_min_duration_s=float(zupt_cfg["min_duration_s"]),
        )
        imu_timeline = imu_prep.build_imu_timeline(acc_df, gyro_df)

        gps_proc = GPSProcessor(
            base_pos_std_m=float(self.ekf_cfg["measurement_noise"]["gps_pos_std_m"]),
            max_speed_m_s=float(self.prep_cfg["gps_max_speed_m_s"]),
            holdout_intervals_s=self.ekf_cfg.get("holdout_intervals_s", []),
        )
        gps_processed = gps_proc.process(gps_df)

        packets = SensorTimelineBuilder.build_packets(imu_timeline, gps_processed)
        estimator = SensorOnlyEKF(self.ekf_cfg)
        full_traj = estimator.run(packets)

        out_dir = self.rec_cfg.output_dir / "sensor_only"
        out_dir.mkdir(parents=True, exist_ok=True)
        full_traj.to_csv(out_dir / "sensor_only_full_rate.csv", index=False)

        # Resample onto common synchronized video timestamps (t_sync_s) for 3-track comparison
        frame_aligned = self._resample_to_frames(full_traj, synced_frames)
        frame_aligned.to_csv(out_dir / "sensor_only_frame_aligned.csv", index=False)

        gps_events = full_traj[full_traj["event_type"] == "gps"]
        summary: Dict[str, Any] = {
            "recording_id": self.rec_cfg.recording_id,
            "track": "A_sensor_only_ekf",
            "total_ekf_steps": int(len(full_traj)),
            "gps_updates_attempted": int(len(gps_events)),
            "gps_updates_accepted": int(gps_events["gps_accepted"].sum()),
            "zupt_steps": int(full_traj["zupt_applied"].sum()),
            "total_path_length_m": round(
                float(np.sum(np.hypot(np.diff(full_traj["px_m"]), np.diff(full_traj["py_m"])))),
                2,
            ),
            "max_speed_m_s": round(float(full_traj["speed_m_s"].max()), 2),
            "mean_pos_std_m": round(
                float(np.hypot(full_traj["std_px_m"], full_traj["std_py_m"]).mean()), 2
            ),
            "frame_aligned_rows": int(len(frame_aligned)),
        }
        with open(out_dir / "sensor_only_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    @staticmethod
    def _resample_to_frames(
        full_traj: pd.DataFrame,
        synced_frames: pd.DataFrame,
    ) -> pd.DataFrame:
        # Keep latest state at any duplicate timestamp before asof join
        dedup = full_traj.drop_duplicates(subset=["t_sec"], keep="last").sort_values("t_sec")
        aligned = pd.merge_asof(
            synced_frames[["frame_idx", "t_video_s", "t_sync_s", "gps_valid", "gps_x_east_m", "gps_y_north_m"]].sort_values("t_sync_s"),
            dedup[
                [
                    "t_sec",
                    "px_m",
                    "py_m",
                    "speed_m_s",
                    "heading_rad",
                    "yaw_rate_rad_s",
                    "acc_bias_m_s2",
                    "gyro_bias_rad_s",
                    "std_px_m",
                    "std_py_m",
                    "std_speed_m_s",
                    "std_heading_rad",
                ]
            ].rename(columns={"t_sec": "t_ekf_s"}),
            left_on="t_sync_s",
            right_on="t_ekf_s",
            direction="nearest",
            tolerance=0.25,
        )
        aligned["track_a_valid"] = aligned["px_m"].notna()
        return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Track A Sensor-Only EKF")
    parser.add_argument("--recording", default="drive_20")
    parser.add_argument("--default-cfg", default="configs/default.yaml")
    parser.add_argument("--recordings-cfg", default="configs/recordings.yaml")
    parser.add_argument("--sensor-ekf-cfg", default="configs/sensor_ekf.yaml")
    args = parser.parse_args()

    manifest = RecordingManifest(args.default_cfg, args.recordings_cfg)
    pipeline = SensorOnlyPipeline(manifest, args.recording, args.sensor_ekf_cfg)
    summary = pipeline.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()