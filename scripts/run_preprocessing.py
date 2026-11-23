import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    CoordinateTransformer,
    IMUCalibrator,
    RecordingManifest,
    SensorDataLoader,
    TimestampSynchronizer,
    VideoLoader,
)


class PreprocessingPipeline:
    """Orchestrates Phase 1 Data Gate, calibration, coordinate projection, and sync."""

    def __init__(self, manifest: RecordingManifest, recording_id: str) -> None:
        self.manifest = manifest
        self.rec_cfg = manifest.get_recording(recording_id)
        self.prep_cfg = manifest.preprocessing_params
        self.active_sensors = manifest.active_sensors

    def run(self) -> Dict[str, Any]:
        # 1. Load & validate active sensor streams (accr, gyro, gps only)
        sensor_loader = SensorDataLoader(
            gps_max_speed_m_s=float(self.prep_cfg["gps_max_speed_m_s"]),
            active_sensors=self.active_sensors,
        )
        raw_df = sensor_loader.load_raw(
            self.rec_cfg.sensor_path,
            has_header=self.rec_cfg.csv_has_header,
        )
        streams = sensor_loader.split_streams(raw_df)

        # 2. Project GPS coordinates to local ENU frame
        coord_tf = CoordinateTransformer()
        streams["gps"], enu_origin = coord_tf.project_gps_dataframe(streams["gps"])

        # 3. Calibrate raw accelerometer (gravity removal) & gyroscope into vehicle axes
        calibrator = IMUCalibrator(
            forward_axis_hint=self.rec_cfg.imu_mounting.forward_axis,
            up_axis_hint=self.rec_cfg.imu_mounting.up_axis_hint,
            yaw_sign=self.rec_cfg.imu_mounting.yaw_sign,
            stationary_init_seconds=float(self.prep_cfg["stationary_init_seconds"]),
            gyro_stationary_thresh_rad_s=float(
                self.prep_cfg["gyro_stationary_thresh_rad_s"]
            ),
        )
        streams["accr"], streams["gyro"], imu_calib_res = calibrator.fit_transform(
            streams["accr"], streams["gyro"]
        )

        # Save cleaned streams to interim directory
        for stype, sdf in streams.items():
            sdf.to_csv(self.rec_cfg.interim_dir / f"{stype}_clean.csv", index=False)

        # 4. Inspect video metadata & frame timestamps
        video_loader = VideoLoader(self.rec_cfg.video_path)
        video_meta = video_loader.inspect_metadata()
        video_df = video_loader.build_frame_table(video_meta)
        video_df.to_csv(self.rec_cfg.interim_dir / "video_frames.csv", index=False)

        # 5. Synchronize sensor streams onto video frame timestamps
        synchronizer = TimestampSynchronizer(
            offset_s=self.rec_cfg.video_to_sensor_offset_s,
            max_staleness_s=self.prep_cfg.get("max_staleness_s"),
        )
        synced_df = synchronizer.align_to_video_frames(video_df, streams)
        synced_path = self.rec_cfg.processed_dir / "synchronized_frames.csv"
        synced_df.to_csv(synced_path, index=False)

        # 6. Build and persist validation report
        report: Dict[str, Any] = {
            "recording_id": self.rec_cfg.recording_id,
            "active_sensors": self.active_sensors,
            "sensor_summary": sensor_loader.summarize(streams),
            "video_metadata": video_meta.to_dict(),
            "enu_origin": enu_origin.to_dict(),
            "imu_calibration": imu_calib_res.to_dict(),
            "sync_coverage": {
                "total_video_frames": int(len(synced_df)),
                "acc_valid_pct": round(float(synced_df["acc_valid"].mean() * 100.0), 2),
                "gyro_valid_pct": round(float(synced_df["gyro_valid"].mean() * 100.0), 2),
                "gps_valid_pct": round(float(synced_df["gps_valid"].mean() * 100.0), 2),
            },
        }
        report_path = self.rec_cfg.processed_dir / "phase1_validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: OOP Data Gate & Preprocessing")
    parser.add_argument("--recording", default="drive_20")
    parser.add_argument("--default-cfg", default="configs/default.yaml")
    parser.add_argument("--recordings-cfg", default="configs/recordings.yaml")
    args = parser.parse_args()

    manifest = RecordingManifest(args.default_cfg, args.recordings_cfg)
    pipeline = PreprocessingPipeline(manifest, args.recording)
    report = pipeline.run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()