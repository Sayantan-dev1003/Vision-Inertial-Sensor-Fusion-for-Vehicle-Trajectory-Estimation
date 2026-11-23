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

from scripts.run_sensor_fusion import SensorFusionPipeline
from src.evaluation import GPSHoldoutEvaluator, StopDetector, TurnDetector
from src.preprocessing import RecordingManifest
from src.sensors import AccelerometerProcessor, GyroscopeProcessor, IMUPreprocessor
from src.visualization import TrajectoryPlotter, TrajectoryVideoOverlay


class EvaluationPipeline:
    """Phase 5 Orchestrator: Turn & Stop Localization, GPS Holdout Benchmark, Plots & Full Side-by-Side Videos."""

    def __init__(
        self,
        manifest: RecordingManifest,
        recording_id: str,
        sensor_ekf_yaml: str | Path = "configs/sensor_ekf.yaml",
        fusion_ekf_yaml: str | Path = "configs/fusion_ekf.yaml",
        eval_yaml: str | Path = "configs/evaluation.yaml",
    ) -> None:
        self.rec_cfg = manifest.get_recording(recording_id)
        self.prep_cfg = manifest.preprocessing_params
        self.sensor_cfg = RecordingManifest._read_yaml(Path(sensor_ekf_yaml))["sensor_ekf"]
        self.fusion_cfg = RecordingManifest._read_yaml(Path(fusion_ekf_yaml))["fusion_ekf"]
        self.eval_cfg = RecordingManifest._read_yaml(Path(eval_yaml))["evaluation"]

    def run(self) -> Dict[str, Any]:
        ta_aligned = pd.read_csv(self.rec_cfg.output_dir / "sensor_only" / "sensor_only_frame_aligned.csv")
        tb_aligned = pd.read_csv(self.rec_cfg.output_dir / "optical_flow_only" / "optical_flow_frame_aligned.csv")
        tc_aligned = pd.read_csv(self.rec_cfg.output_dir / "sensor_fusion" / "sensor_fusion_frame_aligned.csv")

        eval_dir = self.rec_cfg.output_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # 1. Turn Detection (with start, apex, end x, y, timestamp)
        turn_detector = TurnDetector(self.eval_cfg["turn_detection"])
        turns_df = turn_detector.detect(tc_aligned, ta_aligned, tb_aligned)
        turns_csv = eval_dir / "detected_turns.csv"
        turns_df.to_csv(turns_csv, index=False)

        # 2. Stop Detection (with x, y, start/end timestamp, duration)
        stop_detector = StopDetector(self.eval_cfg["stop_detection"])
        stops_df = stop_detector.detect(tc_aligned, ta_aligned)
        stops_csv = eval_dir / "detected_stops.csv"
        stops_df.to_csv(stops_csv, index=False)

        # 3. Full-Drive Track Comparison vs GPS
        valid_gps = ta_aligned[ta_aligned["gps_valid"] & ta_aligned["track_a_valid"] & tc_aligned["track_c_valid"]].copy()
        err_a_full = np.hypot(
            valid_gps["px_m"] - valid_gps["gps_x_east_m"],
            valid_gps["py_m"] - valid_gps["gps_y_north_m"],
        )
        err_c_full = np.hypot(
            tc_aligned.loc[valid_gps.index, "px_fused_m"] - valid_gps["gps_x_east_m"],
            tc_aligned.loc[valid_gps.index, "py_fused_m"] - valid_gps["gps_y_north_m"],
        )

        # 4. Non-Circular GPS Holdout Evaluation
        acc_df = pd.read_csv(self.rec_cfg.interim_dir / "accr_clean.csv")
        gyro_df = pd.read_csv(self.rec_cfg.interim_dir / "gyro_clean.csv")
        gps_df = pd.read_csv(self.rec_cfg.interim_dir / "gps_clean.csv")
        vo_raw_path = self.rec_cfg.output_dir / "optical_flow_only" / "visual_odometry_raw.csv"
        visual_packets = SensorFusionPipeline._load_visual_packets(vo_raw_path)

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

        ho_eval = GPSHoldoutEvaluator(
            sensor_ekf_cfg=self.sensor_cfg,
            fusion_ekf_cfg=self.fusion_cfg,
            gps_max_speed_m_s=float(self.prep_cfg["gps_max_speed_m_s"]),
            holdout_intervals_s=self.eval_cfg["gps_holdout"]["intervals_s"],
        )
        holdout_res = ho_eval.evaluate_holdout(imu_timeline, gps_df, visual_packets)

        # 5. Generate 300-DPI BTP Plots (including Annotated Turns & Stops Map)
        saved_plots: List[str] = []
        if self.eval_cfg["plots"]["save_figures"]:
            plotter = TrajectoryPlotter(eval_dir / "plots", dpi=int(self.eval_cfg["plots"]["dpi"]))
            saved_plots = plotter.plot_all(ta_aligned, tb_aligned, tc_aligned, turns_df, stops_df, holdout_res)

        # 6. Render Full-Video Side-by-Side Comparison Videos (Track A & Track C separately)
        rendered_videos: Dict[str, str] = {}
        ov_cfg = self.eval_cfg["video_overlay"]
        if ov_cfg["enabled"]:
            start_s = 0.0 if ov_cfg.get("full_video", True) else float(ov_cfg.get("start_time_s", 0.0))
            dur_s = None if ov_cfg.get("full_video", True) else ov_cfg.get("duration_s", None)
            renderer = TrajectoryVideoOverlay(
                video_path=self.rec_cfg.video_path,
                output_dir=eval_dir,
                recording_id=self.rec_cfg.recording_id,
                start_time_s=start_s,
                duration_s=dur_s,
                frame_stride=int(ov_cfg.get("frame_stride", 1)),
                output_fps=float(ov_cfg.get("output_fps", 30.0)),
            )
            if ov_cfg.get("render_track_a", True):
                rendered_videos["track_a_side_by_side"] = renderer.render_side_by_side_track(
                    track_label="Track A: Sensor-Only EKF (IMU + GPS)",
                    output_filename=f"{self.rec_cfg.recording_id}_track_a_side_by_side.mp4",
                    ta_df=ta_aligned,
                    tc_df=tc_aligned,
                    turns_df=turns_df,
                    stops_df=stops_df,
                    use_track_a=True,
                )
            if ov_cfg.get("render_track_c", True):
                rendered_videos["track_c_side_by_side"] = renderer.render_side_by_side_track(
                    track_label="Track C: Sensor + Optical Flow Fusion EKF",
                    output_filename=f"{self.rec_cfg.recording_id}_track_c_side_by_side.mp4",
                    ta_df=ta_aligned,
                    tc_df=tc_aligned,
                    turns_df=turns_df,
                    stops_df=stops_df,
                    use_track_a=False,
                )

        report: Dict[str, Any] = {
            "recording_id": self.rec_cfg.recording_id,
            "full_drive_comparison": {
                "track_a_rmse_m": round(float(np.sqrt(np.mean(err_a_full**2))), 2),
                "track_c_rmse_m": round(float(np.sqrt(np.mean(err_c_full**2))), 2),
                "track_a_max_outage_err_m": round(float(np.max(err_a_full)), 2),
                "track_c_max_outage_err_m": round(float(np.max(err_c_full)), 2),
            },
            "turn_detection_summary": {
                "total_turns_detected": int(len(turns_df)),
                "maneuver_breakdown": turns_df["maneuver_type"].value_counts().to_dict() if not turns_df.empty else {},
                "turns_csv_path": str(turns_csv),
            },
            "stop_detection_summary": {
                "total_stops_detected": int(len(stops_df)),
                "total_stopped_duration_s": round(float(stops_df["duration_s"].sum()), 2) if not stops_df.empty else 0.0,
                "stops_csv_path": str(stops_csv),
            },
            "gps_holdout_benchmark": holdout_res["summary"],
            "gps_holdout_windows": holdout_res["per_window"],
            "saved_plots": saved_plots,
            "side_by_side_videos": rendered_videos,
        }

        with open(eval_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: Full-Video Side-by-Side Evaluation, Turns & Stops")
    parser.add_argument("--recording", default="drive_20")
    parser.add_argument("--default-cfg", default="configs/default.yaml")
    parser.add_argument("--recordings-cfg", default="configs/recordings.yaml")
    parser.add_argument("--sensor-ekf-cfg", default="configs/sensor_ekf.yaml")
    parser.add_argument("--fusion-ekf-cfg", default="configs/fusion_ekf.yaml")
    parser.add_argument("--eval-cfg", default="configs/evaluation.yaml")
    args = parser.parse_args()

    manifest = RecordingManifest(args.default_cfg, args.recordings_cfg)
    pipeline = EvaluationPipeline(
        manifest=manifest,
        recording_id=args.recording,
        sensor_ekf_yaml=args.sensor_ekf_cfg,
        fusion_ekf_yaml=args.fusion_ekf_cfg,
        eval_yaml=args.eval_cfg,
    )
    report = pipeline.run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()