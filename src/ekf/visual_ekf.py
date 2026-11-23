from typing import Any, Dict, List
import numpy as np
import pandas as pd

from src.vision.visual_measurements import VisualMeasurementPacket
from .ekf_core import ExtendedKalmanFilter
from .motion_model import PlanarVehicleMotionModel


class VisualOnlyEKF:
    """Track B Estimator: Smooths visual odometry yaw rate & relative/metric forward speed."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        init_u = cfg["initial_uncertainty"]
        proc_n = cfg["process_noise"]
        meas_n = cfg["measurement_noise"]

        self.motion_model = PlanarVehicleMotionModel(
            accel_noise_std=float(proc_n["accel_noise_std"]),
            gyro_noise_std=float(proc_n["gyro_noise_std_rad_s"]),
            acc_bias_walk_std=1e-6,
            gyro_bias_walk_std=1e-6,
            max_dt_s=1.0,
        )
        self.init_u = init_u
        self.vel_std_base = float(meas_n["vel_std_base"])
        self.zupt_vel_std = float(meas_n["zupt_vel_std"])

    def _initialize_filter(self, t0: float) -> tuple[ExtendedKalmanFilter, float]:
        x0 = np.zeros(6, dtype=float)
        P0 = np.diag(
            [
                float(self.init_u["pos_std"]) ** 2,
                float(self.init_u["pos_std"]) ** 2,
                float(self.init_u["vel_std"]) ** 2,
                float(self.init_u["heading_std_rad"]) ** 2,
                1e-6,
                1e-6,
            ]
        )
        return ExtendedKalmanFilter(self.motion_model, x0, P0), t0

    def run(self, packets: List[VisualMeasurementPacket]) -> pd.DataFrame:
        if not packets:
            raise ValueError("Cannot run VisualOnlyEKF on empty visual measurement packets.")

        ekf, prev_t = self._initialize_filter(packets[0].t_sync_s)
        H_vel = np.zeros((1, 6), dtype=float)
        H_vel[0, 2] = 1.0

        records: List[Dict[str, Any]] = []
        for pkt in packets:
            dt = pkt.t_sync_s - prev_t
            omega_u = pkt.yaw_rate_rad_s if pkt.is_valid else 0.0
            if dt > 0.0:
                ekf.predict(np.array([0.0, omega_u], dtype=float), dt)
                prev_t = pkt.t_sync_s

            if pkt.is_valid:
                if pkt.is_stationary:
                    ekf.update(
                        z=np.array([0.0]),
                        h_x=np.array([ekf.x[2]]),
                        H=H_vel,
                        R=np.array([[self.zupt_vel_std**2]]),
                    )
                else:
                    # Inflate measurement noise inversely with RANSAC inlier confidence
                    conf = float(np.clip(pkt.inlier_ratio, 0.25, 1.0))
                    vel_std = self.vel_std_base / conf
                    ekf.update(
                        z=np.array([pkt.forward_speed]),
                        h_x=np.array([ekf.x[2]]),
                        H=H_vel,
                        R=np.array([[vel_std**2]]),
                    )

            records.append(
                {
                    "frame_idx": pkt.frame_idx,
                    "t_video_s": pkt.t_video_s,
                    "t_sync_s": pkt.t_sync_s,
                    "px_vis": float(ekf.x[0]),
                    "py_vis": float(ekf.x[1]),
                    "speed_vis": float(ekf.x[2]),
                    "heading_vis_rad": float(ekf.x[3]),
                    "yaw_rate_vis_rad_s": float(omega_u),
                    "inlier_ratio": pkt.inlier_ratio,
                    "median_flow_px": pkt.median_flow_px,
                    "visual_valid": pkt.is_valid,
                    "is_metric": pkt.is_metric,
                    "scale_status": pkt.scale_status,
                }
            )

        return pd.DataFrame(records)