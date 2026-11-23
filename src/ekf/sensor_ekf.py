from typing import Any, Dict, List
import numpy as np
import pandas as pd

from src.sensors.sensor_measurements import SensorMeasurementPacket
from .ekf_core import ExtendedKalmanFilter
from .motion_model import PlanarVehicleMotionModel


class SensorOnlyEKF:
    """Track A Estimator: Fuses high-rate IMU propagation with asynchronous GPS & ZUPT."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        init_u = cfg["initial_uncertainty"]
        proc_n = cfg["process_noise"]
        meas_n = cfg["measurement_noise"]
        gating = cfg["gating"]
        zupt_cfg = cfg["zupt"]

        self.motion_model = PlanarVehicleMotionModel(
            pos_noise_std=float(proc_n.get("pos_noise_std_m_s", 0.85)),
            accel_noise_std=float(proc_n["accel_noise_std_m_s2"]),
            gyro_noise_std=float(proc_n["gyro_noise_std_rad_s"]),
            acc_bias_walk_std=float(proc_n["acc_bias_walk_std"]),
            gyro_bias_walk_std=float(proc_n["gyro_bias_walk_std"]),
            acc_bias_tau_s=float(proc_n.get("acc_bias_tau_s", 30.0)),
            max_speed_m_s=float(proc_n.get("max_speed_m_s", 28.0)),
        )
        self.init_u = init_u
        self.gps_pos_std_m = float(meas_n["gps_pos_std_m"])
        self.gps_course_std_rad = float(meas_n.get("gps_course_std_rad", 0.30))
        self.zupt_vel_std_m_s = float(meas_n["zupt_vel_std_m_s"])
        self.zupt_yaw_std_rad_s = float(meas_n["zupt_yaw_rate_std_rad_s"])
        self.gps_gate = float(gating["gps_mahalanobis_threshold"])
        self.max_consec_rejects = int(gating.get("max_consecutive_rejects", 3))
        self.min_speed_heading = float(gating["min_speed_for_gps_heading_m_s"])
        self.zupt_enabled = bool(zupt_cfg["enabled"])

    def _initialize_filter(
        self, packets: List[SensorMeasurementPacket]
    ) -> tuple[ExtendedKalmanFilter, float]:
        first_gps = next(
            (p for p in packets if p.event_type == "gps" and p.gps_use_for_update),
            None,
        )
        init_px = first_gps.gps_x_east_m if first_gps is not None else 0.0
        init_py = first_gps.gps_y_north_m if first_gps is not None else 0.0

        init_psi = next(
            (
                p.gps_course_rad
                for p in packets
                if p.event_type == "gps"
                and np.isfinite(p.gps_course_rad)
                and np.isfinite(p.gps_speed_m_s)
                and p.gps_speed_m_s >= self.min_speed_heading
            ),
            0.0,
        )

        x0 = np.array([init_px, init_py, 0.0, init_psi, 0.0, 0.0], dtype=float)
        P0 = np.diag(
            [
                float(self.init_u["pos_std_m"]) ** 2,
                float(self.init_u["pos_std_m"]) ** 2,
                float(self.init_u["vel_std_m_s"]) ** 2,
                float(self.init_u["heading_std_rad"]) ** 2,
                float(self.init_u["acc_bias_std_m_s2"]) ** 2,
                float(self.init_u["gyro_bias_std_rad_s"]) ** 2,
            ]
        )
        return ExtendedKalmanFilter(self.motion_model, x0, P0), packets[0].t_sec

    def run(self, packets: List[SensorMeasurementPacket]) -> pd.DataFrame:
        if not packets:
            raise ValueError("Cannot run SensorOnlyEKF on empty measurement packet list.")

        ekf, prev_t = self._initialize_filter(packets)
        last_u = np.zeros(2, dtype=float)
        consec_gps_rejects = 0
        records: List[Dict[str, Any]] = []

        H_gps = np.zeros((2, 6), dtype=float)
        H_gps[0, 0] = 1.0
        H_gps[1, 1] = 1.0

        H_course = np.zeros((1, 6), dtype=float)
        H_course[0, 3] = 1.0

        H_zupt = np.zeros((1, 6), dtype=float)
        H_zupt[0, 2] = 1.0

        H_bg = np.zeros((1, 6), dtype=float)
        H_bg[0, 5] = 1.0

        for pkt in packets:
            dt = pkt.t_sec - prev_t
            if dt > 0.0:
                ekf.predict(last_u, dt)
                prev_t = pkt.t_sec

            gps_accepted = False
            zupt_applied = False

            if pkt.event_type == "imu":
                last_u = np.array([pkt.a_forward_m_s2, pkt.yaw_rate_rad_s], dtype=float)
                if self.zupt_enabled and pkt.is_stationary:
                    zupt_applied = ekf.update(
                        z=np.array([0.0]),
                        h_x=np.array([ekf.x[2]]),
                        H=H_zupt,
                        R=np.array([[self.zupt_vel_std_m_s**2]]),
                    )
                    ekf.update(
                        z=np.array([pkt.yaw_rate_rad_s]),
                        h_x=np.array([ekf.x[5]]),
                        H=H_bg,
                        R=np.array([[self.zupt_yaw_std_rad_s**2]]),
                    )

            elif pkt.event_type == "gps" and pkt.gps_use_for_update:
                pos_std = pkt.gps_pos_std_m if np.isfinite(pkt.gps_pos_std_m) else self.gps_pos_std_m
                force_reacquire = consec_gps_rejects >= self.max_consec_rejects
                eff_std = pos_std * (1.45 if force_reacquire else 1.0)
                R_gps = np.diag([eff_std**2, eff_std**2])

                gps_accepted = ekf.update(
                    z=np.array([pkt.gps_x_east_m, pkt.gps_y_north_m]),
                    h_x=ekf.x[:2],
                    H=H_gps,
                    R=R_gps,
                    mahalanobis_gate=None if force_reacquire else self.gps_gate,
                )

                if gps_accepted:
                    consec_gps_rejects = 0
                    if (
                        np.isfinite(pkt.gps_course_rad)
                        and np.isfinite(pkt.gps_speed_m_s)
                        and pkt.gps_speed_m_s >= self.min_speed_heading
                    ):
                        ekf.update(
                            z=np.array([pkt.gps_course_rad]),
                            h_x=np.array([ekf.x[3]]),
                            H=H_course,
                            R=np.array([[self.gps_course_std_rad**2]]),
                            angle_indices=(0,),
                        )
                else:
                    consec_gps_rejects += 1

            yaw_rate_est = float(last_u[1] - ekf.x[5])
            records.append(
                {
                    "t_sec": pkt.t_sec,
                    "event_type": pkt.event_type,
                    "px_m": float(ekf.x[0]),
                    "py_m": float(ekf.x[1]),
                    "speed_m_s": float(ekf.x[2]),
                    "heading_rad": float(ekf.x[3]),
                    "yaw_rate_rad_s": yaw_rate_est,
                    "acc_bias_m_s2": float(ekf.x[4]),
                    "gyro_bias_rad_s": float(ekf.x[5]),
                    "std_px_m": float(np.sqrt(max(0.0, ekf.P[0, 0]))),
                    "std_py_m": float(np.sqrt(max(0.0, ekf.P[1, 1]))),
                    "std_speed_m_s": float(np.sqrt(max(0.0, ekf.P[2, 2]))),
                    "std_heading_rad": float(np.sqrt(max(0.0, ekf.P[3, 3]))),
                    "gps_accepted": gps_accepted,
                    "zupt_applied": zupt_applied,
                }
            )

        return pd.DataFrame(records)