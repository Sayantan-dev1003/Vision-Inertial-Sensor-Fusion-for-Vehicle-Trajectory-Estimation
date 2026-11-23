from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from src.sensors.sensor_measurements import SensorMeasurementPacket
from src.vision.visual_measurements import VisualMeasurementPacket
from .ekf_core import ExtendedKalmanFilter
from .motion_model import PlanarVehicleMotionModel


@dataclass(frozen=True)
class FusionEvent:
    """Unified chronological event wrapper for IMU, Visual Odometry, and GPS updates."""

    t_sec: float
    priority: int  # 0 = imu, 1 = visual, 2 = gps
    sensor_pkt: SensorMeasurementPacket | None = None
    visual_pkt: VisualMeasurementPacket | None = None


class FusionEKF:
    """Track C Estimator: Asynchronous EKF fusing IMU prediction, GPS fixes, and Optical Flow VO."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        init_u = cfg["initial_uncertainty"]
        proc_n = cfg["process_noise"]
        meas_n = cfg["measurement_noise"]
        gating = cfg["gating"]
        scale_cfg = cfg["visual_scale_adaptation"]
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
        self.gps_course_std_rad = float(meas_n["gps_course_std_rad"])
        self.zupt_vel_std_m_s = float(meas_n["zupt_vel_std_m_s"])
        self.vis_zupt_vel_std = float(meas_n["visual_zupt_vel_std_m_s"])
        self.vis_yaw_std_base = float(meas_n["visual_yaw_rate_std_base"])
        self.vis_outage_speed_std = float(meas_n["visual_outage_speed_std_m_s"])

        self.gps_gate = float(gating["gps_mahalanobis_threshold"])
        self.max_consec_rejects = int(gating["max_consecutive_rejects"])
        self.min_speed_heading = float(gating["min_speed_for_gps_heading_m_s"])
        self.vis_min_inlier = float(gating["visual_min_inlier_ratio"])
        self.vis_yaw_nis_gate = float(gating["visual_yaw_nis_threshold"])
        self.vis_dead_stop_flow_px = float(gating["visual_dead_stop_flow_px"])

        self.init_vis_scale = float(scale_cfg["initial_scale_m_per_unit"])
        self.scale_ema_alpha = float(scale_cfg["ema_alpha"])
        self.min_vis_scale = float(scale_cfg["min_scale"])
        self.max_vis_scale = float(scale_cfg["max_scale"])
        self.gps_fresh_window_s = float(scale_cfg["gps_fresh_window_s"])
        self.gps_outage_trigger_s = float(scale_cfg["gps_outage_trigger_s"])

        self.zupt_enabled = bool(zupt_cfg["enabled"])

    @staticmethod
    def build_event_queue(
        sensor_packets: List[SensorMeasurementPacket],
        visual_packets: List[VisualMeasurementPacket],
    ) -> List[FusionEvent]:
        events: List[FusionEvent] = []
        for sp in sensor_packets:
            prio = 0 if sp.event_type == "imu" else 2
            events.append(FusionEvent(t_sec=sp.t_sec, priority=prio, sensor_pkt=sp))
        for vp in visual_packets:
            if vp.is_valid:
                events.append(FusionEvent(t_sec=vp.t_sync_s, priority=1, visual_pkt=vp))

        events.sort(key=lambda e: (e.t_sec, e.priority))
        return events

    def _initialize_filter(
        self, sensor_packets: List[SensorMeasurementPacket], t0: float
    ) -> ExtendedKalmanFilter:
        first_gps = next(
            (p for p in sensor_packets if p.event_type == "gps" and p.gps_use_for_update),
            None,
        )
        init_px = first_gps.gps_x_east_m if first_gps is not None else 0.0
        init_py = first_gps.gps_y_north_m if first_gps is not None else 0.0

        init_psi = next(
            (
                p.gps_course_rad
                for p in sensor_packets
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
        return ExtendedKalmanFilter(self.motion_model, x0, P0)

    def run(
        self,
        sensor_packets: List[SensorMeasurementPacket],
        visual_packets: List[VisualMeasurementPacket],
    ) -> pd.DataFrame:
        if not sensor_packets:
            raise ValueError("Cannot run FusionEKF without sensor packets.")

        events = self.build_event_queue(sensor_packets, visual_packets)
        ekf = self._initialize_filter(sensor_packets, events[0].t_sec)

        prev_t = events[0].t_sec
        last_u = np.zeros(2, dtype=float)
        last_gps_t = -999.0
        consec_gps_rejects = 0
        visual_scale = self.init_vis_scale

        H_gps = np.zeros((2, 6), dtype=float)
        H_gps[0, 0] = 1.0
        H_gps[1, 1] = 1.0

        H_course = np.zeros((1, 6), dtype=float)
        H_course[0, 3] = 1.0

        H_vel = np.zeros((1, 6), dtype=float)
        H_vel[0, 2] = 1.0

        # Measurement model for visual yaw-rate: z_vis = omega_m - b_g  =>  dh/db_g = -1
        H_vis_bg = np.zeros((1, 6), dtype=float)
        H_vis_bg[0, 5] = -1.0

        records: List[Dict[str, Any]] = []

        for ev in events:
            dt = ev.t_sec - prev_t
            if dt > 0.0:
                ekf.predict(last_u, dt)
                prev_t = ev.t_sec

            gps_accepted = False
            vis_yaw_accepted = False
            vis_speed_applied = False
            zupt_applied = False
            event_label = "imu"

            if ev.priority == 0 and ev.sensor_pkt is not None:
                pkt = ev.sensor_pkt
                event_label = "imu"
                last_u = np.array([pkt.a_forward_m_s2, pkt.yaw_rate_rad_s], dtype=float)
                if self.zupt_enabled and pkt.is_stationary:
                    zupt_applied = ekf.update(
                        z=np.array([0.0]),
                        h_x=np.array([ekf.x[2]]),
                        H=H_vel,
                        R=np.array([[self.zupt_vel_std_m_s**2]]),
                    )

            elif ev.priority == 1 and ev.visual_pkt is not None:
                vp = ev.visual_pkt
                event_label = "visual"

                if vp.median_flow_px < self.vis_dead_stop_flow_px and vp.inlier_ratio >= 0.85:
                    zupt_applied = ekf.update(
                        z=np.array([0.0]),
                        h_x=np.array([ekf.x[2]]),
                        H=H_vel,
                        R=np.array([[self.vis_zupt_vel_std**2]]),
                    )
                elif vp.inlier_ratio >= self.vis_min_inlier:
                    # 1. Visual yaw-rate update to observe gyro yaw-rate bias b_g
                    w_std = self.vis_yaw_std_base / max(0.5, vp.inlier_ratio)
                    vis_yaw_accepted = ekf.update(
                        z=np.array([vp.yaw_rate_rad_s]),
                        h_x=np.array([last_u[1] - ekf.x[5]]),
                        H=H_vis_bg,
                        R=np.array([[w_std**2]]),
                        mahalanobis_gate=self.vis_yaw_nis_gate,
                    )

                    # 2. Online visual scale calibration when GPS is fresh and vehicle is moving
                    gps_age = ev.t_sec - last_gps_t
                    if gps_age <= self.gps_fresh_window_s and ekf.x[2] > 3.5 and vp.forward_speed > 0.30:
                        inst_scale = float(
                            np.clip(ekf.x[2] / vp.forward_speed, self.min_vis_scale, self.max_vis_scale)
                        )
                        visual_scale = (1.0 - self.scale_ema_alpha) * visual_scale + self.scale_ema_alpha * inst_scale

                    # 3. During GPS outages, fuse scale-calibrated visual speed to bound drift
                    if gps_age > self.gps_outage_trigger_s:
                        v_vis_metric = (
                            0.0
                            if vp.is_stationary
                            else float(np.clip(visual_scale * vp.forward_speed, 0.0, 20.0))
                        )
                        vis_speed_applied = ekf.update(
                            z=np.array([v_vis_metric]),
                            h_x=np.array([ekf.x[2]]),
                            H=H_vel,
                            R=np.array([[self.vis_outage_speed_std**2]]),
                        )

            elif ev.priority == 2 and ev.sensor_pkt is not None:
                pkt = ev.sensor_pkt
                event_label = "gps"
                if pkt.gps_use_for_update:
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
                        last_gps_t = ev.t_sec
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
                    "t_sec": ev.t_sec,
                    "event_type": event_label,
                    "px_fused_m": float(ekf.x[0]),
                    "py_fused_m": float(ekf.x[1]),
                    "speed_fused_m_s": float(ekf.x[2]),
                    "heading_fused_rad": float(ekf.x[3]),
                    "yaw_rate_fused_rad_s": yaw_rate_est,
                    "acc_bias_fused_m_s2": float(ekf.x[4]),
                    "gyro_bias_fused_rad_s": float(ekf.x[5]),
                    "visual_scale_est": float(visual_scale),
                    "std_px_fused_m": float(np.sqrt(max(0.0, ekf.P[0, 0]))),
                    "std_py_fused_m": float(np.sqrt(max(0.0, ekf.P[1, 1]))),
                    "std_speed_fused_m_s": float(np.sqrt(max(0.0, ekf.P[2, 2]))),
                    "std_heading_fused_rad": float(np.sqrt(max(0.0, ekf.P[3, 3]))),
                    "gps_accepted": gps_accepted,
                    "vis_yaw_accepted": vis_yaw_accepted,
                    "vis_speed_applied": vis_speed_applied,
                    "zupt_applied": zupt_applied,
                }
            )

        return pd.DataFrame(records)