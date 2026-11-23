from collections import deque
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


def _wrap_angle_rad(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class FusionEKF:
    """Track C Estimator: Asynchronous EKF + RTS Backward Smoother fusing IMU, GPS, and Optical Flow VO."""

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
        self.gps_course_gate = float(gating.get("gps_course_nis_threshold", 4.0))
        self.max_consec_rejects = int(gating["max_consecutive_rejects"])
        self.reacquire_max_jump_m = float(gating.get("reacquire_max_jump_m", 260.0))
        self.reacquire_std_mult = float(gating.get("reacquire_pos_std_multiplier", 1.5))
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
                and p.gps_use_for_update
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

    @staticmethod
    def _rts_smooth(
        x_filt: np.ndarray,
        P_filt: np.ndarray,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        F_list: np.ndarray,
        dt_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rauch-Tung-Striebel (RTS) backward pass to eliminate post-outage position snaps."""
        n_steps, state_dim = x_filt.shape
        x_smooth = x_filt.copy()
        P_smooth = P_filt.copy()
        eye_reg = np.eye(state_dim, dtype=float) * 1e-9

        for k in range(n_steps - 2, -1, -1):
            if dt_arr[k + 1] <= 0.0:
                x_smooth[k] = x_smooth[k + 1]
                P_smooth[k] = P_smooth[k + 1]
                continue

            P_pred_next = P_pred[k + 1] + eye_reg
            try:
                # C_k = P_{k|k} @ F_{k+1}^T @ inv(P_{k+1|k})
                C_k = np.linalg.solve(P_pred_next.T, (P_filt[k] @ F_list[k + 1].T).T).T
            except np.linalg.LinAlgError:
                C_k = P_filt[k] @ F_list[k + 1].T @ np.linalg.pinv(P_pred_next)

            diff = x_smooth[k + 1] - x_pred[k + 1]
            diff[3] = _wrap_angle_rad(diff[3])

            x_smooth[k] = x_filt[k] + C_k @ diff
            x_smooth[k, 3] = _wrap_angle_rad(x_smooth[k, 3])
            x_smooth[k, 2] = float(np.clip(x_smooth[k, 2], 0.0, 28.0))

            P_smooth[k] = P_filt[k] + C_k @ (P_smooth[k + 1] - P_pred[k + 1]) @ C_k.T
            P_smooth[k] = 0.5 * (P_smooth[k] + P_smooth[k].T)

        return x_smooth, P_smooth

    def run(
        self,
        sensor_packets: List[SensorMeasurementPacket],
        visual_packets: List[VisualMeasurementPacket],
    ) -> pd.DataFrame:
        if not sensor_packets:
            raise ValueError("Cannot run FusionEKF without sensor packets.")

        events = self.build_event_queue(sensor_packets, visual_packets)
        ekf = self._initialize_filter(sensor_packets, events[0].t_sec)
        n_events = len(events)

        prev_t = events[0].t_sec
        last_u = np.zeros(2, dtype=float)
        last_gps_t = -999.0
        consec_gps_rejects = 0
        last_rejected_gps: np.ndarray | None = None
        visual_scale = self.init_vis_scale
        recent_scales: deque[float] = deque(maxlen=40)
        recent_scales.append(self.init_vis_scale)

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

        # Pre-allocate arrays for forward EKF + RTS backward smoother
        x_filt_arr = np.zeros((n_events, 6), dtype=float)
        P_filt_arr = np.zeros((n_events, 6, 6), dtype=float)
        x_pred_arr = np.zeros((n_events, 6), dtype=float)
        P_pred_arr = np.zeros((n_events, 6, 6), dtype=float)
        F_arr = np.broadcast_to(np.eye(6, dtype=float), (n_events, 6, 6)).copy()
        dt_arr = np.zeros(n_events, dtype=float)
        raw_yaw_arr = np.zeros(n_events, dtype=float)

        meta_records: List[Dict[str, Any]] = []

        for idx, ev in enumerate(events):
            dt = ev.t_sec - prev_t
            dt_arr[idx] = max(0.0, dt)
            if dt > 0.0:
                F_step = self.motion_model.jacobian_F(ekf.x, dt)
                F_arr[idx] = F_step
                ekf.predict(last_u, dt)
                prev_t = ev.t_sec

            x_pred_arr[idx] = ekf.x.copy()
            P_pred_arr[idx] = ekf.P.copy()

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
                        recent_scales.append(visual_scale)

                    # 3. During under-flyover GPS outages, use robust median scale to bound drift
                    if gps_age > self.gps_outage_trigger_s:
                        stable_scale = float(np.median(recent_scales)) if recent_scales else visual_scale
                        v_vis_metric = (
                            0.0
                            if vp.is_stationary
                            else float(np.clip(stable_scale * vp.forward_speed, 0.0, 20.0))
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
                    z_gps = np.array([pkt.gps_x_east_m, pkt.gps_y_north_m], dtype=float)
                    innovation_dist_m = float(np.hypot(z_gps[0] - ekf.x[0], z_gps[1] - ekf.x[1]))

                    # Require 2 consecutive spatially consistent fixes (<35m apart) before forced re-acquisition
                    consistent_pair = (
                        last_rejected_gps is not None
                        and float(np.hypot(z_gps[0] - last_rejected_gps[0], z_gps[1] - last_rejected_gps[1])) <= 35.0
                    )
                    can_force_reacquire = (
                        consec_gps_rejects >= self.max_consec_rejects
                        and innovation_dist_m <= self.reacquire_max_jump_m
                        and consistent_pair
                    )
                    eff_std = pos_std * (self.reacquire_std_mult if can_force_reacquire else 1.0)
                    R_gps = np.diag([eff_std**2, eff_std**2])

                    gps_accepted = ekf.update(
                        z=z_gps,
                        h_x=ekf.x[:2],
                        H=H_gps,
                        R=R_gps,
                        mahalanobis_gate=None if can_force_reacquire else self.gps_gate,
                    )

                    if gps_accepted:
                        consec_gps_rejects = 0
                        last_rejected_gps = None
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
                                mahalanobis_gate=self.gps_course_gate,
                                angle_indices=(0,),
                            )
                    else:
                        consec_gps_rejects += 1
                        last_rejected_gps = z_gps.copy()

            x_filt_arr[idx] = ekf.x.copy()
            P_filt_arr[idx] = ekf.P.copy()
            raw_yaw_arr[idx] = float(last_u[1])

            meta_records.append(
                {
                    "t_sec": ev.t_sec,
                    "event_type": event_label,
                    "visual_scale_est": float(visual_scale),
                    "gps_accepted": gps_accepted,
                    "vis_yaw_accepted": vis_yaw_accepted,
                    "vis_speed_applied": vis_speed_applied,
                    "zupt_applied": zupt_applied,
                }
            )

        # Run RTS backward smoother over full trajectory to smoothly bridge under-flyover GPS outages
        x_smooth, P_smooth = self._rts_smooth(
            x_filt=x_filt_arr,
            P_filt=P_filt_arr,
            x_pred=x_pred_arr,
            P_pred=P_pred_arr,
            F_list=F_arr,
            dt_arr=dt_arr,
        )

        df_out = pd.DataFrame(meta_records)
        df_out["px_fused_m"] = x_smooth[:, 0]
        df_out["py_fused_m"] = x_smooth[:, 1]
        df_out["speed_fused_m_s"] = x_smooth[:, 2]
        df_out["heading_fused_rad"] = x_smooth[:, 3]
        df_out["yaw_rate_fused_rad_s"] = raw_yaw_arr - x_smooth[:, 5]
        df_out["acc_bias_fused_m_s2"] = x_smooth[:, 4]
        df_out["gyro_bias_fused_rad_s"] = x_smooth[:, 5]
        df_out["std_px_fused_m"] = np.sqrt(np.maximum(0.0, P_smooth[:, 0, 0]))
        df_out["std_py_fused_m"] = np.sqrt(np.maximum(0.0, P_smooth[:, 1, 1]))
        df_out["std_speed_fused_m_s"] = np.sqrt(np.maximum(0.0, P_smooth[:, 2, 2]))
        df_out["std_heading_fused_rad"] = np.sqrt(np.maximum(0.0, P_smooth[:, 3, 3]))

        # Reorder columns cleanly to match existing schema
        cols = [
            "t_sec",
            "event_type",
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
            "gps_accepted",
            "vis_yaw_accepted",
            "vis_speed_applied",
            "zupt_applied",
        ]
        return df_out[cols]