import copy
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.ekf import FusionEKF, SensorOnlyEKF
from src.sensors import GPSProcessor, SensorTimelineBuilder
from src.vision import VisualMeasurementPacket


class GPSHoldoutEvaluator:
    """
    Non-circular evaluation: masks selected clean GPS windows during active driving,
    re-runs Track A (Sensor-Only) and Track C (Sensor+Vision Fusion), and benchmarks
    open-loop blackout drift strictly against held-out ground-truth GPS fixes.
    """

    def __init__(
        self,
        sensor_ekf_cfg: Dict[str, Any],
        fusion_ekf_cfg: Dict[str, Any],
        gps_max_speed_m_s: float,
        holdout_intervals_s: List[List[float]],
    ) -> None:
        self.sensor_cfg = copy.deepcopy(sensor_ekf_cfg)
        self.fusion_cfg = copy.deepcopy(fusion_ekf_cfg)
        self.gps_max_speed_m_s = float(gps_max_speed_m_s)
        self.holdout_intervals: List[Tuple[float, float]] = [
            (float(s), float(e)) for s, e in holdout_intervals_s
        ]

    @staticmethod
    def umeyama_align_2d(src_xy: np.ndarray, dst_xy: np.ndarray) -> Tuple[np.ndarray, float]:
        """Sim(2) similarity alignment (scale, rotation, translation) for Track B shape evaluation."""
        mu_src = src_xy.mean(axis=0)
        mu_dst = dst_xy.mean(axis=0)
        src_c = src_xy - mu_src
        dst_c = dst_xy - mu_dst

        var_src = float(np.mean(np.sum(src_c**2, axis=1)))
        cov = (dst_c.T @ src_c) / len(src_xy)
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(2)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[1, 1] = -1.0
        R = U @ S @ Vt
        scale = float(np.trace(np.diag(D) @ S) / max(var_src, 1e-8))
        aligned = scale * (src_c @ R.T) + mu_dst
        return aligned, scale

    def evaluate_holdout(
        self,
        imu_timeline: pd.DataFrame,
        gps_clean_df: pd.DataFrame,
        visual_packets: List[VisualMeasurementPacket],
    ) -> Dict[str, Any]:
        gps_proc = GPSProcessor(
            base_pos_std_m=float(self.fusion_cfg["measurement_noise"]["gps_pos_std_m"]),
            max_speed_m_s=self.gps_max_speed_m_s,
            holdout_intervals_s=self.holdout_intervals,
        )
        gps_masked = gps_proc.process(gps_clean_df)
        packets = SensorTimelineBuilder.build_packets(imu_timeline, gps_masked)

        track_a_ho = SensorOnlyEKF(self.sensor_cfg).run(packets)
        track_c_ho = FusionEKF(self.fusion_cfg).run(packets, visual_packets)

        gt_ho = gps_masked[gps_masked["is_valid"] & gps_masked["is_holdout"]].copy()
        gt_t = gt_ho["t_sec"].to_numpy(dtype=float)
        gt_x = gt_ho["gps_x_east_m"].to_numpy(dtype=float)
        gt_y = gt_ho["gps_y_north_m"].to_numpy(dtype=float)

        ta_dedup = track_a_ho.drop_duplicates(subset=["t_sec"], keep="last").sort_values("t_sec")
        tc_dedup = track_c_ho.drop_duplicates(subset=["t_sec"], keep="last").sort_values("t_sec")

        err_a = np.hypot(
            np.interp(gt_t, ta_dedup["t_sec"], ta_dedup["px_m"]) - gt_x,
            np.interp(gt_t, ta_dedup["t_sec"], ta_dedup["py_m"]) - gt_y,
        )
        err_c = np.hypot(
            np.interp(gt_t, tc_dedup["t_sec"], tc_dedup["px_fused_m"]) - gt_x,
            np.interp(gt_t, tc_dedup["t_sec"], tc_dedup["py_fused_m"]) - gt_y,
        )

        per_window: List[Dict[str, Any]] = []
        for idx, (s_t, e_t) in enumerate(self.holdout_intervals, start=1):
            mask = (gt_t >= s_t) & (gt_t <= e_t)
            if not np.any(mask):
                continue
            per_window.append(
                {
                    "window_id": idx,
                    "t_start_s": s_t,
                    "t_end_s": e_t,
                    "duration_s": round(e_t - s_t, 1),
                    "held_out_fixes": int(np.sum(mask)),
                    "track_a_rmse_m": round(float(np.sqrt(np.mean(err_a[mask] ** 2))), 2),
                    "track_c_rmse_m": round(float(np.sqrt(np.mean(err_c[mask] ** 2))), 2),
                    "track_a_max_drift_m": round(float(np.max(err_a[mask])), 2),
                    "track_c_max_drift_m": round(float(np.max(err_c[mask])), 2),
                }
            )

        ta_rmse = float(np.sqrt(np.mean(err_a**2)))
        tc_rmse = float(np.sqrt(np.mean(err_c**2)))
        rmse_reduction_pct = ((ta_rmse - tc_rmse) / max(ta_rmse, 1e-6)) * 100.0

        return {
            "summary": {
                "holdout_intervals_s": self.holdout_intervals,
                "total_held_out_fixes": int(len(gt_ho)),
                "track_a_holdout_rmse_m": round(ta_rmse, 2),
                "track_c_holdout_rmse_m": round(tc_rmse, 2),
                "track_a_holdout_mae_m": round(float(np.mean(err_a)), 2),
                "track_c_holdout_mae_m": round(float(np.mean(err_c)), 2),
                "track_a_holdout_max_drift_m": round(float(np.max(err_a)), 2),
                "track_c_holdout_max_drift_m": round(float(np.max(err_c)), 2),
                "rmse_reduction_pct": round(rmse_reduction_pct, 2),
            },
            "per_window": per_window,
            "track_a_holdout_df": ta_dedup,
            "track_c_holdout_df": tc_dedup,
            "gt_holdout_df": gt_ho,
        }