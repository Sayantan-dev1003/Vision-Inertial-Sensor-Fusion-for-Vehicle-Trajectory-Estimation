from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IMUCalibrationResult:
    gravity_bias_vector_sensor_m_s2: list[float]
    gravity_norm_m_s2: float
    up_unit_sensor: list[float]
    forward_unit_sensor: list[float]
    lateral_unit_sensor: list[float]
    gyro_bias_sensor_rad_s: list[float]
    yaw_sign: float
    stationary_gyro_samples: int
    stationary_acc_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IMUCalibrator:
    """Calibrates raw Accelerometer (with gravity) and Gyroscope into the 2D vehicle frame."""

    def __init__(
        self,
        forward_axis_hint: Sequence[float],
        up_axis_hint: Sequence[float],
        yaw_sign: float = -1.0,
        stationary_init_seconds: float = 5.0,
        gyro_stationary_thresh_rad_s: float = 0.03,
    ) -> None:
        self.forward_axis_hint = np.asarray(forward_axis_hint, dtype=float)
        self.up_axis_hint = np.asarray(up_axis_hint, dtype=float)
        self.yaw_sign = float(yaw_sign)
        self.stationary_init_seconds = float(stationary_init_seconds)
        self.gyro_stationary_thresh_rad_s = float(gyro_stationary_thresh_rad_s)
        self.result: IMUCalibrationResult | None = None

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            raise ValueError("Cannot normalize near-zero axis vector.")
        return vec / norm

    def fit(self, acc_df: pd.DataFrame, gyro_df: pd.DataFrame) -> IMUCalibrationResult:
        # 1. Estimate Gyroscope zero-rate bias during stationary startup window
        g_init = gyro_df[(gyro_df["t_sec"] <= self.stationary_init_seconds) & gyro_df["is_valid"]]
        g_vecs = g_init[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
        if len(g_vecs) > 0:
            low_motion = np.linalg.norm(g_vecs, axis=1) < self.gyro_stationary_thresh_rad_s
            if int(np.sum(low_motion)) >= 5:
                gyro_bias = np.mean(g_vecs[low_motion], axis=0)
                gyro_stat_count = int(np.sum(low_motion))
            else:
                gyro_bias = np.median(g_vecs, axis=0)
                gyro_stat_count = int(len(g_vecs))
        else:
            gyro_bias = np.zeros(3, dtype=float)
            gyro_stat_count = 0

        # 2. Estimate combined Gravity + Static Bias vector directly from raw Accelerometer
        a_init = acc_df[(acc_df["t_sec"] <= self.stationary_init_seconds) & acc_df["is_valid"]]
        a_vecs = a_init[["acc_x", "acc_y", "acc_z"]].to_numpy()
        if len(a_vecs) >= 5:
            g_bias_vec = np.median(a_vecs, axis=0)
            up_unit = self._normalize(g_bias_vec)
            acc_stat_count = int(len(a_vecs))
        else:
            up_unit = self._normalize(self.up_axis_hint)
            g_bias_vec = 9.80665 * up_unit
            acc_stat_count = int(len(a_vecs))

        # 3. Build orthonormal vehicle basis: (forward_unit, lateral_unit, up_unit)
        fwd_proj = self.forward_axis_hint - np.dot(self.forward_axis_hint, up_unit) * up_unit
        forward_unit = self._normalize(fwd_proj)
        lateral_unit = self._normalize(np.cross(up_unit, forward_unit))

        self.result = IMUCalibrationResult(
            gravity_bias_vector_sensor_m_s2=np.round(g_bias_vec, 6).tolist(),
            gravity_norm_m_s2=round(float(np.linalg.norm(g_bias_vec)), 4),
            up_unit_sensor=np.round(up_unit, 6).tolist(),
            forward_unit_sensor=np.round(forward_unit, 6).tolist(),
            lateral_unit_sensor=np.round(lateral_unit, 6).tolist(),
            gyro_bias_sensor_rad_s=np.round(gyro_bias, 6).tolist(),
            yaw_sign=self.yaw_sign,
            stationary_gyro_samples=gyro_stat_count,
            stationary_acc_samples=acc_stat_count,
        )
        return self.result

    def transform(
        self,
        acc_df: pd.DataFrame,
        gyro_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.result is None:
            self.fit(acc_df, gyro_df)
        assert self.result is not None

        acc = acc_df.copy()
        gyro = gyro_df.copy()

        g_bias_vec = np.asarray(self.result.gravity_bias_vector_sensor_m_s2, dtype=float)
        up_unit = np.asarray(self.result.up_unit_sensor, dtype=float)
        forward_unit = np.asarray(self.result.forward_unit_sensor, dtype=float)
        lateral_unit = np.asarray(self.result.lateral_unit_sensor, dtype=float)
        gyro_bias = np.asarray(self.result.gyro_bias_sensor_rad_s, dtype=float)

        # Gravity + static offset removed directly from raw accelerometer
        a_dynamic = acc[["acc_x", "acc_y", "acc_z"]].to_numpy() - g_bias_vec
        acc["a_forward_m_s2"] = a_dynamic @ forward_unit
        acc["a_lateral_m_s2"] = a_dynamic @ lateral_unit
        acc["a_vertical_m_s2"] = a_dynamic @ up_unit

        # Bias-corrected gyroscope projected onto vertical axis (CCW positive yaw rate)
        g_corrected = gyro[["gyro_x", "gyro_y", "gyro_z"]].to_numpy() - gyro_bias
        gyro["yaw_rate_rad_s"] = self.result.yaw_sign * (g_corrected @ up_unit)

        return acc, gyro

    def fit_transform(
        self,
        acc_df: pd.DataFrame,
        gyro_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, IMUCalibrationResult]:
        res = self.fit(acc_df, gyro_df)
        acc_out, gyro_out = self.transform(acc_df, gyro_df)
        return acc_out, gyro_out, res