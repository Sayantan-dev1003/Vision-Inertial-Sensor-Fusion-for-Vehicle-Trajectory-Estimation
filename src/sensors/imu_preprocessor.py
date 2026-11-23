import numpy as np
import pandas as pd

from .accelerometer import AccelerometerProcessor
from .gyroscope import GyroscopeProcessor


class IMUPreprocessor:
    """Combines Accelerometer and Gyroscope streams onto a unified IMU timeline with ZUPT flags."""

    def __init__(
        self,
        acc_processor: AccelerometerProcessor | None = None,
        gyro_processor: GyroscopeProcessor | None = None,
        zupt_gyro_thresh_rad_s: float = 0.022,
        zupt_accel_thresh_m_s2: float = 0.45,
        zupt_accel_std_thresh_m_s2: float = 0.18,
        zupt_min_duration_s: float = 0.80,
    ) -> None:
        self.acc_processor = acc_processor or AccelerometerProcessor(smoothing_window=5)
        self.gyro_processor = gyro_processor or GyroscopeProcessor(smoothing_window=3)
        self.zupt_gyro_thresh_rad_s = float(zupt_gyro_thresh_rad_s)
        self.zupt_accel_thresh_m_s2 = float(zupt_accel_thresh_m_s2)
        self.zupt_accel_std_thresh_m_s2 = float(zupt_accel_std_thresh_m_s2)
        self.zupt_min_duration_s = float(zupt_min_duration_s)

    def build_imu_timeline(
        self,
        acc_df: pd.DataFrame,
        gyro_df: pd.DataFrame,
    ) -> pd.DataFrame:
        acc = self.acc_processor.process(acc_df)
        gyro = self.gyro_processor.process(gyro_df)

        acc_valid = acc.loc[acc["is_valid"], ["t_sec", "a_forward_clean_m_s2", "a_lateral_clean_m_s2"]].copy()
        gyro_valid = gyro.loc[gyro["is_valid"], ["t_sec", "yaw_rate_clean_rad_s"]].copy()

        t_imu = acc_valid["t_sec"].to_numpy(dtype=float)
        yaw_interp = np.interp(
            t_imu,
            gyro_valid["t_sec"].to_numpy(dtype=float),
            gyro_valid["yaw_rate_clean_rad_s"].to_numpy(dtype=float),
        )

        imu_df = pd.DataFrame(
            {
                "t_sec": t_imu,
                "a_forward_m_s2": acc_valid["a_forward_clean_m_s2"].to_numpy(dtype=float),
                "a_lateral_m_s2": acc_valid["a_lateral_clean_m_s2"].to_numpy(dtype=float),
                "yaw_rate_rad_s": yaw_interp,
            }
        )
        imu_df["dt_s"] = imu_df["t_sec"].diff().fillna(0.0)
        imu_df["is_stationary"] = self._detect_stationary_intervals(imu_df)
        return imu_df

    def _detect_stationary_intervals(self, imu_df: pd.DataFrame) -> np.ndarray:
        a_std = (
            imu_df["a_forward_m_s2"]
            .rolling(window=15, min_periods=1, center=True)
            .std()
            .fillna(0.0)
            .to_numpy()
        )
        w_abs = (
            imu_df["yaw_rate_rad_s"]
            .abs()
            .rolling(window=9, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )
        raw_stat = (
            (w_abs < self.zupt_gyro_thresh_rad_s)
            & (a_std < self.zupt_accel_std_thresh_m_s2)
            & (imu_df["a_forward_m_s2"].abs().to_numpy() < self.zupt_accel_thresh_m_s2)
        )

        t = imu_df["t_sec"].to_numpy()
        confirmed = np.zeros(len(imu_df), dtype=bool)
        start_idx: int | None = None

        for i, flag in enumerate(raw_stat):
            if flag and start_idx is None:
                start_idx = i
            elif not flag and start_idx is not None:
                if (t[i - 1] - t[start_idx]) >= self.zupt_min_duration_s:
                    confirmed[start_idx:i] = True
                start_idx = None

        if start_idx is not None and (t[-1] - t[start_idx]) >= self.zupt_min_duration_s:
            confirmed[start_idx:] = True

        return confirmed