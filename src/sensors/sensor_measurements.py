from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SensorMeasurementPacket:
    """Chronological event packet consumed by SensorOnlyEKF and FusionEKF."""

    t_sec: float
    event_type: str  # "imu" | "gps"
    a_forward_m_s2: float = 0.0
    yaw_rate_rad_s: float = 0.0
    is_stationary: bool = False
    gps_x_east_m: float = np.nan
    gps_y_north_m: float = np.nan
    gps_pos_std_m: float = np.nan
    gps_speed_m_s: float = np.nan
    gps_course_rad: float = np.nan
    gps_use_for_update: bool = False


class SensorTimelineBuilder:
    """Merges high-rate IMU propagation steps and asynchronous GPS updates in time order."""

    @staticmethod
    def build_packets(
        imu_df: pd.DataFrame,
        gps_df: pd.DataFrame,
    ) -> List[SensorMeasurementPacket]:
        packets: List[SensorMeasurementPacket] = []

        for row in imu_df.itertuples(index=False):
            packets.append(
                SensorMeasurementPacket(
                    t_sec=float(row.t_sec),
                    event_type="imu",
                    a_forward_m_s2=float(row.a_forward_m_s2),
                    yaw_rate_rad_s=float(row.yaw_rate_rad_s),
                    is_stationary=bool(row.is_stationary),
                )
            )

        valid_gps = gps_df.loc[gps_df["is_valid"]].copy()
        for row in valid_gps.itertuples(index=False):
            packets.append(
                SensorMeasurementPacket(
                    t_sec=float(row.t_sec),
                    event_type="gps",
                    gps_x_east_m=float(row.gps_x_east_m),
                    gps_y_north_m=float(row.gps_y_north_m),
                    gps_pos_std_m=float(row.gps_pos_std_m),
                    gps_speed_m_s=float(row.gps_speed_enu_m_s)
                    if np.isfinite(row.gps_speed_enu_m_s)
                    else np.nan,
                    gps_course_rad=float(row.gps_course_rad)
                    if np.isfinite(row.gps_course_rad)
                    else np.nan,
                    gps_use_for_update=bool(row.gps_use_for_update),
                )
            )

        # Sort chronologically; process IMU propagation before GPS update on exact timestamp ties
        packets.sort(key=lambda p: (p.t_sec, 0 if p.event_type == "imu" else 1))
        return packets