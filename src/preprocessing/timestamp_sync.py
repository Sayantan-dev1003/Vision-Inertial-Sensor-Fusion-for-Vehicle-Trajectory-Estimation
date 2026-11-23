from typing import Dict
import numpy as np
import pandas as pd


class TimestampSynchronizer:
    """Aligns asynchronous accr, gyro, and gps streams onto video frame timestamps."""

    DEFAULT_STALENESS_S: Dict[str, float] = {
        "accr": 0.20,
        "gyro": 0.35,
        "gps": 2.00,
    }

    def __init__(
        self,
        offset_s: float = 0.0,
        max_staleness_s: Dict[str, float] | None = None,
    ) -> None:
        self.offset_s = float(offset_s)
        self.max_staleness_s = dict(self.DEFAULT_STALENESS_S)
        if max_staleness_s:
            self.max_staleness_s.update(max_staleness_s)

    def align_to_video_frames(
        self,
        video_df: pd.DataFrame,
        streams: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        master = video_df.copy()
        master["t_sync_s"] = master["t_video_s"] + self.offset_s
        master = master.sort_values("t_sync_s").reset_index(drop=True)

        # 1. Gyroscope -> yaw_rate_rad_s
        gyro_valid = streams["gyro"].loc[streams["gyro"]["is_valid"]].sort_values("t_sec")
        gyro_sub = gyro_valid[["t_sec", "yaw_rate_rad_s"]].rename(columns={"t_sec": "t_gyro_s"})
        master = pd.merge_asof(
            master,
            gyro_sub,
            left_on="t_sync_s",
            right_on="t_gyro_s",
            direction="nearest",
            tolerance=self.max_staleness_s["gyro"],
        )
        master["gyro_valid"] = master["yaw_rate_rad_s"].notna()

        # 2. Accelerometer (gravity-compensated) -> a_forward_m_s2, a_lateral_m_s2
        acc_valid = streams["accr"].loc[streams["accr"]["is_valid"]].sort_values("t_sec")
        acc_sub = acc_valid[["t_sec", "a_forward_m_s2", "a_lateral_m_s2", "a_vertical_m_s2"]].rename(columns={"t_sec": "t_acc_s"})
        master = pd.merge_asof(
            master,
            acc_sub,
            left_on="t_sync_s",
            right_on="t_acc_s",
            direction="nearest",
            tolerance=self.max_staleness_s["accr"],
        )
        master["acc_valid"] = master["a_forward_m_s2"].notna()

        # 3. GPS -> local ENU coordinates
        gps_valid = streams["gps"].loc[streams["gps"]["is_valid"]].sort_values("t_sec")
        gps_sub = gps_valid[
            [
                "t_sec",
                "gps_lat",
                "gps_lon",
                "gps_alt",
                "gps_x_east_m",
                "gps_y_north_m",
                "gps_z_up_m",
                "gps_course_rad",
                "gps_speed_enu_m_s",
            ]
        ].rename(columns={"t_sec": "t_gps_s"})
        master = pd.merge_asof(
            master,
            gps_sub,
            left_on="t_sync_s",
            right_on="t_gps_s",
            direction="nearest",
            tolerance=self.max_staleness_s["gps"],
        )
        master["gps_valid"] = master["gps_x_east_m"].notna()

        return master

    @staticmethod
    def estimate_video_sensor_offset(
        t_video_s: np.ndarray,
        visual_rot_rate: np.ndarray,
        t_gyro_s: np.ndarray,
        gyro_yaw_rate: np.ndarray,
        search_range_s: tuple[float, float] = (-30.0, 30.0),
        step_s: float = 0.05,
    ) -> tuple[float, float]:
        """Signed cross-correlate CCW visual yaw rate with CCW gyro yaw rate."""
        dt_grid = 0.1
        t_min = max(float(t_video_s.min()), float(t_gyro_s.min()))
        t_max = min(float(t_video_s.max()), float(t_gyro_s.max()))
        grid = np.arange(t_min, t_max, dt_grid)

        g_interp = np.interp(grid, t_gyro_s, gyro_yaw_rate)
        g_norm = (g_interp - g_interp.mean()) / (g_interp.std() + 1e-8)

        best_offset = 0.0
        best_corr = -1.0
        for offset in np.arange(search_range_s[0], search_range_s[1] + step_s, step_s):
            v_interp = np.interp(
                grid, t_video_s + offset, visual_rot_rate, left=0.0, right=0.0
            )
            v_std = float(v_interp.std())
            if v_std < 1e-8:
                continue
            v_norm = (v_interp - v_interp.mean()) / v_std
            corr = float(np.mean(g_norm * v_norm))
            if corr > best_corr:
                best_corr = corr
                best_offset = float(offset)

        return round(best_offset, 3), round(best_corr, 4)