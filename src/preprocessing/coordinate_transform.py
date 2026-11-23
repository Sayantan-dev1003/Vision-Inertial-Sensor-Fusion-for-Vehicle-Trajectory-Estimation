import numpy as np
import pandas as pd

from .data_schema import ENUOrigin


class CoordinateTransformer:
    """Handles WGS84 geodetic <-> local East-North-Up (ENU) metric projections and angle wrapping."""

    WGS84_A: float = 6378137.0
    WGS84_F: float = 1.0 / 298.257223563
    WGS84_E2: float = WGS84_F * (2.0 - WGS84_F)

    def __init__(self, origin: ENUOrigin | None = None) -> None:
        self.origin = origin

    @staticmethod
    def wrap_angle(angle_rad: np.ndarray | float) -> np.ndarray | float:
        """Wrap angle(s) into [-pi, pi] radians (CCW from East convention)."""
        return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi

    def set_origin_from_gps(self, gps_df: pd.DataFrame) -> ENUOrigin:
        valid_mask = gps_df["is_valid"].to_numpy()
        if not np.any(valid_mask):
            raise ValueError("No valid GPS fixes available to define ENU origin.")

        first_valid = gps_df.loc[valid_mask].iloc[0]
        self.origin = ENUOrigin(
            lat0_deg=float(first_valid["gps_lat"]),
            lon0_deg=float(first_valid["gps_lon"]),
            alt0_m=float(first_valid["gps_alt"]),
        )
        return self.origin

    def geodetic_to_enu(
        self,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        alt_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.origin is None:
            raise RuntimeError("ENU origin must be initialized before projection.")

        lat0 = np.deg2rad(self.origin.lat0_deg)
        sin_lat0 = np.sin(lat0)
        cos_lat0 = np.cos(lat0)

        denom = np.sqrt(1.0 - self.WGS84_E2 * sin_lat0**2)
        r_n = self.WGS84_A / denom
        r_m = self.WGS84_A * (1.0 - self.WGS84_E2) / (denom**3)

        dlat = np.deg2rad(lat_deg - self.origin.lat0_deg)
        dlon = np.deg2rad(lon_deg - self.origin.lon0_deg)

        east_m = (r_n + self.origin.alt0_m) * cos_lat0 * dlon
        north_m = (r_m + self.origin.alt0_m) * dlat
        up_m = alt_m - self.origin.alt0_m
        return east_m, north_m, up_m

    def project_gps_dataframe(self, gps_df: pd.DataFrame) -> tuple[pd.DataFrame, ENUOrigin]:
        if self.origin is None:
            self.set_origin_from_gps(gps_df)
        assert self.origin is not None

        df = gps_df.copy()
        east, north, up = self.geodetic_to_enu(
            df["gps_lat"].to_numpy(),
            df["gps_lon"].to_numpy(),
            df["gps_alt"].to_numpy(),
        )
        df["gps_x_east_m"] = east
        df["gps_y_north_m"] = north
        df["gps_z_up_m"] = up

        # Course-over-ground heading (CCW from East) & ENU step speed
        df["gps_course_rad"] = np.nan
        df["gps_speed_enu_m_s"] = np.nan
        valid_idx = df.index[df["is_valid"]].to_numpy()
        if len(valid_idx) >= 2:
            dx = np.diff(df.loc[valid_idx, "gps_x_east_m"].to_numpy())
            dy = np.diff(df.loc[valid_idx, "gps_y_north_m"].to_numpy())
            dt = np.diff(df.loc[valid_idx, "t_sec"].to_numpy()).clip(min=1e-3)
            dist = np.hypot(dx, dy)
            speed = dist / dt
            course = np.where(dist > 0.5, np.arctan2(dy, dx), np.nan)
            df.loc[valid_idx[1:], "gps_course_rad"] = course
            df.loc[valid_idx[1:], "gps_speed_enu_m_s"] = speed

        return df, self.origin