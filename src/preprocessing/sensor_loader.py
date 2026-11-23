from pathlib import Path
from typing import Any, Dict, Iterable
import numpy as np
import pandas as pd

from .data_schema import SensorSchema


class SensorDataLoader:
    """Loads raw multiplexed sensor CSVs, retains only active sensors, and validates quality."""

    def __init__(
        self,
        gps_max_speed_m_s: float = 45.0,
        active_sensors: Iterable[str] = ("accr", "gyro", "gps"),
    ) -> None:
        self.gps_max_speed_m_s = float(gps_max_speed_m_s)
        self.active_sensors = tuple(active_sensors)
        for stype in self.active_sensors:
            if stype not in SensorSchema.ACTIVE_SENSOR_COLUMNS:
                raise ValueError(f"Invalid active sensor requested: '{stype}'")

    def load_raw(self, csv_path: str | Path, has_header: bool = False) -> pd.DataFrame:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Sensor CSV not found: {path}")

        df = pd.read_csv(
            path,
            header=0 if has_header else None,
            names=list(SensorSchema.RAW_COLUMNS),
        )

        if df.shape[1] != len(SensorSchema.RAW_COLUMNS):
            raise ValueError(f"Expected {len(SensorSchema.RAW_COLUMNS)} columns, got {df.shape[1]}")

        df["sensor_type"] = df["sensor_type"].astype(str).str.strip()
        unknown_types = set(df["sensor_type"].unique()) - SensorSchema.all_known_sensor_types()
        if unknown_types:
            raise ValueError(f"Unexpected sensor_type values in CSV: {unknown_types}")

        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        if df["timestamp_ms"].isna().any():
            raise ValueError("Corrupted non-numeric values found in timestamp_ms")

        # Establish common relative start time BEFORE dropping ignored sensor rows
        df = df.sort_values("timestamp_ms", kind="mergesort").reset_index(drop=True)
        t0_ms = int(df["timestamp_ms"].iloc[0])
        df["t_sec"] = (df["timestamp_ms"] - t0_ms) / 1000.0
        df["t0_ms"] = t0_ms

        # Keep only accr, gyro, gps (discard lin_accr and mag_field)
        df = df[df["sensor_type"].isin(self.active_sensors)].reset_index(drop=True)
        return df

    def split_streams(self, raw_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        streams: Dict[str, pd.DataFrame] = {}
        for stype in self.active_sensors:
            active_cols = SensorSchema.get_active_columns(stype)
            sub = raw_df.loc[raw_df["sensor_type"] == stype, ["timestamp_ms", "t_sec", "datetime_str"] + active_cols,].copy()
            sub = sub.reset_index(drop=True)
            sub["dt_s"] = sub["t_sec"].diff()
            sub["is_valid"] = np.isfinite(sub[active_cols]).all(axis=1)

            if stype == "gps":
                sub = self._validate_gps_stream(sub)
            else:
                all_zero = sub[active_cols].abs().sum(axis=1) == 0.0
                sub["is_valid"] = sub["is_valid"] & (~all_zero)

            streams[stype] = sub
        return streams

    def _validate_gps_stream(self, gps_df: pd.DataFrame) -> pd.DataFrame:
        non_zero = (gps_df["gps_lat"].abs() > 1e-6) & (gps_df["gps_lon"].abs() > 1e-6)
        dlat = gps_df["gps_lat"].diff().abs().fillna(1.0)
        dlon = gps_df["gps_lon"].diff().abs().fillna(1.0)
        gps_df["is_duplicate"] = (dlat == 0.0) & (dlon == 0.0)

        lat_rad = np.deg2rad(gps_df["gps_lat"].to_numpy())
        dy = gps_df["gps_lat"].diff().fillna(0.0).to_numpy() * 111132.92
        dx = (gps_df["gps_lon"].diff().fillna(0.0).to_numpy() * 111412.84 * np.cos(lat_rad))
        step_m = np.hypot(dx, dy)
        dt = gps_df["dt_s"].fillna(1.0).clip(lower=1e-3).to_numpy()
        implied_speed = step_m / dt
        gps_df["implied_gps_speed_m_s"] = implied_speed

        gps_df["is_valid"] = (gps_df["is_valid"] & non_zero & (~gps_df["is_duplicate"]) & (implied_speed <= self.gps_max_speed_m_s))
        return gps_df

    @staticmethod
    def summarize(streams: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for stype, df in streams.items():
            valid_dt = df["dt_s"].dropna()
            med_dt = float(valid_dt.median()) if len(valid_dt) > 0 else np.nan
            summary[stype] = {
                "total_samples": int(len(df)),
                "valid_samples": int(df["is_valid"].sum()),
                "valid_ratio": round(float(df["is_valid"].mean()), 4) if len(df) else 0.0,
                "median_dt_ms": round(med_dt * 1000.0, 2) if np.isfinite(med_dt) else None,
                "approx_hz": round(1.0 / med_dt, 2) if med_dt and med_dt > 0 else None,
                "max_gap_s": round(float(valid_dt.max()), 3) if len(valid_dt) else None,
            }
        return summary