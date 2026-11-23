import numpy as np
import pandas as pd


class GyroscopeProcessor:
    """Validates and filters vehicle-frame yaw rate measurements."""

    def __init__(self, smoothing_window: int = 3, clip_abs_rad_s: float = 2.5) -> None:
        self.smoothing_window = max(1, int(smoothing_window))
        self.clip_abs_rad_s = float(clip_abs_rad_s)

    def process(self, gyro_df: pd.DataFrame) -> pd.DataFrame:
        df = gyro_df.copy().sort_values("t_sec").reset_index(drop=True)

        yaw_rate = df["yaw_rate_rad_s"].to_numpy(dtype=float)
        df["is_valid"] = df["is_valid"].astype(bool) & (np.abs(yaw_rate) <= self.clip_abs_rad_s)

        if self.smoothing_window > 1:
            df["yaw_rate_clean_rad_s"] = (
                df["yaw_rate_rad_s"]
                .rolling(window=self.smoothing_window, min_periods=1, center=True)
                .mean()
            )
        else:
            df["yaw_rate_clean_rad_s"] = df["yaw_rate_rad_s"]

        return df