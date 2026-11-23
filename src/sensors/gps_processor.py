from typing import List, Tuple
import numpy as np
import pandas as pd


class GPSProcessor:
    """Filters GPS fixes, flags holdout windows, and estimates per-fix measurement noise."""

    def __init__(
        self,
        base_pos_std_m: float = 4.5,
        max_speed_m_s: float = 45.0,
        holdout_intervals_s: List[Tuple[float, float]] | None = None,
    ) -> None:
        self.base_pos_std_m = float(base_pos_std_m)
        self.max_speed_m_s = float(max_speed_m_s)
        self.holdout_intervals_s = list(holdout_intervals_s or [])

    def process(self, gps_df: pd.DataFrame) -> pd.DataFrame:
        df = gps_df.copy().sort_values("t_sec").reset_index(drop=True)

        # Compute step distance & implied speed only across valid fixes
        df["gps_use_for_update"] = df["is_valid"].astype(bool)
        df["is_holdout"] = False

        for t_start, t_end in self.holdout_intervals_s:
            in_window = (df["t_sec"] >= float(t_start)) & (df["t_sec"] <= float(t_end))
            df.loc[in_window, "is_holdout"] = True
            df.loc[in_window, "gps_use_for_update"] = False

        # Scale position std slightly when time gap since previous valid GPS fix is large
        valid_times = df.loc[df["is_valid"], "t_sec"].to_numpy()
        gap_s = np.zeros(len(df), dtype=float)
        if len(valid_times) > 1:
            valid_indices = df.index[df["is_valid"]].to_numpy()
            dt_valid = np.diff(valid_times, prepend=valid_times[0])
            gap_s[valid_indices] = dt_valid

        df["gps_gap_s"] = gap_s
        gap_inflation = np.clip(1.0 + 0.05 * np.maximum(0.0, gap_s - 1.5), 1.0, 2.5)
        df["gps_pos_std_m"] = self.base_pos_std_m * gap_inflation
        return df