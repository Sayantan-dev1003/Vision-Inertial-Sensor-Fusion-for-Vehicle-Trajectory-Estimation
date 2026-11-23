from typing import List, Tuple
import numpy as np
import pandas as pd


class GPSProcessor:
    """Filters GPS multipath jumps, flags holdout windows, and estimates per-fix measurement noise."""

    def __init__(
        self,
        base_pos_std_m: float = 4.5,
        max_speed_m_s: float = 32.0,
        max_single_step_m: float = 45.0,
        holdout_intervals_s: List[Tuple[float, float]] | None = None,
    ) -> None:
        self.base_pos_std_m = float(base_pos_std_m)
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_single_step_m = float(max_single_step_m)
        self.holdout_intervals_s = list(holdout_intervals_s or [])

    def process(self, gps_df: pd.DataFrame) -> pd.DataFrame:
        df = gps_df.copy().sort_values("t_sec").reset_index(drop=True)

        valid_mask = df["is_valid"].astype(bool).to_numpy(copy=True)
        t_arr = df["t_sec"].to_numpy(dtype=float, copy=True)
        x_arr = df["gps_x_east_m"].to_numpy(dtype=float, copy=True)
        y_arr = df["gps_y_north_m"].to_numpy(dtype=float, copy=True)

        is_outlier = np.zeros(len(df), dtype=bool)
        gap_s = np.zeros(len(df), dtype=float)

        # Anchor-based kinematic jump & speed spike rejection
        anchor_t: float | None = None
        anchor_x: float | None = None
        anchor_y: float | None = None

        for i in range(len(df)):
            if not valid_mask[i] or not (np.isfinite(x_arr[i]) and np.isfinite(y_arr[i])):
                valid_mask[i] = False
                continue

            if anchor_t is None:
                anchor_t = t_arr[i]
                anchor_x = x_arr[i]
                anchor_y = y_arr[i]
                gap_s[i] = 0.0
                continue

            dt = max(1e-3, t_arr[i] - anchor_t)
            dist = float(np.hypot(x_arr[i] - anchor_x, y_arr[i] - anchor_y))
            implied_speed = dist / dt

            # Reject multipath spikes: impossible speed or sudden >45m step within <=2.5s
            if implied_speed > self.max_speed_m_s or (dt <= 2.5 and dist > self.max_single_step_m):
                is_outlier[i] = True
                valid_mask[i] = False
                gap_s[i] = dt
                continue

            gap_s[i] = dt
            anchor_t = t_arr[i]
            anchor_x = x_arr[i]
            anchor_y = y_arr[i]

        df["is_gps_outlier"] = is_outlier
        df["gps_use_for_update"] = valid_mask
        df["is_holdout"] = False

        for t_start, t_end in self.holdout_intervals_s:
            in_window = (df["t_sec"] >= float(t_start)) & (df["t_sec"] <= float(t_end))
            df.loc[in_window, "is_holdout"] = True
            df.loc[in_window, "gps_use_for_update"] = False

        df["gps_gap_s"] = gap_s
        gap_inflation = np.clip(1.0 + 0.08 * np.maximum(0.0, gap_s - 1.5), 1.0, 3.0)
        df["gps_pos_std_m"] = self.base_pos_std_m * gap_inflation
        return df