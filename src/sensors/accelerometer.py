import numpy as np
import pandas as pd


class AccelerometerProcessor:
    """Smooths and validates vehicle-frame forward/lateral acceleration streams."""

    def __init__(self, smoothing_window: int = 3, clip_abs_m_s2: float = 15.0) -> None:
        self.smoothing_window = max(1, int(smoothing_window))
        self.clip_abs_m_s2 = float(clip_abs_m_s2)

    def process(self, acc_df: pd.DataFrame) -> pd.DataFrame:
        df = acc_df.copy().sort_values("t_sec").reset_index(drop=True)

        a_fwd = df["a_forward_m_s2"].to_numpy(dtype=float)
        a_lat = df["a_lateral_m_s2"].to_numpy(dtype=float)

        plausible = (np.abs(a_fwd) <= self.clip_abs_m_s2) & (np.abs(a_lat) <= self.clip_abs_m_s2)
        df["is_valid"] = df["is_valid"].astype(bool) & plausible

        if self.smoothing_window > 1:
            df["a_forward_clean_m_s2"] = (
                df["a_forward_m_s2"]
                .rolling(window=self.smoothing_window, min_periods=1, center=True)
                .mean()
            )
            df["a_lateral_clean_m_s2"] = (
                df["a_lateral_m_s2"]
                .rolling(window=self.smoothing_window, min_periods=1, center=True)
                .mean()
            )
        else:
            df["a_forward_clean_m_s2"] = df["a_forward_m_s2"]
            df["a_lateral_clean_m_s2"] = df["a_lateral_m_s2"]

        return df