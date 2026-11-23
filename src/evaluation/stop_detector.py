from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StopEvent:
    stop_id: int
    stop_type: str       # Always "Stop" (counted uniformly for all stop & slow-moving events)
    display_state: str   # "Stop" (v <= 0.12 m/s) or "Slow Moving" (0.12 < v <= 0.50 m/s)
    start_frame: int
    banner_start_frame: int
    end_frame: int
    start_t_video_s: float
    banner_start_t_video_s: float
    end_t_video_s: float
    start_t_sync_s: float
    end_t_sync_s: float
    duration_s: float
    px_fused_m: float
    py_fused_m: float
    px_track_a_m: float
    py_track_a_m: float
    pos_diff_a_vs_c_m: float
    mean_speed_fused_m_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StopDetector:
    """
    Detects stationary and very slow-moving (creeping traffic) intervals.
    All detected intervals are counted under 'Stop', while very slow-moving
    states are tagged with display_state='Slow Moving' (no 'Brief Halt' state).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.smooth_win = int(cfg.get("smoothing_window_frames", 11))
        self.dead_stop_thresh = float(cfg.get("dead_stop_speed_m_s", 0.12))
        # Backwards-compatible fallback if old key 'speed_thresh_m_s' is present
        self.slow_move_thresh = float(
            cfg.get("slow_move_speed_m_s", cfg.get("speed_thresh_m_s", 0.50))
        )
        self.max_yaw_rate = float(cfg.get("max_yaw_rate_rad_s", 0.10))
        self.merge_gap_s = float(cfg.get("merge_gap_s", 1.0))
        self.min_dur_s = float(cfg.get("min_duration_s", 1.2))
        self.banner_delay_frames = int(cfg.get("banner_delay_frames", 24))

    def detect(
        self,
        track_c_df: pd.DataFrame,
        track_a_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        t_vid = track_c_df["t_video_s"].to_numpy(dtype=float)
        t_sync = track_c_df["t_sync_s"].to_numpy(dtype=float)
        frames = track_c_df["frame_idx"].to_numpy(dtype=int)

        v_c = track_c_df["speed_fused_m_s"].fillna(0.0).to_numpy(dtype=float)
        w_c = track_c_df["yaw_rate_fused_rad_s"].fillna(0.0).to_numpy(dtype=float)
        px_c = track_c_df["px_fused_m"].ffill().bfill().to_numpy(dtype=float)
        py_c = track_c_df["py_fused_m"].ffill().bfill().to_numpy(dtype=float)

        if track_a_df is not None:
            px_a = track_a_df["px_m"].ffill().bfill().to_numpy(dtype=float)
            py_a = track_a_df["py_m"].ffill().bfill().to_numpy(dtype=float)
        else:
            px_a, py_a = px_c, py_c

        # Causal rolling average (center=False)
        v_smooth = (
            pd.Series(v_c)
            .rolling(window=self.smooth_win, min_periods=1, center=False)
            .mean()
            .to_numpy()
        )
        w_smooth = (
            pd.Series(np.abs(w_c))
            .rolling(window=self.smooth_win, min_periods=1, center=False)
            .mean()
            .to_numpy()
        )

        # Unified detection mask covering both dead-stops and creeping/slow-moving traffic
        is_stopped_or_slow = (v_smooth <= self.slow_move_thresh) & (w_smooth <= self.max_yaw_rate)

        raw_segments: List[List[int]] = []
        in_stop = False
        s_idx = 0
        for i, flag in enumerate(is_stopped_or_slow):
            if flag and not in_stop:
                in_stop = True
                s_idx = i
            elif not flag and in_stop:
                raw_segments.append([s_idx, i - 1])
                in_stop = False
        if in_stop:
            raw_segments.append([s_idx, len(t_vid) - 1])

        merged: List[List[int]] = []
        for seg in raw_segments:
            if not merged:
                merged.append(seg)
            else:
                prev = merged[-1]
                if (t_vid[seg[0]] - t_vid[prev[1]]) <= self.merge_gap_s:
                    prev[1] = seg[1]
                else:
                    merged.append(seg)

        events: List[Dict[str, Any]] = []
        for s_i, e_i in merged:
            dur_s = float(t_vid[e_i] - t_vid[s_i])
            if dur_s < self.min_dur_s:
                continue

            x_c_med = float(np.median(px_c[s_i : e_i + 1]))
            y_c_med = float(np.median(py_c[s_i : e_i + 1]))
            x_a_med = float(np.median(px_a[s_i : e_i + 1]))
            y_a_med = float(np.median(py_a[s_i : e_i + 1]))
            diff_m = float(np.hypot(x_a_med - x_c_med, y_a_med - y_c_med))
            mean_spd = float(np.mean(v_c[s_i : e_i + 1]))

            # Count every event as "Stop", but distinguish UI display label for slow-moving crawl
            display_state = "Stop" if mean_spd <= self.dead_stop_thresh else "Slow Moving"

            # Apply banner confirmation delay so the alert appears after visible deceleration
            b_idx = min(e_i, s_i + self.banner_delay_frames)

            ev = StopEvent(
                stop_id=len(events) + 1,
                stop_type="Stop",
                display_state=display_state,
                start_frame=int(frames[s_i]),
                banner_start_frame=int(frames[b_idx]),
                end_frame=int(frames[e_i]),
                start_t_video_s=round(float(t_vid[s_i]), 2),
                banner_start_t_video_s=round(float(t_vid[b_idx]), 2),
                end_t_video_s=round(float(t_vid[e_i]), 2),
                start_t_sync_s=round(float(t_sync[s_i]), 2),
                end_t_sync_s=round(float(t_sync[e_i]), 2),
                duration_s=round(dur_s, 2),
                px_fused_m=round(x_c_med, 2),
                py_fused_m=round(y_c_med, 2),
                px_track_a_m=round(x_a_med, 2),
                py_track_a_m=round(y_a_med, 2),
                pos_diff_a_vs_c_m=round(diff_m, 2),
                mean_speed_fused_m_s=round(mean_spd, 3),
            )
            events.append(ev.to_dict())

        return pd.DataFrame(events)