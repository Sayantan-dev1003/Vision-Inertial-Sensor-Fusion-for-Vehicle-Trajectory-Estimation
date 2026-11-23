from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TurnEvent:
    turn_id: int
    start_frame: int
    commit_frame: int
    apex_frame: int
    end_frame: int
    start_t_video_s: float
    commit_t_video_s: float
    apex_t_video_s: float
    end_t_video_s: float
    start_t_sync_s: float
    end_t_sync_s: float
    duration_s: float
    direction: str
    maneuver_type: str
    delta_heading_fused_deg: float
    delta_heading_track_a_deg: float
    delta_heading_track_b_deg: float
    peak_yaw_rate_rad_s: float
    mean_speed_m_s: float
    start_px_m: float
    start_py_m: float
    apex_px_m: float
    apex_py_m: float
    end_px_m: float
    end_py_m: float
    apex_px_track_a_m: float
    apex_py_track_a_m: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TurnDetector:
    """Causal hysteresis turn detector with steering-commit onset & slight-turn sensitivity."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.smooth_win = int(cfg.get("smoothing_window_frames", 9))
        self.enter_thresh = float(cfg.get("enter_yaw_rate_rad_s", 0.065))
        self.commit_thresh = float(cfg.get("commit_yaw_rate_rad_s", 0.095))
        self.exit_thresh = float(cfg.get("exit_yaw_rate_rad_s", 0.032))
        self.merge_gap_s = float(cfg.get("merge_gap_s", 0.85))
        self.min_dur_s = float(cfg.get("min_duration_s", 0.9))
        self.min_angle_deg = float(cfg.get("min_angle_deg", 13.0))
        self.full_turn_deg = float(cfg.get("full_turn_angle_deg", 55.0))
        self.u_turn_deg = float(cfg.get("u_turn_angle_deg", 145.0))
        self.min_entry_speed = float(cfg.get("min_entry_speed_m_s", 0.8))
        self.min_mean_speed = float(cfg.get("min_mean_speed_m_s", 1.2))

    def detect(
        self,
        track_c_df: pd.DataFrame,
        track_a_df: pd.DataFrame | None = None,
        track_b_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        t = track_c_df["t_sync_s"].to_numpy(dtype=float)
        t_vid = track_c_df["t_video_s"].to_numpy(dtype=float)
        frames = track_c_df["frame_idx"].to_numpy(dtype=int)
        w = track_c_df["yaw_rate_fused_rad_s"].fillna(0.0).to_numpy(dtype=float)
        v = track_c_df["speed_fused_m_s"].fillna(0.0).to_numpy(dtype=float)
        px = track_c_df["px_fused_m"].ffill().bfill().to_numpy(dtype=float)
        py = track_c_df["py_fused_m"].ffill().bfill().to_numpy(dtype=float)

        # Causal smoothing (center=False) prevents pre-turn early trigger in video
        w_smooth = (
            pd.Series(w)
            .rolling(window=self.smooth_win, min_periods=1, center=False)
            .mean()
            .to_numpy()
        )

        if track_a_df is not None:
            w_a = track_a_df["yaw_rate_rad_s"].fillna(0.0).to_numpy(dtype=float)
            px_a = track_a_df["px_m"].ffill().bfill().to_numpy(dtype=float)
            py_a = track_a_df["py_m"].ffill().bfill().to_numpy(dtype=float)
        else:
            w_a, px_a, py_a = w_smooth, px, py

        w_b = (
            track_b_df["yaw_rate_vis_rad_s"].fillna(0.0).to_numpy(dtype=float)
            if track_b_df is not None
            else w_smooth
        )

        raw_segments: List[List[int]] = []
        in_turn = False
        start_idx = 0
        sign = 0

        for i in range(len(t)):
            if not in_turn:
                if abs(w_smooth[i]) >= self.enter_thresh and v[i] >= self.min_entry_speed:
                    in_turn = True
                    start_idx = i
                    sign = 1 if w_smooth[i] > 0.0 else -1
            else:
                if (sign * w_smooth[i]) < self.exit_thresh:
                    raw_segments.append([start_idx, i - 1, sign])
                    in_turn = False
        if in_turn:
            raw_segments.append([start_idx, len(t) - 1, sign])

        merged: List[List[int]] = []
        for seg in raw_segments:
            if not merged:
                merged.append(seg)
            else:
                prev = merged[-1]
                if seg[2] == prev[2] and (t[seg[0]] - t[prev[1]]) <= self.merge_gap_s:
                    prev[1] = seg[1]
                else:
                    merged.append(seg)

        events: List[Dict[str, Any]] = []
        for s_idx, e_idx, _ in merged:
            dur_s = float(t[e_idx] - t[s_idx])
            if dur_s < self.min_dur_s:
                continue

            seg_w = w_smooth[s_idx : e_idx + 1]
            dt_seg = np.diff(t[s_idx : e_idx + 1], prepend=t[s_idx])
            dpsi_c_deg = float(np.degrees(np.sum(seg_w * dt_seg)))
            dpsi_a_deg = float(np.degrees(np.sum(w_a[s_idx : e_idx + 1] * dt_seg)))
            dpsi_b_deg = float(np.degrees(np.sum(w_b[s_idx : e_idx + 1] * dt_seg)))

            abs_deg = abs(dpsi_c_deg)
            mean_spd = float(np.mean(v[s_idx : e_idx + 1]))
            apex_rel = int(np.argmax(np.abs(seg_w)))
            apex_idx = s_idx + apex_rel
            peak_w = float(abs(seg_w[apex_rel]))

            if abs_deg < self.min_angle_deg or mean_spd < self.min_mean_speed:
                continue

            if abs_deg >= self.u_turn_deg:
                maneuver = "U-Turn"
            elif abs_deg >= self.full_turn_deg:
                maneuver = "Left Turn" if dpsi_c_deg > 0 else "Right Turn"
            else:
                maneuver = "Slight Left" if dpsi_c_deg > 0 else "Slight Right"

            # Commit frame: when steering yaw rate genuinely commits (eliminates early banner)
            active_commit_th = self.commit_thresh if abs_deg >= self.full_turn_deg else self.enter_thresh
            commit_rel = np.where(np.abs(seg_w) >= active_commit_th)[0]
            commit_idx = s_idx + int(commit_rel[0]) if len(commit_rel) > 0 else s_idx

            ev = TurnEvent(
                turn_id=len(events) + 1,
                start_frame=int(frames[s_idx]),
                commit_frame=int(frames[commit_idx]),
                apex_frame=int(frames[apex_idx]),
                end_frame=int(frames[e_idx]),
                start_t_video_s=round(float(t_vid[s_idx]), 2),
                commit_t_video_s=round(float(t_vid[commit_idx]), 2),
                apex_t_video_s=round(float(t_vid[apex_idx]), 2),
                end_t_video_s=round(float(t_vid[e_idx]), 2),
                start_t_sync_s=round(float(t[s_idx]), 2),
                end_t_sync_s=round(float(t[e_idx]), 2),
                duration_s=round(dur_s, 2),
                direction="LEFT (CCW)" if dpsi_c_deg > 0 else "RIGHT (CW)",
                maneuver_type=maneuver,
                delta_heading_fused_deg=round(dpsi_c_deg, 1),
                delta_heading_track_a_deg=round(dpsi_a_deg, 1),
                delta_heading_track_b_deg=round(dpsi_b_deg, 1),
                peak_yaw_rate_rad_s=round(peak_w, 3),
                mean_speed_m_s=round(mean_spd, 2),
                start_px_m=round(float(px[s_idx]), 2),
                start_py_m=round(float(py[s_idx]), 2),
                apex_px_m=round(float(px[apex_idx]), 2),
                apex_py_m=round(float(py[apex_idx]), 2),
                end_px_m=round(float(px[e_idx]), 2),
                end_py_m=round(float(py[e_idx]), 2),
                apex_px_track_a_m=round(float(px_a[apex_idx]), 2),
                apex_py_track_a_m=round(float(py_a[apex_idx]), 2),
            )
            events.append(ev.to_dict())

        return pd.DataFrame(events)