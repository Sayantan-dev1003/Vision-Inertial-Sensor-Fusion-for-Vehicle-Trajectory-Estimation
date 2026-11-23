from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import pandas as pd


class TrajectoryVideoOverlay:
    """
    Renders full-length side-by-side comparison videos:
      - Left Panel  (800x720): Synchronized dashcam footage with live telemetry & Turn/Stop/Slow-Moving alerts
      - Right Panel (800x720): Live 2D ENU Trajectory Plot with Turn/Stop markers & heading vector
    """

    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        recording_id: str,
        start_time_s: float = 0.0,
        duration_s: float | None = None,
        frame_stride: int = 1,
        output_fps: float = 30.0,
    ) -> None:
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.recording_id = str(recording_id)
        self.start_time_s = float(start_time_s)
        self.duration_s = float(duration_s) if duration_s is not None else None
        self.frame_stride = max(1, int(frame_stride))
        self.output_fps = float(output_fps) / self.frame_stride

    @staticmethod
    def _compute_unified_bounds(
        ta: pd.DataFrame, tc: pd.DataFrame
    ) -> Tuple[float, float, float, float]:
        """Use identical isotropic ENU map bounds for both Track A and Track C videos."""
        min_x = float(min(ta["px_m"].min(), tc["px_fused_m"].min())) - 60.0
        max_x = float(max(ta["px_m"].max(), tc["px_fused_m"].max())) + 60.0
        min_y = float(min(ta["py_m"].min(), tc["py_fused_m"].min())) - 60.0
        max_y = float(max(ta["py_m"].max(), tc["py_fused_m"].max())) + 60.0

        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        half_span = 0.5 * max(max_x - min_x, max_y - min_y)
        return cx - half_span, cx + half_span, cy - half_span, cy + half_span

    @staticmethod
    def _world_to_canvas(
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        bounds: Tuple[float, float, float, float],
        plot_rect: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        min_x, max_x, min_y, max_y = bounds
        rx, ry, rw, rh = plot_rect
        u = rx + ((x_arr - min_x) / max(max_x - min_x, 1.0)) * rw
        v = ry + rh - ((y_arr - min_y) / max(max_y - min_y, 1.0)) * rh
        return np.round(u).astype(np.int32), np.round(v).astype(np.int32)

    def _build_static_map_canvas(
        self,
        panel_w: int,
        panel_h: int,
        plot_rect: Tuple[int, int, int, int],
        bounds: Tuple[float, float, float, float],
        track_title: str,
        trail_color_bgr: Tuple[int, int, int],
        gps_df: pd.DataFrame,
        turns_df: pd.DataFrame,
        stops_df: pd.DataFrame,
        use_track_a_coords: bool = False,
    ) -> np.ndarray:
        canvas = np.full((panel_h, panel_w, 3), (15, 23, 42), dtype=np.uint8)
        rx, ry, rw, rh = plot_rect
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (30, 41, 59), -1)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (71, 85, 105), 1)

        min_x, max_x, min_y, max_y = bounds
        # Draw 200m coordinate grid lines and labels
        grid_step = 200.0
        gx_start = np.ceil(min_x / grid_step) * grid_step
        for gx in np.arange(gx_start, max_x, grid_step):
            u_pt, _ = self._world_to_canvas(np.array([gx]), np.array([min_y]), bounds, plot_rect)
            cv2.line(canvas, (int(u_pt[0]), ry), (int(u_pt[0]), ry + rh), (51, 65, 85), 1)
            cv2.putText(
                canvas,
                f"{int(gx)}m",
                (int(u_pt[0]) - 18, ry + rh + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (148, 163, 184),
                1,
            )

        gy_start = np.ceil(min_y / grid_step) * grid_step
        for gy in np.arange(gy_start, max_y, grid_step):
            _, v_pt = self._world_to_canvas(np.array([min_x]), np.array([gy]), bounds, plot_rect)
            cv2.line(canvas, (rx, int(v_pt[0])), (rx + rw, int(v_pt[0])), (51, 65, 85), 1)
            cv2.putText(
                canvas,
                f"{int(gy)}m",
                (8, int(v_pt[0]) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (148, 163, 184),
                1,
            )

        # Plot faint background GPS reference dots
        if not gps_df.empty:
            gu, gv = self._world_to_canvas(
                gps_df["gps_x_east_m"].to_numpy(dtype=float),
                gps_df["gps_y_north_m"].to_numpy(dtype=float),
                bounds,
                plot_rect,
            )
            for u_i, v_i in zip(gu, gv):
                if rx <= u_i <= rx + rw and ry <= v_i <= ry + rh:
                    cv2.circle(canvas, (int(u_i), int(v_i)), 2, (100, 116, 139), -1)

        # Plot Stop markers (red squares — all stop & slow-moving events counted under Stop)
        if not stops_df.empty:
            sx_col = "px_track_a_m" if use_track_a_coords else "px_fused_m"
            sy_col = "py_track_a_m" if use_track_a_coords else "py_fused_m"
            su, sv = self._world_to_canvas(
                stops_df[sx_col].to_numpy(dtype=float),
                stops_df[sy_col].to_numpy(dtype=float),
                bounds,
                plot_rect,
            )
            for row, u_i, v_i in zip(stops_df.itertuples(index=False), su, sv):
                if rx <= u_i <= rx + rw and ry <= v_i <= ry + rh:
                    cv2.rectangle(canvas, (int(u_i) - 4, int(v_i) - 4), (int(u_i) + 4, int(v_i) + 4), (68, 68, 239), -1)
                    if row.duration_s >= 15.0:
                        cv2.putText(
                            canvas,
                            f"S{row.stop_id}({row.duration_s:.0f}s)",
                            (int(u_i) + 6, int(v_i) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (252, 165, 165),
                            1,
                        )

        # Plot Turn markers (amber circles for turns, purple diamonds for U-turns)
        if not turns_df.empty:
            tx_col = "apex_px_track_a_m" if use_track_a_coords else "apex_px_m"
            ty_col = "apex_py_track_a_m" if use_track_a_coords else "apex_py_m"
            tu, tv = self._world_to_canvas(
                turns_df[tx_col].to_numpy(dtype=float),
                turns_df[ty_col].to_numpy(dtype=float),
                bounds,
                plot_rect,
            )
            for row, u_i, v_i in zip(turns_df.itertuples(index=False), tu, tv):
                if rx <= u_i <= rx + rw and ry <= v_i <= ry + rh:
                    if row.maneuver_type == "U-Turn":
                        pts = np.array(
                            [[u_i, v_i - 6], [u_i + 6, v_i], [u_i, v_i + 6], [u_i - 6, v_i]],
                            dtype=np.int32,
                        )
                        cv2.fillPoly(canvas, [pts], (234, 51, 147))
                        cv2.putText(
                            canvas,
                            f"T{row.turn_id}:U-Turn",
                            (int(u_i) + 7, int(v_i) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.37,
                            (245, 208, 254),
                            1,
                        )
                    else:
                        cv2.circle(canvas, (int(u_i), int(v_i)), 4, (11, 158, 245), -1)
                        if row.maneuver_type in ("Left Turn", "Right Turn"):
                            cv2.putText(
                                canvas,
                                f"T{row.turn_id}",
                                (int(u_i) + 5, int(v_i) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.34,
                                (253, 230, 138),
                                1,
                            )

        # Header & Legend
        cv2.putText(canvas, track_title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (248, 250, 252), 2)
        cv2.putText(canvas, "East [m] vs North [m] (Local ENU)", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)

        # Legend row at bottom
        ly = panel_h - 16
        cv2.line(canvas, (20, ly - 4), (42, ly - 4), trail_color_bgr, 3)
        cv2.putText(canvas, "Live Path", (48, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (226, 232, 240), 1)
        cv2.circle(canvas, (130, ly - 4), 4, (11, 158, 245), -1)
        cv2.putText(canvas, "Turn (T#)", (140, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (226, 232, 240), 1)
        cv2.rectangle(canvas, (222, ly - 8), (230, ly), (68, 68, 239), -1)
        cv2.putText(canvas, "Stop (S#)", (236, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (226, 232, 240), 1)
        cv2.circle(canvas, (318, ly - 4), 3, (100, 116, 139), -1)
        cv2.putText(canvas, "1Hz GPS", (326, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (226, 232, 240), 1)

        return canvas

    def render_side_by_side_track(
        self,
        track_label: str,
        output_filename: str,
        ta_df: pd.DataFrame,
        tc_df: pd.DataFrame,
        turns_df: pd.DataFrame,
        stops_df: pd.DataFrame,
        use_track_a: bool = False,
    ) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / output_filename

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open dashcam video: {self.video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(tc_df))
        start_frame = int(np.clip(round(self.start_time_s * src_fps), 0, total_frames - 1))
        if self.duration_s is None:
            end_frame = total_frames
        else:
            end_frame = int(np.clip(round((self.start_time_s + self.duration_s) * src_fps), start_frame + 1, total_frames))

        panel_w, panel_h = 800, 720
        out_w, out_h = panel_w * 2, panel_h
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self.output_fps, (out_w, out_h))

        bounds = self._compute_unified_bounds(ta_df, tc_df)
        plot_rect = (65, 70, 705, 585)
        trail_color = (68, 68, 239) if use_track_a else (248, 189, 56)  # Red for Track A, Cyan-Blue for Track C

        gps_dedup = ta_df[ta_df["gps_valid"]].drop_duplicates(subset=["gps_x_east_m", "gps_y_north_m"])
        trail_canvas = self._build_static_map_canvas(
            panel_w=panel_w,
            panel_h=panel_h,
            plot_rect=plot_rect,
            bounds=bounds,
            track_title=track_label,
            trail_color_bgr=trail_color,
            gps_df=gps_dedup,
            turns_df=turns_df,
            stops_df=stops_df,
            use_track_a_coords=use_track_a,
        )

        # Precompute trajectory pixel arrays for O(1) incremental drawing
        if use_track_a:
            px_arr = ta_df["px_m"].ffill().bfill().to_numpy(dtype=float)
            py_arr = ta_df["py_m"].ffill().bfill().to_numpy(dtype=float)
            spd_arr = ta_df["speed_m_s"].fillna(0.0).to_numpy(dtype=float)
            psi_arr = ta_df["heading_rad"].fillna(0.0).to_numpy(dtype=float)
            yaw_arr = ta_df["yaw_rate_rad_s"].fillna(0.0).to_numpy(dtype=float)
            std_arr = np.hypot(ta_df["std_px_m"].fillna(0.0), ta_df["std_py_m"].fillna(0.0)).to_numpy(dtype=float)
        else:
            px_arr = tc_df["px_fused_m"].ffill().bfill().to_numpy(dtype=float)
            py_arr = tc_df["py_fused_m"].ffill().bfill().to_numpy(dtype=float)
            spd_arr = tc_df["speed_fused_m_s"].fillna(0.0).to_numpy(dtype=float)
            psi_arr = tc_df["heading_fused_rad"].fillna(0.0).to_numpy(dtype=float)
            yaw_arr = tc_df["yaw_rate_fused_rad_s"].fillna(0.0).to_numpy(dtype=float)
            std_arr = np.hypot(tc_df["std_px_fused_m"].fillna(0.0), tc_df["std_py_fused_m"].fillna(0.0)).to_numpy(dtype=float)

        t_vid_arr = tc_df["t_video_s"].to_numpy(dtype=float)
        t_sync_arr = tc_df["t_sync_s"].to_numpy(dtype=float)
        u_arr, v_arr = self._world_to_canvas(px_arr, py_arr, bounds, plot_rect)

        # Lookup arrays for fast O(1) active Turn / Stop / Slow-Moving banner lookup per frame
        turn_lookup = {}
        if not turns_df.empty:
            for row in turns_df.itertuples(index=False):
                label = f"TURN #{row.turn_id}: {row.maneuver_type.upper()} ({row.delta_heading_fused_deg:+.0f} deg)"
                onset_frame = int(getattr(row, "commit_frame", row.start_frame))
                for fi in range(onset_frame, min(int(row.end_frame) + 1, total_frames)):
                    turn_lookup[fi] = (label, row.maneuver_type == "U-Turn")

        stop_lookup = {}
        if not stops_df.empty:
            for row in stops_df.itertuples(index=False):
                # Counted as Stop (S#), displayed as "STOP" or "SLOW MOVING" (never "BRIEF HALT")
                disp_state = str(getattr(row, "display_state", "Stop")).upper()
                is_slow_moving = disp_state == "SLOW MOVING"
                label = (
                    f"{disp_state} (S#{row.stop_id}): ({row.duration_s:.1f}s) "
                    f"@ ({row.px_fused_m:.0f}m, {row.py_fused_m:.0f}m)"
                )
                onset_frame = int(getattr(row, "banner_start_frame", row.start_frame))
                for fi in range(onset_frame, min(int(row.end_frame) + 1, total_frames)):
                    stop_lookup[fi] = (label, is_slow_moving)

        # Draw historical trail if start_frame > 0
        for fi in range(1, start_frame + 1):
            cv2.line(trail_canvas, (int(u_arr[fi - 1]), int(v_arr[fi - 1])), (int(u_arr[fi]), int(v_arr[fi])), trail_color, 2)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        prev_drawn_idx = start_frame

        for f_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if f_idx > prev_drawn_idx:
                for k in range(prev_drawn_idx + 1, f_idx + 1):
                    cv2.line(
                        trail_canvas,
                        (int(u_arr[k - 1]), int(v_arr[k - 1])),
                        (int(u_arr[k]), int(v_arr[k])),
                        trail_color,
                        2,
                    )
                prev_drawn_idx = f_idx

            if (f_idx - start_frame) % self.frame_stride != 0:
                continue

            # Left Panel: Clean Dashcam Footage + Bottom Telemetry Bar
            left_panel = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            bar_overlay = left_panel.copy()
            cv2.rectangle(bar_overlay, (0, panel_h - 68), (panel_w, panel_h), (15, 23, 42), -1)
            left_panel = cv2.addWeighted(bar_overlay, 0.78, left_panel, 0.22, 0)

            mins = int(t_vid_arr[f_idx] // 60)
            secs = t_vid_arr[f_idx] % 60.0
            spd_kmh = spd_arr[f_idx] * 3.6
            yaw_deg = np.degrees(yaw_arr[f_idx])

            cv2.putText(
                left_panel,
                f"Dashcam Video | {mins:02d}:{secs:04.1f} (t_vid={t_vid_arr[f_idx]:.1f}s | t_sync={t_sync_arr[f_idx]:.1f}s)",
                (16, panel_h - 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (226, 232, 240),
                1,
            )
            cv2.putText(
                left_panel,
                f"Speed: {spd_kmh:5.1f} km/h ({spd_arr[f_idx]:.2f} m/s) | Yaw Rate: {yaw_deg:+5.1f} deg/s",
                (16, panel_h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (56, 189, 248) if not use_track_a else (252, 165, 165),
                2,
            )

            # Active Turn, Slow Moving, or Stop Banner on Left Video
            if f_idx in turn_lookup:
                t_text, is_uturn = turn_lookup[f_idx]
                b_col = (147, 51, 234) if is_uturn else (217, 119, 6)
                cv2.rectangle(left_panel, (16, 16), (panel_w - 16, 62), b_col, -1)
                cv2.putText(left_panel, t_text, (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            elif f_idx in stop_lookup:
                s_text, is_slow_moving = stop_lookup[f_idx]
                # Steel-teal banner for Slow Moving; Crimson-red banner for full Stop
                b_col = (180, 119, 14) if is_slow_moving else (28, 28, 185)
                cv2.rectangle(left_panel, (16, 16), (panel_w - 16, 62), b_col, -1)
                cv2.putText(left_panel, s_text, (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)

            # Right Panel: Live 2D Trajectory Map + Current Vehicle Pose Marker
            right_panel = trail_canvas.copy()
            cx_px, cy_px = int(u_arr[f_idx]), int(v_arr[f_idx])
            psi = float(psi_arr[f_idx])
            tip_x = int(round(cx_px + 16 * np.cos(psi)))
            tip_y = int(round(cy_px - 16 * np.sin(psi)))

            cv2.circle(right_panel, (cx_px, cy_px), 6, (34, 197, 94), -1)
            cv2.arrowedLine(right_panel, (cx_px, cy_px), (tip_x, tip_y), (255, 255, 255), 2, tipLength=0.35)

            # Live (x, y, timestamp) telemetry box on map top-right
            cv2.rectangle(right_panel, (430, 14), (780, 66), (30, 41, 59), -1)
            cv2.putText(
                right_panel,
                f"Pos: ({px_arr[f_idx]:6.1f}m, {py_arr[f_idx]:6.1f}m)",
                (440, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (248, 250, 252),
                1,
            )
            cv2.putText(
                right_panel,
                f"1-Sigma: +/-{std_arr[f_idx]:.1f}m | Frame: {f_idx}",
                (440, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (148, 163, 184),
                1,
            )

            combined = np.hstack([left_panel, right_panel])
            writer.write(combined)

            if f_idx > 0 and f_idx % 10000 == 0:
                print(f"[{track_label}] Rendered {f_idx}/{end_frame} frames...")

        cap.release()
        writer.release()
        return str(out_path)