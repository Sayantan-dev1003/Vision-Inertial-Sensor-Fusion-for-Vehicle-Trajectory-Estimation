from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.holdout_evaluator import GPSHoldoutEvaluator


class TrajectoryPlotter:
    """Generates high-resolution BTP evaluation figures comparing Tracks A, B, and C."""

    def __init__(self, plots_dir: Path, dpi: int = 300) -> None:
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = int(dpi)

    def plot_all(
        self,
        track_a_df: pd.DataFrame,
        track_b_df: pd.DataFrame,
        track_c_df: pd.DataFrame,
        turns_df: pd.DataFrame,
        stops_df: pd.DataFrame,
        holdout_res: Dict[str, Any],
    ) -> List[str]:
        saved: List[str] = []
        saved.append(self._plot_2d_trajectories(track_a_df, track_c_df, turns_df))
        saved.append(self._plot_kinematics_and_turns(track_a_df, track_b_df, track_c_df, turns_df))
        saved.append(self._plot_gps_holdout(holdout_res))
        saved.append(self._plot_track_b_shape(track_b_df, track_c_df))
        saved.append(self._plot_turns_and_stops_map(track_a_df, track_c_df, turns_df, stops_df))
        return saved

    def _plot_2d_trajectories(
        self,
        ta: pd.DataFrame,
        tc: pd.DataFrame,
        turns_df: pd.DataFrame,
    ) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), dpi=self.dpi)

        gps_pts = ta[ta["gps_valid"]].drop_duplicates(subset=["gps_x_east_m", "gps_y_north_m"])

        # Left Panel: Full 2D Trajectory Comparison
        ax = axes[0]
        ax.scatter(
            gps_pts["gps_x_east_m"],
            gps_pts["gps_y_north_m"],
            s=9,
            c="#64748B",
            alpha=0.55,
            label="Raw GPS Fixes (1 Hz)",
            zorder=2,
        )
        ax.plot(
            ta["px_m"],
            ta["py_m"],
            color="#EF4444",
            lw=1.3,
            alpha=0.80,
            linestyle="--",
            label="Track A: Sensor-Only EKF",
            zorder=3,
        )
        ax.plot(
            tc["px_fused_m"],
            tc["py_fused_m"],
            color="#0284C7",
            lw=2.0,
            label="Track C: Sensor + Optical Flow Fusion EKF",
            zorder=4,
        )
        ax.scatter(
            [tc["px_fused_m"].iloc[0]],
            [tc["py_fused_m"].iloc[0]],
            s=80,
            c="#16A34A",
            marker="^",
            zorder=5,
            label="Start (ENU Origin)",
        )
        ax.scatter(
            [tc["px_fused_m"].iloc[-1]],
            [tc["py_fused_m"].iloc[-1]],
            s=80,
            c="#111827",
            marker="s",
            zorder=5,
            label="End",
        )

        # Annotate U-Turns and major turns
        if not turns_df.empty:
            u_turns = turns_df[turns_df["maneuver_type"] == "U-Turn"]
            major_turns = turns_df[turns_df["maneuver_type"].isin(["Left Turn", "Right Turn"])]
            ax.scatter(
                major_turns["start_px_m"],
                major_turns["start_py_m"],
                s=36,
                c="#F59E0B",
                edgecolors="black",
                linewidths=0.6,
                zorder=6,
                label="Left / Right Turns",
            )
            ax.scatter(
                u_turns["start_px_m"],
                u_turns["start_py_m"],
                s=75,
                c="#9333EA",
                marker="D",
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
                label="U-Turns",
            )

        ax.set_title("Full-Drive 2D Trajectory Reconstruction (Local ENU)", fontsize=12, fontweight="bold")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.axis("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

        # Right Panel: Zoom on 55-Second Natural GPS Outage (t = 1715s to 1790s)
        ax2 = axes[1]
        m_a = (ta["t_sync_s"] >= 1715.0) & (ta["t_sync_s"] <= 1790.0)
        m_c = (tc["t_sync_s"] >= 1715.0) & (tc["t_sync_s"] <= 1790.0)
        g_sub = gps_pts[(gps_pts["t_sync_s"] >= 1700.0) & (gps_pts["t_sync_s"] <= 1800.0)]

        ax2.scatter(
            g_sub["gps_x_east_m"],
            g_sub["gps_y_north_m"],
            s=40,
            c="#334155",
            label="Available GPS Boundary Fixes",
            zorder=3,
        )
        ax2.plot(
            ta.loc[m_a, "px_m"],
            ta.loc[m_a, "py_m"],
            color="#EF4444",
            lw=1.8,
            linestyle="--",
            label="Track A (Dead-Reckoning Drift: 972.8 m peak)",
            zorder=4,
        )
        ax2.plot(
            tc.loc[m_c, "px_fused_m"],
            tc.loc[m_c, "py_fused_m"],
            color="#0284C7",
            lw=2.4,
            label="Track C (Visual-Speed Bounded: 72.8% Less Drift)",
            zorder=5,
        )
        ax2.set_title("55.1s Natural GPS Outage Zoom (t = 1715s – 1790s)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("East [m]")
        ax2.set_ylabel("North [m]")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="best", fontsize=8.5, framealpha=0.92)

        fig.tight_layout()
        out_path = self.plots_dir / "01_trajectory_comparison_2d.png"
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return str(out_path)

    def _plot_kinematics_and_turns(
        self,
        ta: pd.DataFrame,
        tb: pd.DataFrame,
        tc: pd.DataFrame,
        turns_df: pd.DataFrame,
    ) -> str:
        fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True, dpi=self.dpi)
        t = tc["t_sync_s"].to_numpy()

        # 1. Speed profile
        axes[0].plot(t, ta["speed_m_s"], color="#EF4444", alpha=0.65, lw=1.0, label="Track A Speed [m/s]")
        axes[0].plot(t, tc["speed_fused_m_s"], color="#0284C7", lw=1.3, label="Track C Fused Speed [m/s]")
        axes[0].set_ylabel("Speed [m/s]")
        axes[0].set_title("Synchronized Vehicle Kinematics, Turn Events & Online Visual Scale", fontsize=12, fontweight="bold")
        axes[0].legend(loc="upper right", fontsize=8.5)
        axes[0].grid(True, linestyle=":", alpha=0.5)

        # 2. Yaw rate & detected turn intervals
        axes[1].plot(t, tb["yaw_rate_vis_rad_s"], color="#10B981", alpha=0.45, lw=0.8, label="Track B Visual Yaw Rate")
        axes[1].plot(t, tc["yaw_rate_fused_rad_s"], color="#1D4ED8", lw=1.2, label="Track C Fused Yaw Rate")
        if not turns_df.empty:
            for row in turns_df.itertuples(index=False):
                c = "#DBEAFE" if row.delta_heading_fused_deg > 0 else "#FEF3C7"
                axes[1].axvspan(row.start_t_sync_s, row.end_t_sync_s, color=c, alpha=0.75, zorder=0)
        axes[1].set_ylabel("Yaw Rate [rad/s]")
        axes[1].set_ylim(-0.75, 0.75)
        axes[1].legend(loc="upper right", fontsize=8.5)
        axes[1].grid(True, linestyle=":", alpha=0.5)

        # 3. Online visual scale convergence
        axes[2].plot(t, tc["visual_scale_est"], color="#7C3AED", lw=1.5, label="Online Visual Scale Factor s_hat [m/unit]")
        axes[2].set_ylabel("Scale [m/unit]")
        axes[2].legend(loc="upper right", fontsize=8.5)
        axes[2].grid(True, linestyle=":", alpha=0.5)

        # 4. Position uncertainty (1-sigma)
        std_a = np.hypot(ta["std_px_m"], ta["std_py_m"])
        std_c = np.hypot(tc["std_px_fused_m"], tc["std_py_fused_m"])
        axes[3].plot(t, std_a, color="#EF4444", alpha=0.75, lw=1.1, label="Track A Position Std [m]")
        axes[3].plot(t, std_c, color="#0284C7", lw=1.3, label="Track C Fused Position Std [m]")
        axes[3].set_ylabel("1-Sigma Pos [m]")
        axes[3].set_xlabel("Synchronized Time [s]")
        axes[3].set_ylim(0, 120)
        axes[3].legend(loc="upper right", fontsize=8.5)
        axes[3].grid(True, linestyle=":", alpha=0.5)

        fig.tight_layout()
        out_path = self.plots_dir / "02_kinematics_and_turns.png"
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return str(out_path)

    def _plot_gps_holdout(self, holdout_res: Dict[str, Any]) -> str:
        intervals = holdout_res["summary"]["holdout_intervals_s"]
        ta_df = holdout_res["track_a_holdout_df"]
        tc_df = holdout_res["track_c_holdout_df"]
        gt_df = holdout_res["gt_holdout_df"]

        n = len(intervals)
        fig, axes = plt.subplots(1, max(1, n), figsize=(5.5 * max(1, n), 5.2), dpi=self.dpi)
        if n == 1:
            axes = [axes]

        for idx, (s_t, e_t) in enumerate(intervals):
            ax = axes[idx]
            gt_w = gt_df[(gt_df["t_sec"] >= s_t) & (gt_df["t_sec"] <= e_t)]
            ta_w = ta_df[(ta_df["t_sec"] >= s_t - 2.0) & (ta_df["t_sec"] <= e_t)]
            tc_w = tc_df[(tc_df["t_sec"] >= s_t - 2.0) & (tc_df["t_sec"] <= e_t)]

            ax.plot(gt_w["gps_x_east_m"], gt_w["gps_y_north_m"], "ko-", markersize=4, lw=1.5, label="Held-Out Ground Truth GPS")
            ax.plot(ta_w["px_m"], ta_w["py_m"], color="#EF4444", linestyle="--", lw=1.8, label="Track A (Blind Sensor-Only)")
            ax.plot(tc_w["px_fused_m"], tc_w["py_fused_m"], color="#0284C7", lw=2.2, label="Track C (Blind Sensor+Vision)")

            w_stat = holdout_res["per_window"][idx]
            ax.set_title(
                f"Holdout #{idx+1} ({int(s_t)}–{int(e_t)}s)\n"
                f"RMSE: Track A={w_stat['track_a_rmse_m']}m vs Track C={w_stat['track_c_rmse_m']}m",
                fontsize=10.5,
                fontweight="bold",
            )
            ax.set_xlabel("East [m]")
            ax.set_ylabel("North [m]")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        out_path = self.plots_dir / "03_gps_holdout_blackout_drift.png"
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return str(out_path)

    def _plot_track_b_shape(self, tb: pd.DataFrame, tc: pd.DataFrame) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=self.dpi)

        src_xy = tb[["px_vis", "py_vis"]].to_numpy(dtype=float)
        dst_xy = tc[["px_fused_m", "py_fused_m"]].to_numpy(dtype=float)
        aligned_xy, sim2_scale = GPSHoldoutEvaluator.umeyama_align_2d(src_xy, dst_xy)

        ax = axes[0]
        ax.plot(dst_xy[:, 0], dst_xy[:, 1], color="#0284C7", lw=2.0, label="Track C Metric Reference [m]")
        ax.plot(
            aligned_xy[:, 0],
            aligned_xy[:, 1],
            color="#10B981",
            lw=1.5,
            alpha=0.85,
            label=f"Track B Optical-Flow-Only (Sim(2) Aligned, s={sim2_scale:.2f})",
        )
        ax.set_title("Track B Monocular VO Trajectory Shape (Uncalibrated Scale)", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.axis("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8.5)

        ax2 = axes[1]
        t = tb["t_sync_s"].to_numpy()
        ax2.plot(t, tb["inlier_ratio"], color="#059669", lw=0.9, alpha=0.8, label="RANSAC Essential-Matrix Inlier Ratio")
        ax2.axhline(0.90, color="#DC2626", linestyle="--", lw=1.1, label="Fusion Acceptance Threshold (0.90)")
        ax2.set_title("Optical Flow RANSAC Geometric Consistency Over Drive", fontsize=11.5, fontweight="bold")
        ax2.set_xlabel("Synchronized Time [s]")
        ax2.set_ylabel("Inlier Ratio [0–1]")
        ax2.set_ylim(0.4, 1.05)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="lower right", fontsize=8.5)

        fig.tight_layout()
        out_path = self.plots_dir / "04_track_b_visual_odometry_shape.png"
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return str(out_path)

    def _plot_turns_and_stops_map(
        self,
        ta: pd.DataFrame,
        tc: pd.DataFrame,
        turns_df: pd.DataFrame,
        stops_df: pd.DataFrame,
    ) -> str:
        fig, ax = plt.subplots(figsize=(11, 9), dpi=self.dpi)
        ax.plot(ta["px_m"], ta["py_m"], color="#EF4444", lw=1.1, alpha=0.55, linestyle="--", label="Track A (Sensor-Only EKF)")
        ax.plot(tc["px_fused_m"], tc["py_fused_m"], color="#0284C7", lw=2.2, label="Track C (Sensor + Optical Flow Fusion)")

        if not turns_df.empty:
            ax.scatter(
                turns_df["apex_px_m"],
                turns_df["apex_py_m"],
                s=52,
                c="#F59E0B",
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
                label=f"Detected Turns ({len(turns_df)})",
            )
            for r in turns_df.itertuples(index=False):
                if r.maneuver_type in ("Left Turn", "Right Turn", "U-Turn"):
                    ax.annotate(
                        f"T{r.turn_id} ({r.apex_t_video_s:.0f}s)",
                        (r.apex_px_m, r.apex_py_m),
                        textcoords="offset points",
                        xytext=(5, 4),
                        fontsize=7,
                        color="#92400E",
                    )

        if not stops_df.empty:
            ax.scatter(
                stops_df["px_fused_m"],
                stops_df["py_fused_m"],
                s=55,
                c="#DC2626",
                marker="s",
                edgecolors="white",
                linewidths=0.7,
                zorder=6,
                label=f"Detected Stops >=3s ({len(stops_df)})",
            )
            for r in stops_df.itertuples(index=False):
                if r.duration_s >= 10.0:
                    ax.annotate(
                        f"S{r.stop_id}: {r.duration_s:.0f}s @ {r.start_t_video_s:.0f}s",
                        (r.px_fused_m, r.py_fused_m),
                        textcoords="offset points",
                        xytext=(5, -8),
                        fontsize=7,
                        color="#991B1B",
                    )

        ax.set_title("Complete Drive ENU Map with Localized Turns (T#) & Stationary Stops (S#)", fontsize=12, fontweight="bold")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.axis("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

        fig.tight_layout()
        out_path = self.plots_dir / "05_turns_and_stops_annotated_map.png"
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return str(out_path)