# Vision-Inertial Sensor Fusion for Vehicle Trajectory Estimation

## 1. Project Overview & Objectives

### 1.1 Overview
**Vision-Inertial Sensor Fusion for Vehicle Trajectory Estimation** is an end-to-end multi-sensor localization, kinematics reconstruction, and driving maneuver detection system designed for unstructured urban road environments. Using low-cost consumer hardware—a smartphone's Inertial Measurement Unit (IMU) and GPS receiver paired with a front-facing monocular dashcam—the pipeline reconstructs smooth, metric 2D trajectories ($p_x, p_y$), instantaneous vehicle speed ($v$), heading ($\psi$), and online sensor bias estimates ($b_a, b_g$).

To address asynchronous sensor rates, noisy MEMS inertial drift, dynamic traffic occlusions, and intermittent GPS signal dropouts (such as under-flyover urban canyons), the system implements a **6-State Planar Extended Kalman Filter (EKF)** fused with **RANSAC-filtered Monocular Optical Flow Visual Odometry**, **Zero-Velocity Updates (ZUPT)**, and a **Rauch–Tung–Striebel (RTS) Backward Smoother**.

### 1.2 Key Capabilities & Quantitative Summary
* **Asynchronous Multi-Rate Sensor Fusion:** Chronologically merges high-rate IMU accelerations/angular rates, `30 FPS` visual odometry measurements, and `1 Hz` GPS fixes via a unified priority event queue.
* **Online Calibration & Scale Adaptation:** Automatically removes static gravity vectors, estimates stationary gyroscope offsets, and continuously calibrates monocular visual scale ($\hat{s}$) when GPS is healthy.
* **GPS Outage & Occlusion Resilience:** Bridges prolonged urban GPS blackouts (e.g., a `55.1 s` under-flyover occlusion) using scale-calibrated visual speed, visual yaw-rate updates, ZUPT dead-stop constraints, and RTS backward smoothing—eliminating post-outage position snaps.
* **Traffic State & Maneuver Classification:** Automatically segments and localizes **Stationary Stops ($\ge 3\text{s}$)**, **Slow-Moving Crawls**, and **Left / Right / U-Turns** synchronized with dashcam video playback.

| Metric / Parameter | Benchmark Value (`drive_20`) |
| :--- | :--- |
| **Total Drive Duration** | `34.0 min` (`2,038 s` / `60,630` video frames) |
| **Total EKF Asynchronous Steps** | `66,026` steps |
| **Reconstructed Path Length (Track C)** | `8,386.17 m` (`8.39 km`) |
| **Mean 1-$\sigma$ Position Uncertainty** | `2.79 m` (reduced from `48.78 m` un-smoothed) |
| **Max Natural Under-Flyover GPS Blackout** | `55.1 s` (`t = 1725.9 s – 1781.0 s`) |
| **Outage Drift Reduction (Track C vs. Track A)** | `72.8%` peak drift reduction (`72.2%` RMSE reduction in Holdout #3) |
| **Detected Maneuvers & Traffic Events** | `33` Localized Turns (`T1–T33`), `50` Stationary Stops (`S1–S50`) |

---

## 2. System Architecture & Multi-Track Estimation Design

### 2.1 End-to-End Pipeline Workflow

```
  [Smartphone IMU Logs]           [Smartphone GPS Stream]           [Dashcam Video (1080p)]
  Accr (~17 Hz), Gyro (~5.6 Hz)   1 Hz Lat/Lon/Speed/Course         30 FPS Front-Facing Camera
               │                             │                                 │
               ▼                             ▼                                 ▼
  ┌──────────────────────────────────────────────────────┐      ┌──────────────────────────────┐
  │ Phase 1: Preprocessing, Calibration & Sync           │      │ Phase 2 (Track B): Optical   │
  │ • 30 FPS Frame-Index Alignment                       │      │ Flow Visual Odometry         │
  │ • Gravity Removal & Vehicle Frame Rotation           │      │ • Shi-Tomasi + LK Flow       │
  │ • WGS84 -> Local ENU Metric Projection               │      │ • RANSAC Essential Matrix    │
  │ • Stationary Gyro Bias & ZUPT Flag Extraction        │      │ • Visual Yaw & Unscaled Speed│
  └───────────────────┬──────────────────────┬───────────┘      └──────────────┬───────────────┘
                      │ Synced IMU + GPS     │ Calibrated IMU + GPS            │ Visual Yaw & Speed
                      ▼                      ▼                                 ▼
  ┌──────────────────────────────────┐      ┌──────────────────────────────────────────────────┐
  │ Phase 3 (Track A): Sensor-Only   │      │ Phase 4 (Track C): Asynchronous Fusion EKF +     │
  │ EKF Baseline                     │      │ RTS Backward Smoother                            │
  │ • Kinematic GPS Outlier Filter   │      │ • Priority Event Queue (IMU + Visual + GPS)      │
  │ • 6-State Planar EKF + ZUPT      │      │ • Online Median Visual Scale Calibration         │
  │ • Benchmark: ~972.8m flyover drift│     │ • Outage Bridging + Chi-Square Gating            │
  └───────────────────┬──────────────┘      │ • RTS Backward Smoother (2.79m Mean Pos Std)     │
                      │                     └────────────────────────┬─────────────────────────┘
                      │ Baseline Trajectory                          │ Smoothed Fused Trajectory
                      └──────────────────────┬───────────────────────┘
                                             ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Phase 5: Maneuver Detection, Quantitative Evaluation & Telemetry Rendering                 │
  │ • State & Maneuver Classifier: Stationary Stops (>=3s), Slow-Moving Crawls, Left/Right/U   │
  │ • Multi-Track Evaluation: Track A vs Track B vs Track C, Simulated GPS Holdout RMSE        │
  │ • Outputs: 2D ENU Maps, Kinematics/Event CSV Logs, Side-by-Side Dashcam Telemetry Video    │
  └────────────────────────────────────────────────────────────────────────────────────────────┘
```

*(Full visual workflow diagram available in `btp_workflow_diagram.png`.)*

### 2.2 Comparative Track Architecture
To rigorously validate the contribution of each sensor modality, the pipeline implements three parallel estimation tracks:
1. **Track A — Sensor-Only Planar EKF (`run_sensor_only.py`):** Fuses kinematic-filtered GPS fixes with IMU dead-reckoning and ZUPT constraints. Serves as the inertial-GPS baseline and demonstrates severe quadratic drift (`~972.8 m`) during prolonged GPS outages.
2. **Track B — Monocular Optical Flow VO (`run_optical_flow_only.py`):** Estimates frame-to-frame camera rotation and translation direction purely from dashcam imagery using Lucas-Kanade Optical Flow and RANSAC Essential Matrix decomposition. Demonstrates visual heading accuracy alongside monocular scale ambiguity and open-loop drift.
3. **Track C — Proposed Vision-Inertial-GPS Fusion EKF + RTS Smoother (`run_sensor_fusion.py`):** Combines Track A and Track B inside an asynchronous 6-state EKF with online median visual scale calibration, $\chi^2$ Mahalanobis gating, consistent-pair re-acquisition, and a Rauch–Tung–Striebel backward smoothing pass.

---

## 3. Repository Structure

```text
Vision-Inertial-Sensor-Fusion-for-Vehicle-Trajectory-Estimation/
├── configs/
│   ├── preprocessing.yaml          # Sensor sync tolerances, ENU origin & ZUPT thresholds
│   ├── optical_flow.yaml           # Feature detector, LK pyramid & RANSAC parameters
│   ├── sensor_only_ekf.yaml        # Track A baseline noise & GPS gating config
│   ├── fusion_ekf.yaml             # Track C fusion noise, scale EMA, gating & ZUPT config
│   └── evaluation.yaml             # Turn/stop thresholds, holdout windows & video render settings
├── data/
│   ├── raw/                        # Raw dashcam MP4 videos & smartphone sensor CSV logs
│   └── processed/                  # Time-synced sensor streams & Optical Flow VO outputs
├── scripts/
│   ├── run_preprocessing.py        # Phase 1: Multi-sensor ingestion, calibration & sync
│   ├── run_optical_flow_only.py    # Phase 2: Track B monocular optical flow odometry
│   ├── run_sensor_only.py          # Phase 3: Track A sensor-only EKF baseline
│   ├── run_sensor_fusion.py        # Phase 4: Track C asynchronous fusion EKF + RTS smoother
│   └── run_evaluation_and_plots.py # Phase 5: Maneuver detection, metrics & video overlay
├── src/
│   ├── preprocessing/              # Timestamp alignment, gravity removal, ENU projection
│   ├── sensors/                  # Kinematic GPS outlier filter & IMU packet builders
│   ├── vision/                   # Shi-Tomasi + LK tracker & RANSAC Essential Matrix solver
│   ├── ekf/                      # Planar vehicle motion model, EKF core & RTS smoother
│   ├── evaluation/               # Stop/Slow-Moving/Turn detectors & RMSE holdout evaluators
│   └── visualization/            # 2D ENU trajectory plots & side-by-side video renderer
├── outputs/                        # Generated plots, event CSVs & annotated telemetry videos
├── main.py                         # Unified CLI orchestrator across all 5 phases
└── requirements.txt                # Python dependencies
```

---

## 4. Sensor Modalities & Dataset Overview

### 4.1 Input Streams & Sampling Rates
* **Smartphone Accelerometer (`accr`):** Tri-axial specific force measurements sampled at `~17 Hz` in the phone body frame.
* **Smartphone Gyroscope (`gyro`):** Tri-axial angular velocity measurements sampled at `~5.6 Hz` in the phone body frame.
* **Smartphone GPS Receiver (`gps`):** Geographic coordinates (latitude, longitude), ground speed, course over ground, and horizontal accuracy sampled at `1.0 Hz`.
* **Front-Facing Dashcam (`video`):** `1080p` monocular RGB video stream recorded at `30.0 FPS`, providing visual road texture and traffic context.

### 4.2 Benchmark Drive Summary (`drive_20`)
The primary benchmark recording (`drive_20`) captures a `34.0-minute` (`8.39 km`) urban drive in dense Indian traffic conditions featuring:
* `1,753` raw `1 Hz` GPS fixes with zero raw multipath velocity spikes (`> 35 m/s`), making it ideal for ground-truth trajectory validation.
* Multiple stop-and-go traffic intersections (`50` stationary stops $\ge 3\text{s}$), slow-moving traffic crawls, and `33` distinct turns (including tight U-turns).
* Two natural urban GPS occlusion events caused by driving beneath elevated concrete flyovers—including a continuous **`55.1 s` GPS blackout (`t = 1725.9 s – 1781.0 s`)** and a **`37.0 s` dropout (`t = 1000.0 s – 1037.0 s`)**.

---

## 5. Detailed Phase-Wise Pipeline & Implementation

### 5.1 Phase 1: Multi-Sensor Ingestion, Calibration & Frame Synchronization (`run_preprocessing.py`)
* **Timeline Alignment:** Parses raw smartphone sensor logs and aligns asynchronous IMU and GPS timestamps onto the `30 FPS` dashcam frame timeline (`frame_idx`, `t_sync_s`).
* **Local ENU Metric Projection:** Projects geodetic WGS84 coordinates ($\text{lat}, \text{lon}$) onto a local tangent plane in **East-North-Up (ENU)** meters relative to the trip origin.
* **Gravity Removal & Frame Alignment:** Identifies initial/extended stationary windows to estimate the static gravity vector and rotates smartphone body accelerations into the vehicle frame (`a_forward_m_s2`, `a_lateral_m_s2`).
* **Stationary Gyro Bias & ZUPT Extraction:** Computes static gyroscope offset biases (`gyro_bias_sensor_rad_s`) during zero-motion intervals and flags low-variance stationary windows (`is_stationary = True`) for downstream ZUPT updates.

### 5.2 Phase 2: Monocular Visual Odometry via Sparse Optical Flow — Track B (`run_optical_flow_only.py`)
* **Feature Detection & Tracking:** Detects salient road and structural corners via **Shi-Tomasi** (`goodFeaturesToTrack`) and tracks them across consecutive frames using **Pyramidal Lucas-Kanade Optical Flow**.
* **Geometric Outlier Rejection:** Fits a 5-point **Essential Matrix ($E$) with RANSAC** to separate static background geometry (road surface, poles, buildings) from dynamic foreground traffic (cars, buses, two-wheelers). Frames with `inlier_ratio < 0.90` are flagged as visually occluded.
* **Visual Kinematics Extraction:** Decomposes $E$ into relative rotation ($R$) and unit translation direction ($\hat{t}$) to output frame-to-frame **visual yaw-rate** ($\omega_{\text{vis}}$), **unscaled forward speed**, and a **visual dead-stop indicator** (`median_flow_px < 0.75`).

### 5.3 Phase 3: Sensor-Only Planar EKF Baseline & Kinematic GPS Filtering — Track A (`run_sensor_only.py`)
* **Kinematic GPS Pre-Filtering (`GPSProcessor`):** Rejects physically implausible GPS jumps (`implied_speed > 32 m/s` or single-step jumps `> 45 m`) before filter ingestion.
* **Planar EKF + ZUPT Baseline:** Propagates vehicle state using calibrated IMU forward acceleration and yaw-rate, correcting drift with `1 Hz` GPS fixes and applying Zero-Velocity Updates (`z_vel = 0.0 m/s`) whenever `is_stationary` is active. Establishes the baseline error curve (`972.8 m` peak drift during the `55.1 s` flyover blackout).

### 5.4 Phase 4: Asynchronous Vision-Inertial Fusion EKF & RTS Backward Smoother — Track C (`run_sensor_fusion.py`)
* **Chronological Priority Queue:** Interleaves IMU (`priority 0`), Visual (`priority 1`), and GPS (`priority 2`) measurement packets in exact timestamp order.
* **Online Median Visual Scale Calibration:** Continuously estimates metric camera scale ($\hat{s} = v_{\text{EKF}} / v_{\text{vis}}$) when GPS is fresh (`gps_age <= 2.0 s`) and maintains a rolling median buffer (`window = 40`) to prevent noisy scale spikes.
* **Outage Bridging Mode:** When `gps_age > 2.5 s`, fuses scale-calibrated visual speed ($v_{\text{metric}} = \hat{s}_{\text{median}} \cdot v_{\text{vis}}$), high-inlier visual yaw-rate, and visual/IMU ZUPT constraints to bound dead-reckoning drift.
* **$\chi^2$ Gating, Consistent-Pair Re-acquisition & RTS Smoothing:** Validates returning GPS fixes using Mahalanobis gating plus a 2-fix spatial consistency check (`<= 35 m` apart), followed by a full-trajectory **Rauch–Tung–Striebel (RTS) backward smoothing pass** that eliminates post-outage position jumps.

### 5.5 Phase 5: Maneuver Detection, Quantitative Evaluation & Telemetry Rendering (`run_evaluation_and_plots.py`)
* **Traffic State & Maneuver Classification:** Processes the smoothed Track C trajectory to detect **Stationary Stops ($\ge 3.0\text{s}$)**, **Slow-Moving Crawls**, and **Left / Right / U-Turns** with frame-synchronized confirmation offsets.
* **Quantitative Holdout Benchmarking:** Simulates artificial `30 s` GPS blackouts on healthy road segments to compute comparative RMSE between Track A and Track C.
* **Visualization & Video Overlay:** Generates 5 diagnostic figures, structured CSV logs (`detected_turns.csv`, `detected_stops.csv`), and a synchronized side-by-side dashcam + live ENU map telemetry video.

---

## 6. Core Algorithms, Mathematical Formulation & Implementation Details

### 6.1 Coordinate Transformations & Inertial Calibration
Geodetic coordinates $(\phi, \lambda)$ are converted to local metric East-North coordinates $(p_x, p_y)$ relative to reference origin $(\phi_0, \lambda_0)$:

$$p_x = R_E (\lambda - \lambda_0) \cos(\phi_0), \qquad p_y = R_N (\phi - \phi_0)$$

During stationary initialization windows ($\mathcal{S}_0$), static gyroscope bias $\mathbf{b}_{g,0}$ and gravity alignment are estimated as:

$$\mathbf{b}_{g,0} = \frac{1}{|\mathcal{S}_0|} \sum_{k \in \mathcal{S}_0} \boldsymbol{\omega}_{\text{raw}, k}, \qquad \omega_{\text{clean}, k} = \omega_{z,\text{raw}, k} - b_{g,z,0}$$

### 6.2 Computer Vision Pipeline (Track B)
Given tracked 2D point correspondences $(\mathbf{u}_{k-1}, \mathbf{u}_k)$ normalized by camera intrinsic matrix $\mathbf{K}$, the epipolar constraint is solved via 5-point RANSAC:

$$\tilde{\mathbf{x}}_k^\top \mathbf{E} \tilde{\mathbf{x}}_{k-1} = 0, \qquad \mathbf{E} = [\hat{\mathbf{t}}]_\times \mathbf{R}$$

Frames are accepted into Track C only when the geometric inlier ratio satisfies $\eta_{\text{inlier}} \ge 0.90$. Camera yaw-rate is extracted from the incremental rotation matrix $\mathbf{R}$ over frame interval $\Delta t_{\text{cam}}$:

$$\omega_{\text{vis}} = \frac{\Delta \psi(\mathbf{R})}{\Delta t_{\text{cam}}}$$

### 6.3 6-State Planar Vehicle Motion Model & Multi-Rate EKF
The vehicle state vector $\mathbf{x}_k \in \mathbb{R}^6$ and control input $\mathbf{u}_k = [a_{\text{fwd}}, \omega_z]^\top$ are defined as:

$$\mathbf{x}_k = \begin{bmatrix} p_x & p_y & v & \psi & b_a & b_g \end{bmatrix}^\top$$

Over time step $\Delta t$, the discrete-time non-linear state transition $f(\mathbf{x}_k, \mathbf{u}_k, \Delta t)$ is:

$$\begin{aligned}
p_{x,k+1} &= p_{x,k} + v_k \cos(\psi_k)\,\Delta t \\
p_{y,k+1} &= p_{y,k} + v_k \sin(\psi_k)\,\Delta t \\
v_{k+1} &= \text{clip}\Big(v_k + (a_{\text{fwd},k} - b_{a,k})\,\Delta t,\; 0,\; v_{\max}\Big) \\
\psi_{k+1} &= \text{wrap}_{[-\pi,\pi]}\Big(\psi_k + (\omega_{z,k} - b_{g,k})\,\Delta t\Big) \\
b_{a,k+1} &= b_{a,k} e^{-\Delta t / \tau_a} \\
b_{g,k+1} &= b_{g,k}
\end{aligned}$$

with covariance propagation $\mathbf{P}_{k+1|k} = \mathbf{F}_k \mathbf{P}_{k|k} \mathbf{F}_k^\top + \mathbf{Q}_k$, where $\mathbf{F}_k = \frac{\partial f}{\partial \mathbf{x}}\big|_{\hat{\mathbf{x}}_{k|k}}$.

Asynchronous measurement updates apply standard EKF correction equations with specialized observation models:
* **GPS Position:** $\mathbf{z}_{\text{gps}} = [p_{x,\text{gps}}, p_{y,\text{gps}}]^\top$, $\mathbf{H}_{\text{gps}} = [\mathbf{I}_2 \;\; \mathbf{0}_{2 \times 4}]$.
* **ZUPT (IMU or Visual Dead-Stop):** $z_{\text{zupt}} = 0.0\text{ m/s}$, $\mathbf{H}_{\text{vel}} = [0, 0, 1, 0, 0, 0]$.
* **Visual Yaw-Rate Bias Observation:** $z_{\omega,\text{vis}} = \omega_{z,k} - b_{g,k}$, $\mathbf{H}_{\text{vis},b_g} = [0, 0, 0, 0, 0, -1]$.
* **Outage Visual Speed:** $z_{v,\text{vis}} = \hat{s}_{\text{median}} \cdot v_{\text{vis}}$, $\mathbf{H}_{\text{vel}} = [0, 0, 1, 0, 0, 0]$.

### 6.4 Online Median Visual Scale Calibration & Outage Bridging Algorithm
During healthy GPS reception ($t - t_{\text{last\_gps}} \le 2.0\text{ s}$) and forward motion ($v_k > 3.5\text{ m/s}$, $v_{\text{vis}} > 0.30$):

$$s_{\text{inst}} = \text{clip}\left(\frac{v_k}{v_{\text{vis}}},\; s_{\min},\; s_{\max}\right), \qquad \hat{s}_k = (1 - \alpha)\hat{s}_{k-1} + \alpha s_{\text{inst}}$$

Each updated $\hat{s}_k$ is appended to a circular buffer $\mathcal{B}_s$ of length $40$. During GPS blackouts ($t - t_{\text{last\_gps}} > 2.5\text{ s}$), the filter uses $\hat{s}_{\text{outage}} = \text{median}(\mathcal{B}_s)$ to supply metric velocity measurements, preventing single-frame scale outliers from corrupting dead-reckoning distance.

### 6.5 Outlier Rejection & Rauch–Tung–Striebel (RTS) Backward Smoother
1. **$\chi^2$ Mahalanobis Gating:** Each GPS innovation $\boldsymbol{\nu}_k = \mathbf{z}_{\text{gps}} - \hat{\mathbf{x}}_{k|k-1}[0:2]$ is tested against $\gamma_{\text{gps}} = 18.5$:

$$d_M^2 = \boldsymbol{\nu}_k^\top \mathbf{S}_k^{-1} \boldsymbol{\nu}_k \le \gamma_{\text{gps}}, \qquad \mathbf{S}_k = \mathbf{H}_{\text{gps}} \mathbf{P}_{k|k-1} \mathbf{H}_{\text{gps}}^\top + \mathbf{R}_{\text{gps}}$$

2. **Consistent-Pair Re-acquisition:** After prolonged outages (`consec_rejects >= 3`), forced re-acquisition is permitted only if $\|\boldsymbol{\nu}_k\|_2 \le 260\text{ m}$ **and** two consecutive rejected fixes lie within `35.0 m` of each other—preventing single-fix multipath snaps.
3. **RTS Backward Smoothing Pass:** After forward filtering ($k = 0 \dots N-1$), states and covariances are smoothed backward ($k = N-2 \dots 0$):

$$\mathbf{C}_k = \mathbf{P}_{k|k} \mathbf{F}_{k+1}^\top \mathbf{P}_{k+1|k}^{-1}$$

$$\hat{\mathbf{x}}_{k|N} = \hat{\mathbf{x}}_{k|k} + \mathbf{C}_k \Big(\hat{\mathbf{x}}_{k+1|N} - \hat{\mathbf{x}}_{k+1|k}\Big), \qquad \mathbf{P}_{k|N} = \mathbf{P}_{k|k} + \mathbf{C}_k \Big(\mathbf{P}_{k+1|N} - \mathbf{P}_{k+1|k}\Big)\mathbf{C}_k^\top$$

### 6.6 Traffic State & Maneuver Classification Algorithms
* **ZUPT Stationary Flag Extraction:** Marks `is_stationary = True` when $|\omega_z| < 0.022\text{ rad/s}$, $|a_{\text{fwd}}| < 0.45\text{ m/s}^2$, and rolling acceleration standard deviation $< 0.18\text{ m/s}^2$ for at least $0.80\text{ s}$.
* **Stop vs. Slow-Moving Classification:** Segments where smoothed speed $v_{k|N} < 0.35\text{ m/s}$ continuously for $\ge 3.0\text{ s}$ are logged as **Stationary Stops (`S#`)**; non-stationary intervals with $0.35\text{ m/s} \le v_{k|N} < 2.2\text{ m/s}$ in traffic are classified as **Slow Moving** without creating duplicate stop markers.
* **Turn Classification (`T#`):** Integrates smoothed yaw-rate $\omega_{k|N}$ over contiguous turning windows ($|\omega_{k|N}| > 0.08\text{ rad/s}$). Net heading changes $|\Delta \psi| \in [35^\circ, 135^\circ]$ are labeled **Left / Right Turns**, while $|\Delta \psi| > 135^\circ$ are classified as **U-Turns**.

---

## 7. Experimental Results & Performance Analysis

### 7.1 Full-Drive Trajectory & Kinematics Comparison (Track A vs. Track B vs. Track C)
Across the `34.0-minute` (`8.39 km`) urban test drive (`drive_20`):
* **Track A (Sensor-Only EKF):** Tracks open-sky segments accurately (`~4.5 m` nominal std) but suffers severe trajectory divergence (`972.8 m` peak drift) during natural under-flyover GPS blackouts.
* **Track B (Optical Flow Only):** Captures local turn geometry (`Sim(2)` aligned scale `s = 0.48`) but exhibits open-loop orientation drift over `34 minutes` due to uncalibrated scale and heavy traffic occlusions.
* **Track C (Proposed Fusion + RTS Smoother):** Reconstructs a closed-loop trajectory (`8,386.17 m` total length, `1,409 / 1,505` GPS updates accepted, `25,030` visual yaw updates, `4,703` outage visual speed updates, `7,201` ZUPT steps) with a **mean 1-$\sigma$ position uncertainty of `2.79 m`** and zero post-outage discontinuities.

### 7.2 GPS Outage & Simulated Holdout Drift Evaluation
* **Natural Under-Flyover Occlusion (`t = 1715 s – 1790 s`, `55.1 s` blackout):** Track C reduces peak lateral/longitudinal drift by **`72.8%`** compared to Track A dead-reckoning, and RTS smoothing seamlessly bridges the exit onto the service road.
* **Simulated 30-Second Blind GPS Holdout Benchmarks:**

| Holdout Window | Scenario | Track A RMSE (Blind IMU) | Track C RMSE (Blind IMU + Vision) | Error Reduction |
| :--- | :--- | :---: | :---: | :---: |
| **Holdout #1 (`1250–1280 s`)** | Curved arterial road | `55.83 m` | **`50.25 m`** | `10.0%` better |
| **Holdout #3 (`1800–1830 s`)** | Post-flyover straightaway | `164.14 m` | **`45.69 m`** | **`72.2%` reduction** |

### 7.3 Localized Traffic State & Maneuver Detection Results
* **Turns (`detected_turns.csv`):** `33` distinct turning maneuvers (`T1–T33`) accurately localized on the ENU map, including sharp U-turns at major intersections.
* **Stops (`detected_stops.csv`):** `50` stationary stops (`S1–S50`, duration $\ge 3.0\text{ s}$) cleanly separated from slow-moving traffic crawls, with zero false triggers during prolonged red-light halts.

---

## 8. Output Artifacts & Visualizations

Running the complete pipeline populates `outputs/<recording_id>/` with the following deliverables:
1. **`01_trajectory_comparison_2d.jpg`:** Full-drive local ENU trajectory comparison (Raw GPS, Track A, Track C, Start/End markers, Turns, and U-Turns) alongside a zoomed-in view of the `55.1 s` under-flyover GPS blackout.
2. **`02_kinematics_and_turns.jpg`:** 4-panel synchronized time-series plot showing Fused Speed, Visual vs. Fused Yaw-Rate (with shaded turn windows), Online Visual Scale ($\hat{s}$), and 1-$\sigma$ Position Uncertainty.
3. **`03_gps_holdout_blackout_drift.jpg`:** Side-by-side spatial comparison of blind Track A vs. Track C trajectories against held-out ground-truth GPS across 30-second simulated blackouts.
4. **`04_track_b_visual_odometry_shape.jpg`:** `Sim(2)`-aligned monocular Optical Flow trajectory shape and frame-by-frame RANSAC Essential Matrix inlier ratio timeline.
5. **`05_turns_and_stops_annotated_map.jpg`:** High-resolution ENU map annotated with numbered turn markers (`T1–T33`) and stationary stop markers (`S1–S50`).
6. **Event & Trajectory CSV Logs:** `detected_turns.csv`, `detected_stops.csv`, and frame-aligned fused kinematics tables.
7. **Side-by-Side Telemetry Video (`*.mp4`):** Synchronized `30 FPS` dashcam playback paired with a live animated ENU trajectory map, real-time kinematic readouts, and dynamic **Stop**, **Slow Moving**, and **Turn** event banners.

---

## 9. Installation, Configuration & Reproduction Guide

### 9.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/Vision-Inertial-Sensor-Fusion-for-Vehicle-Trajectory-Estimation.git
cd Vision-Inertial-Sensor-Fusion-for-Vehicle-Trajectory-Estimation

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 9.2 Filter & Threshold Configuration (`configs/fusion_ekf.yaml`)
Key parameters governing Track C sensor fusion, outlier rejection, and ZUPT behavior:
* `gyro_bias_walk_std: 0.0002` & `visual_yaw_rate_std_base: 0.35`: Prevents passing traffic optical flow from corrupting pre-calibrated gyroscope bias during long GPS outages.
* `reacquire_max_jump_m: 260.0` & `max_consecutive_rejects: 3`: Enables smooth post-flyover GPS re-locking when paired with the 2-fix spatial consistency check (`<= 35.0 m`).
* `visual_min_inlier_ratio: 0.90` & `visual_dead_stop_flow_px: 0.75`: Filters dynamic traffic outliers and triggers visual ZUPT at dead stops.

### 9.3 Step-by-Step Execution Commands
Run all 5 phases sequentially for any recording (e.g., `drive_20` or `drive_25`):

```bash
# Phase 1: Preprocessing, Calibration & 30 FPS Synchronization
python scripts/run_preprocessing.py --recording drive_20

# Phase 2: Track B — Monocular Optical Flow Visual Odometry
python scripts/run_optical_flow_only.py --recording drive_20

# Phase 3: Track A — Sensor-Only Planar EKF Baseline
python scripts/run_sensor_only.py --recording drive_20

# Phase 4: Track C — Asynchronous Fusion EKF + RTS Backward Smoother
python scripts/run_sensor_fusion.py --recording drive_20

# Phase 5: Maneuver Detection, Quantitative Evaluation & Video Rendering
python scripts/run_evaluation_and_plots.py --recording drive_20
```

---

## 10. Limitations & Future Work
* **Monocular Scale Observability During Cold Starts:** If a trip begins inside an underground parking structure with zero initial GPS fixes, visual scale $\hat{s}$ relies on the configured prior (`1.75 m/unit`) until the first valid GPS lock is acquired. Incorporating known camera mounting height above the ground plane (`homography ground-plane constraint`) can provide immediate metric scale without GPS.
* **3D Terrain & Pitch Compensation:** The current planar motion model assumes negligible road grade; extending the state vector to 3D orientation (`ESKF` / Lie-group `SE(3)` formulation) will improve vertical elevation tracking on steep flyovers and hilly terrain.
* **Deep Feature / Semantic Masking:** Replacing sparse Shi-Tomasi corners with learned keypoint descriptors and semantic vehicle masking will further improve visual odometry robustness during night driving and heavy rain.