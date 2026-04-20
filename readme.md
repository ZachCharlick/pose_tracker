# Pose Tracker — Invariant EKF for Robust 3D Forearm Tracking

A ROS 2 pipeline that fuses a MediaPipe-based vision measurement with IMU data through a **Right-Invariant Extended Kalman Filter (RI-EKF) on SE₂(3)** to produce a robust 3D forearm pose estimate that degrades gracefully under camera occlusion, blackout, or lens degradation.

> Final project for **ROB 530: Mobile Robotics** (University of Michigan). The core motivation for choosing an InEKF over a standard EKF is that the InEKF's error dynamics on the Lie group SE₂(3) are state-independent, which gives better consistency and recovery behavior during extended periods of measurement dropout.

---

## Table of Contents

- [System Overview](#system-overview)
- [Package Layout](#package-layout)
- [Data Flow](#data-flow)
- [Mathematical Background](#mathematical-background)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Run](#run)
- [Topics and Messages](#topics-and-messages)
- [Configuration](#configuration)
- [Degradation Experiments](#degradation-experiments)
- [Design Notes and Caveats](#design-notes-and-caveats)
- [Repository Structure](#repository-structure)
- [Authors](#authors)

---

## System Overview

The pipeline estimates the 3D pose (position + orientation) of a person's forearm in a fixed world frame using two asynchronous information sources:

1. **Prediction (fast, always available):** A body-mounted IMU provides angular velocity and linear acceleration, which are propagated through the SE₂(3) motion model at IMU rate.
2. **Correction (slower, drop-prone):** An RGB-D camera produces a per-frame 3D forearm pose by running MediaPipe's Pose Landmarker on the color image and fusing the 2D landmarks with the aligned depth stream and camera intrinsics.

Ground-truth validation is provided by a motion capture system (100 Hz over ROS topics), which is used only for evaluation, not in the filter.

The pipeline's key property is that **when the camera signal is degraded or lost entirely, the filter continues to propagate its pose estimate using the IMU alone, and the RI-EKF's geometric structure keeps the covariance and mean consistent on the manifold** so that recovery is fast and well-behaved once vision returns.

## Package Layout

Four ROS 2 packages, each independently buildable:

| Package | Language | Purpose |
| --- | --- | --- |
| `pose_landmarker_interfaces` | CMake / IDL | Custom `PoseLandmark` and `PoseLandmarksStamped` messages |
| `pose_landmarker_ros` | ament_python | MediaPipe Pose Landmarker node + 3D forearm-pose node (landmarks + depth → `PoseStamped`) |
| `pose_filter` | ament_python | The RI-EKF on SE₂(3); subscribes to IMU and camera pose, publishes fused pose with covariance |
| `image_augment_ros` | ament_python | Optional image-degradation node used for the robustness experiments (splotches, flicker/blackout, defocus+distortion) |

## Data Flow

```
 RealSense color ──► (optional) image_augment_node ──► pose_landmarker_node ──► PoseLandmarksStamped
                                                                                     │
 RealSense depth ───────────────────────────────────────────────┐                    │
 RealSense info ───────────────────────────────────────────────► forearm_pose_3d_node ◄┘
                                                                 │
                                                                 ▼
                                                          forearm_pose_camera (PoseStamped)
                                                                 │
 IMU (sensor_msgs/Imu) ──────────────────────────────────────────┤
                                                                 ▼
                                                            InEKF node (pose_filter)
                                                                 │
                                                                 ▼
                                                    handPose (PoseWithCovarianceStamped)
```

The image-augment node is optional and is only inserted into the graph for the degradation experiments; in normal operation, the landmarker subscribes directly to the camera's color image.

## Mathematical Background

**State.** The filter state lives on SE₂(3), the group of extended rigid-body transformations, represented as a 5×5 matrix:

```
        ┌ R  v  p ┐
    X = │ 0  1  0 │       R ∈ SO(3), v ∈ ℝ³, p ∈ ℝ³
        └ 0  0  1 ┘
```

The corresponding error state is a 9-dimensional vector `ξ = [φ; ζ; ρ]` (rotation / velocity / position Lie-algebra increments), and the covariance `P ∈ ℝ⁹ˣ⁹` lives in the tangent space at the current state.

**Prediction.** Given a body-frame IMU reading (ω, a) over a time step dt, the node:

1. Rotates the world-frame gravity vector into the body frame and subtracts it from the accelerometer to obtain a gravity-compensated linear acceleration.
2. Builds the 9-vector `ξ = [ω·dt, a·dt, R_curr^T·v_world·dt]` and forms its `wedge` into the 5×5 Lie-algebra matrix.
3. Right-multiplies: `X ← X · exp(ξ^)`.
4. Propagates covariance with a linearized state-transition matrix `Φ = I + A·dt`, where `A` is the standard SE₂(3) continuous-time error dynamics, and `P ← Φ P Φᵀ + Q_d`, with `Q_d = Φ (Q·dt) Φᵀ`.

**Correction.** The vision measurement gives a pose (R_meas, p_meas) of the forearm in the camera frame. We form the innovation **in the current body frame**:

```
    v_φ  = log(R_state^T · R_meas)           ∈ ℝ³
    v_p  = R_state^T · (p_meas − p_state)    ∈ ℝ³
    y    = [v_φ; v_p]                        ∈ ℝ⁶
```

With a constant observation Jacobian `H_pose ∈ ℝ⁶ˣ⁹` (identity on the rotation and position blocks, zero on the velocity block), the Kalman gain, state update, and Joseph-form covariance update are standard:

```
    S = H P Hᵀ + N
    L = P Hᵀ S⁻¹
    X ← X · exp((L·y)^)
    P ← (I − L H) P (I − L H)ᵀ + L N Lᵀ
```

Because the update is applied in the body-frame tangent space and then lifted back onto the group with `exp`, orientation stays a valid rotation matrix at every step.

**Why RI-EKF instead of a standard EKF?** In a standard EKF, linearizing the orientation update around the current estimate makes the error dynamics depend on the state itself, which causes the filter to become inconsistent during large rotational excursions or long propagation intervals. On SE₂(3), the right-invariant error has autonomous (state-independent) linearized dynamics, which is exactly the regime where IMU-only propagation over a camera blackout lives. This is the key theoretical reason the InEKF recovers more cleanly than a standard EKF when vision comes back online.

For the full derivation, see the ROB 530 lecture notes on matrix Lie groups and the Invariant EKF, which this implementation follows directly.

## Prerequisites

| Component | Version |
| --- | --- |
| Ubuntu | 22.04 (Jammy) |
| ROS 2 | Humble |
| Python | 3.10 (system default on Jammy) |
| RealSense SDK | `realsense-ros` for the RealSense camera driver |

Python packages (install into the same interpreter that `ros2 run` uses):

```bash
pip install mediapipe numpy scipy opencv-python
```

System packages:

```bash
sudo apt install ros-humble-cv-bridge ros-humble-message-filters ros-humble-realsense2-camera
```

**MediaPipe model file.** The Pose Landmarker requires a `.task` model that is *not* bundled with the `mediapipe` pip package. Download it once:

```bash
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

Pass the absolute path via the `model_path` parameter when launching the node. The `heavy` variant is recommended for the best 33-landmark accuracy; `full` and `lite` variants are also available.

## Build

From the workspace root:

```bash
colcon build --symlink-install
source install/setup.bash
```

Build `pose_landmarker_interfaces` first if you hit message-import errors on a clean checkout:

```bash
colcon build --packages-select pose_landmarker_interfaces
source install/setup.bash
colcon build
source install/setup.bash
```

## Run

Each node is run independently. A typical full-pipeline launch with the RealSense driver looks like this (open each command in its own terminal after sourcing the workspace):

**1. Camera driver:**

```bash
ros2 launch realsense2_camera rs_launch.py \
    enable_color:=true enable_depth:=true align_depth.enable:=true
```

**2. MediaPipe pose landmarker:**

```bash
ros2 run pose_landmarker_ros pose_landmarker_node --ros-args \
    -p model_path:=/abs/path/to/pose_landmarker_heavy.task \
    -p image_topic:=/camera/camera/color/image_raw \
    -p publish_annotated_image:=true
```

**3. 3D forearm pose fusion:**

```bash
ros2 run pose_landmarker_ros forearm_pose_3d_node
```

By default this subscribes to `pose_landmarks`, `/camera/camera/aligned_depth_to_color/image_raw`, and `/camera/camera/color/camera_info`, and publishes `forearm_pose_camera`. The forearm is defined by landmarks 14 (right elbow) and 16 (right wrist); change `landmark_proximal` and `landmark_distal` for a different limb.

**4. IMU source:** any driver that publishes `sensor_msgs/Imu` on `imu/data`. Remap as needed.

**5. The filter:**

```bash
ros2 run pose_filter pose_filter --ros-args \
    --params-file src/pose_filter/config/config.yaml
```

**Degradation experiment.** Insert the augmenter between the camera and the landmarker by remapping the landmarker's `image_topic`:

```bash
ros2 run image_augment_ros image_augment_node --ros-args -p mode:=flicker
ros2 run pose_landmarker_ros pose_landmarker_node --ros-args \
    -p model_path:=/abs/path/to/pose_landmarker_heavy.task \
    -p image_topic:=/camera/camera/color/image_augmented
```

Modes: `splotches` (lens-dirt Gaussian darkening blobs at fixed positions), `flicker` (random multi-frame blackouts), `distortion` (Gaussian defocus + radial barrel distortion). Tune per-mode parameters via the node's declared parameters.

## Topics and Messages

**Custom messages** (`pose_landmarker_interfaces`):

- `PoseLandmark`: one landmark in MediaPipe's normalized image coordinates — `uint32 index`, `float32 x`, `float32 y`, `float32 z`, `float32 visibility`, `float32 presence`.
- `PoseLandmarksStamped`: `Header header`, `uint32 image_width`, `uint32 image_height`, `PoseLandmark[] landmarks`.

**Default topic graph:**

| Direction | Topic | Type | Publisher |
| --- | --- | --- | --- |
| pub | `pose_landmarks` | `PoseLandmarksStamped` | `pose_landmarker_node` |
| pub | `pose_image_annotated` (opt.) | `sensor_msgs/Image` | `pose_landmarker_node` |
| pub | `forearm_pose_camera` | `geometry_msgs/PoseStamped` | `forearm_pose_3d_node` |
| pub | `handPose` | `geometry_msgs/PoseWithCovarianceStamped` | `pose_filter` (5 Hz timer) |
| pub | `/camera/camera/color/image_augmented` | `sensor_msgs/Image` | `image_augment_node` |
| sub | `imu/data` | `sensor_msgs/Imu` | `pose_filter` |
| sub | `forearm_pose_camera` | `geometry_msgs/PoseStamped` | `pose_filter` |

The filter's published covariance is extracted from the 9×9 SE₂(3) covariance by pulling out the rotation and position blocks (plus their cross-correlation) and optionally rotating into the world frame. Velocity covariance is not exposed on the ROS pose message.

## Configuration

Filter tuning lives in `src/pose_filter/config/config.yaml`:

- `initial_state` — 25-element row-major 5×5 SE₂(3) matrix. Defaults to identity (zero position, zero velocity, identity rotation).
- `initial_covariance` — 81-element 9×9. Default: 0.1·I (moderate uncertainty).
- `process_noise` (Q) — 9×9 continuous-time PSD. Default: 0.001·I. This governs how much the filter trusts the IMU prediction.
- `measurement_noise` (N) — 9×9. Default: 0.01·I (one order of magnitude less trust in vision than in IMU). The correction uses only the 6×6 rotation+position sub-block.
- `measurement_jacobian` (H) — 9×9 identity (declared but overridden at runtime by the constant `H_pose ∈ ℝ⁶ˣ⁹` used in the correction step).
- `gravity_vector` — measured from a stationary IMU reading. The default `[-0.3, -0.116, -10.222]` is a placeholder; **you must replace this with a reading from your own static IMU calibration** or the filter will diverge.
- `imu_accel_includes_gravity` — set `true` when the accelerometer publishes specific force (the common case), `false` if the driver has already compensated for gravity.

Re-tune Q and N for your sensor suite. As a rough rule: raise N if the camera pose is jittery; raise Q if the IMU integration is drifting faster than expected during blackouts.

## Degradation Experiments

The planned evaluation compares the RI-EKF against a standard EKF baseline under three conditions:

1. **Nominal tracking** — unimpeded camera, moderate motion. Measure steady-state RMS error against mocap and check NEES consistency.
2. **Camera blackout** — 2–5 second full dropouts (`mode:=flicker` with a long burst) combined with aggressive forearm motion. Metrics: recovery error at the moment of vision return, convergence time back to the nominal error band, NEES during the blackout interval.
3. **Lens degradation** — `splotches` and `distortion` modes to emulate partial sensor failure rather than complete dropout. Measures filter behavior when vision is available but noisy/biased rather than missing.

The blackout experiment is where the InEKF is expected to most clearly outperform a standard EKF, because the state-independent error dynamics prevent the runaway linearization errors that a standard EKF accumulates during extended IMU-only propagation.

## Design Notes and Caveats

- **Forearm frame convention.** `forearm_pose_3d_node` builds a right-handed frame with `+x` pointing from the proximal landmark (elbow, 14) to the distal landmark (wrist, 16). The `y` axis is chosen via a reference cross product, so the *roll about the bone axis is under-determined* from two landmarks alone — the filter is essentially tracking position and bone orientation, not fingertip roll. Adding a third landmark (or a wrist IMU) would fully observe orientation.
- **Depth-to-color mapping.** The 3D node assumes the depth stream has been aligned to color (RealSense's `align_depth.enable:=true`); if you use unaligned depth, the `_map_uv_to_depth` simple scaling will be wrong and you will need a proper extrinsic reprojection.
- **`wedge` is defined twice** in `InEKF.py` — the second definition shadows the first. Both produce the same 5×5 matrix; the duplicate is a stylistic leftover from refactoring and is not a bug.
- **Gravity vector must be measured.** The default in `config.yaml` is a placeholder. Record ~1 s of IMU accelerometer samples with the sensor stationary and flat, average them, and use that vector (with the correct sign for your convention).
- **Timing.** `InEKF.prediction` uses the ROS clock, not the IMU message timestamp, to compute `dt`. On a wall-clock-synced system this is fine; if you play back bagfiles with `--clock`, remember that `dt` will reflect sim time only if the node is started in sim-time mode.
- **Covariance frame.** The 9×9 `P` lives in the body-frame tangent space. `get_ros_pose_covariance` optionally rotates the position/rotation blocks into the world frame before publishing — useful for rviz ellipsoid visualization, but downstream consumers should be aware the raw `P` inside the filter is body-framed.

## Repository Structure

```
src/
├── pose_landmarker_interfaces/          # Custom ROS 2 messages
│   ├── msg/
│   │   ├── PoseLandmark.msg
│   │   └── PoseLandmarksStamped.msg
│   ├── CMakeLists.txt
│   └── package.xml
│
├── pose_landmarker_ros/                 # Vision nodes
│   ├── pose_landmarker_ros/
│   │   ├── pose_landmarker_node.py      # Image → 33 landmarks (MediaPipe Tasks API)
│   │   └── forearm_pose_3d_node.py      # Landmarks + depth → 3D PoseStamped
│   ├── package.xml
│   └── setup.py
│
├── pose_filter/                         # RI-EKF on SE_2(3)
│   ├── pose_filter/
│   │   └── InEKF.py                     # Prediction, correction, publishing
│   ├── config/
│   │   └── config.yaml                  # Initial state, Q, N, gravity
│   ├── package.xml
│   └── setup.py
│
└── image_augment_ros/                   # Experimental-degradation node
    ├── image_augment_ros/
    │   └── image_augment_node.py        # splotches | flicker | distortion
    ├── package.xml
    └── setup.py
```

## Authors

University of Michigan, ROB 530 Mobile Robotics — Winter 2026.

- **Austen Goddu** — ajgoddu@umich.edu
- **Zachary Charlick** — zsc@umich.edu
- **Thomas Joseph** — trjosep@umich.edu
- **Matthew Pacas-McCarthy** — mpacas@umich.edu
