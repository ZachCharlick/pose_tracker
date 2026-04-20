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
    align_depth.enable:=true \
    pointcloud.enable:=true \
    depth_module.depth_profile:=640x480x30 \
    rgb_camera.color_profile:=640x480x30

RGB image is at /camera/camera/color/image_raw

To get the landmarks and visualize
## On install, need mediapipe
# sudo apt install python3-pip -y
# pip3 install mediapipe
# pip3 install "numpy<2.0.0"

source install/setup.bash
ros2 run pose_landmarker_ros pose_landmarker_node \
  --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p model_path:=/home/armlab/ros2_ws/pose_tracker/assets/pose_landmarker_heavy.task \
  -p publish_annotated_image:=true \
  -p landmarks_topic:=pose_landmarks

To get forearm pose

source install/setup.bash
ros2 run pose_landmarker_ros forearm_pose_3d_node \
  --ros-args \
  -p landmarks_topic:=pose_landmarks \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/color/camera_info \
  -p output_pose_topic:=forearm_pose_camera


NEXT ---- SETTING UP IMU w/ microROS
# for setup git clone -b humble https://github.com/micro-ROS/micro_ros_setup.git
# ros2 run micro_ros_setup create_agent_ws.sh
# ros2 run micro_ros_setup build_agent.sh
# sudo usermod -a -G dialout $USER
# newgrp dialout
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# sudo apt install ros-humble-imu-tools -y (for IMU vis plugin)
# TO FIND PORT:: ls /dev/ttyUSB*

# w/ example port
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0

# Our IMU only provides angular velocities and linear accelerations. We need a filter! Try w/ the
# ROS provided "imu_complementary_filter"

ros2 run imu_complementary_filter complementary_filter_node \
  --ros-args \
  -p use_mag:=false \
  -p publish_tf:=false \
  -r /imu/data_raw:=/imu/data \
  -r /imu/data:=/imu_filtered


How to run augmented video from ros bag

source install/setup.bash
ros2 bag play /path/to/bag_folder --clock -l

source install/setup.bash
ros2 run image_augment_ros image_augment_node \
  --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p output_topic:=/camera/camera/color/image_augmented \
  -p mode:=splotches \
  -p effect_seed:=7
# mode: splotches | flicker | distortion
# optional (splotches / dark dirt): -p splotches_count:=150 -p splotches_sigma_min:=10.0 -p splotches_sigma_max:=40.0 -p splotches_peak_darken_min:=0.06 -p splotches_peak_darken_max:=0.48 -p splotches_spawn_center_beta:=2.5
# optional (flicker): -p flicker_blackout_probability:=0.05 -p flicker_burst_frames_min:=1 -p flicker_burst_frames_max:=5
# optional (distortion): -p distortion_blur_sigma:=3.2 -p distortion_radial_k:=0.22

source install/setup.bash
ros2 run rqt_image_view rqt_image_view
# topic: /camera/camera/color/image_augmented