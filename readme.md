# Pose Tracker — SE(3) Invariant EKF for Robust 3D Forearm Tracking

A ROS 2 pipeline that fuses a MediaPipe-based vision measurement with IMU data through an **Invariant Extended Kalman Filter (InEKF) on SE(3)**: **gyro-only** propagation on the group plus a **left-invariant** vision correction with adjoint-weighted measurement noise. The result is a robust 3D forearm pose estimate that degrades gracefully under camera occlusion, blackout, or lens degradation.

> Final project for **ROB 530: Mobile Robotics** (University of Michigan). The core motivation for choosing an InEKF over a standard EKF is that the state and covariance live on **SE(3)** and **𝔰𝔢(3)** rather than in a Euclidean parameterization of pose, which keeps uncertainty geometrically meaningful and recovery well-behaved during extended periods of vision dropout while the filter runs **IMU-only** (orientation from the gyro; accelerometer is not used for translation in prediction).

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

1. **Prediction (fast, always available):** A body-mounted IMU provides angular velocity (and linear acceleration, which the filter does not use for propagation). **Only the gyro** drives the **SE(3)** prediction at IMU rate; translation is not integrated from the accelerometer (see [Mathematical Background](#mathematical-background)).
2. **Correction (slower, drop-prone):** An RGB-D camera produces a per-frame 3D forearm pose by running MediaPipe's Pose Landmarker on the color image and fusing the 2D landmarks with the aligned depth stream and camera intrinsics.

Ground-truth validation is provided by a motion capture system (100 Hz over ROS topics), which is used only for evaluation, not in the filter.

The pipeline's key property is that **when the camera signal is degraded or lost entirely, the filter continues to propagate orientation using the gyro alone, and the InEKF's SE(3) structure keeps the mean on the group and the covariance on 𝔰𝔢(3)** so that recovery is fast and well-behaved once vision returns.

## Package Layout

Four ROS 2 packages, each independently buildable:

| Package | Language | Purpose |
| --- | --- | --- |
| `pose_landmarker_interfaces` | CMake / IDL | Custom `PoseLandmark` and `PoseLandmarksStamped` messages |
| `pose_landmarker_ros` | ament_python | MediaPipe Pose Landmarker node + 3D forearm-pose node (landmarks + depth → `PoseStamped`) |
| `pose_filter` | ament_python | InEKF on **SE(3)** (6×6 covariance); subscribes to IMU and camera pose, publishes fused pose with covariance |
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

We use an **Invariant Extended Kalman Filter on SE(3)** with a compact pose matrix, a **6×6** covariance on the Lie algebra **𝔰𝔢(3)**, **gyro-only prediction** (translation not driven by the accelerometer, which we treat as too noisy for propagation), and a **left-invariant vision correction** with measurement noise transported by the adjoint.

**State.** The mean state is `X ∈ SE(3)`: orientation and position are stored in one homogeneous transform

```
        ┌ R   p ┐
    X = │       │ ,     R ∈ SO(3),  p ∈ ℝ³
        └ 0   1 ┘
```

The covariance is `P ∈ ℝ⁶ˣ⁶`, representing uncertainty in the **𝔰𝔢(3)** tangent coordinates. We write `u^∧` for the **hat** map ℝ⁶ → 𝔰𝔢(3) (angular rates in the top-left 3×3 skew block, linear part in the first three rows of the last column), and `(·)^∨` for the **vee** map back to ℝ⁶.

**Prediction — integrate the gyro.** The mean is propagated on the group with the angular-velocity twist; **position is not predicted from the IMU** (no translational part in the twist):

```
    X⁺ = X · exp(u^∧)
    P⁺ = P + Q
```

with `u = (ωₓ, ωᵧ, ω_z, 0, 0, 0)` (body-frame angular velocity in the first three components, zeros in the translation slots so `exp(u^∧)` updates **R** only). In discrete time, `Q` is the usual process-noise increment on 𝔰𝔢(3) (implementation may scale by `Δt`).

**Correction — left-invariant vision update.** Let `Y ∈ SE(3)` be the vision pose measurement. The innovation is computed **on the group** and then mapped to ℝ⁶:

```
    v = ( log( X⁻¹ Y ) )^∨
```

Measurement noise `N` (6×6 in ℝ⁶) is expressed in the sensor frame; for the update we transport it with the **adjoint** `Ad_{X⁻¹}` so it is consistent with the error coordinates at `X`:

```
    S = H P Hᵀ + Ad_{X⁻¹} N Ad_{X⁻¹}ᵀ
    L = P Hᵀ S⁻¹
    X⁺ = X · exp( (L v)^∧ )
```

Covariance is updated in **Joseph form** for numerical stability:

```
    P⁺ = (I − L H) P (I − L H)ᵀ + L ( Ad_{X⁻¹} N Ad_{X⁻¹}ᵀ ) Lᵀ
```

Here `H` is the linearized observation map ℝ⁶ → ℝ⁶ (in our setup a full-pose measurement uses a constant `H`; the structure above is the generic invariant-EKF correction). The adjoint **Ad** appears wherever noise must be expressed in the same frame as the innovation.

**Why an InEKF instead of a Euclidean EKF?** Pose lives on a curved manifold; a naive EKF on Euler angles or a concatenated vector can distort uncertainty and consistency, especially over **IMU-only** stretches when vision drops. Propagating and correcting **on SE(3)** with `log` / `exp` and an invariant measurement update keeps the mean on the group and the covariance in a meaningful tangent space, which improves behavior when measurements return after occlusion or dropout.

For derivations and adjoint details, see the ROB 530 notes on matrix Lie groups and the Invariant EKF.

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