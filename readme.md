For motion capture: Need to be on a computer in the ARM Lab

Terminal 1: (Vicon Bridge)
source install/setup.bash
ros2 run lightweight_vicon_bridge lightweight_vicon_bridge_node

For realsense ROS integration, need to install package:
sudo apt install ros-humble-realsense2-camera ros-humble-realsense2-description

then launch with

source install/setup.bash
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