For motion capture: Need to be on a computer in the ARM Lab

Terminal 1: (Vicon Bridge)
source install/setup.bash
ros2 run lightweight_vicon_bridge lightweight_vicon_bridge_node

For realsense ROS integration, need to install package:
sudo apt install ros-humble-realsense2-camera ros-humble-realsense2-description

then launch with

ros2 launch realsense2_camera rs_launch.py \
    align_depth.enable:=true \
    pointcloud.enable:=true \
    depth_module.depth_profile:=640x480x30 \
    rgb_camera.color_profile:=640x480x30

RGB image is at /camera/camera/color/image_raw

To get the landmarks and visualize

ros2 run pose_landmarker_ros pose_landmarker_node \
  --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p model_path:=/home/zacharycharlick/Downloads/google-pose-api-main/pose_landmarker_heavy.task \
  -p publish_annotated_image:=true \
  -p landmarks_topic:=pose_landmarks

To get forearm pose

ros2 run pose_landmarker_ros forearm_pose_3d_node \
  --ros-args \
  -p landmarks_topic:=pose_landmarks \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/color/camera_info \
  -p output_pose_topic:=forearm_pose_camera