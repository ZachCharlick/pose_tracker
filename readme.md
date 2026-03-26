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

## MediaPipe pose (pose_landmarker_ros)

Install MediaPipe for the **same interpreter** `ros2 run` uses (on Humble this is usually `/usr/bin/python3`, not Conda):

```bash
/usr/bin/python3 -m pip install --user mediapipe
```

If `pip` pulls **NumPy 2.x** and you see `ImportError: numpy.core.multiarray failed to import` (or matplotlib errors when importing MediaPipe), pin NumPy for compatibility with Ubuntu’s system `matplotlib`:

```bash
/usr/bin/python3 -m pip install --user 'numpy>=1.26.4,<2'
```

Download a pose landmarker `.task` model (e.g. `pose_landmarker_heavy.task`) and pass its absolute path as `model_path`.

Build (if Conda’s `python3` breaks `rosidl_adapter`, put `/usr/bin` first in `PATH` or use your known-good build setup):

```bash
colcon build --packages-select pose_landmarker_interfaces pose_landmarker_ros
source install/setup.bash
```

Run with RealSense RGB (optional annotated image for RViz **Image** display):

```bash
ros2 run pose_landmarker_ros pose_landmarker_node \
  --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p model_path:=/real/path/to/pose_landmarker_heavy.task \
  -p publish_annotated_image:=true
```

Topics: `pose_landmarks` (`pose_landmarker_interfaces/PoseLandmarksStamped`, normalized x/y in image plane), `pose_image_annotated` when enabled.
