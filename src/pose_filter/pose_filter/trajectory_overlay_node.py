"""Publish TF frames for EKF wrist pose, mocap wrist truth, and mocap camera pose."""
from __future__ import annotations

from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    Transform,
    TransformStamped,
)
from lightweight_vicon_bridge.msg import MocapState
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class TrajectoryOverlayNode(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_overlay_node")

        self.declare_parameter("ekf_topic", "/handPose")
        self.declare_parameter("mocap_topic", "/mocap_tracking")
        self.declare_parameter("wrist_segment_name", "wrist_strap")
        self.declare_parameter("camera_segment_name", "camera_530_proj")
        self.declare_parameter("ekf_parent_frame", "camera_color_optical_frame")
        self.declare_parameter("ekf_child_frame", "wrist_strap_ekf")
        self.declare_parameter("mocap_parent_frame", "map")
        self.declare_parameter("wrist_truth_frame", "wrist_strap_truth")
        self.declare_parameter("camera_truth_frame", "camera_link")

        self._ekf_topic = self.get_parameter("ekf_topic").value
        self._mocap_topic = self.get_parameter("mocap_topic").value
        self._wrist_name = self.get_parameter("wrist_segment_name").value
        self._camera_name = self.get_parameter("camera_segment_name").value
        self._ekf_parent_frame = str(self.get_parameter("ekf_parent_frame").value)
        self._ekf_child_frame = str(self.get_parameter("ekf_child_frame").value)
        self._mocap_parent_frame = str(self.get_parameter("mocap_parent_frame").value)
        self._wrist_truth_frame = str(self.get_parameter("wrist_truth_frame").value)
        self._camera_truth_frame = str(self.get_parameter("camera_truth_frame").value)

        self._warn_missing_segments = False
        self._warn_occluded = False

        self._tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            PoseWithCovarianceStamped, self._ekf_topic, self._on_ekf, 10
        )
        self.create_subscription(MocapState, self._mocap_topic, self._on_mocap, 10)

        self.get_logger().info(
            "TF publisher started. EKF: %s -> %s ; mocap: %s -> {%s, %s}"
            % (
                self._ekf_parent_frame, self._ekf_child_frame,
                self._mocap_parent_frame, self._wrist_truth_frame, self._camera_truth_frame,
            )
        )

    def _on_ekf(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        ekf_transform = TransformStamped()
        ekf_transform.transform.translation.x = p.x
        ekf_transform.transform.translation.y = p.y
        ekf_transform.transform.translation.z = p.z
        ekf_transform.transform.rotation.x = q.x
        ekf_transform.transform.rotation.y = q.y
        ekf_transform.transform.rotation.z = q.z
        ekf_transform.transform.rotation.w = q.w
        ekf_transform.header.frame_id = self._ekf_parent_frame
        ekf_transform.child_frame_id = self._ekf_child_frame
        ekf_transform.header.stamp = self.get_clock().now().to_msg()
        self._tf_broadcaster.sendTransform(ekf_transform)

    def _on_mocap(self, msg: MocapState) -> None:
        wrist_tf, wrist_occ = self._find_segment(msg, self._wrist_name)
        cam_tf, cam_occ = self._find_segment(msg, self._camera_name)

        if wrist_tf is None or cam_tf is None:
            if not self._warn_missing_segments:
                self._warn_missing_segments = True
                self.get_logger().warning(
                    'Missing mocap segments in message. Expected "%s" and "%s".'
                    % (self._wrist_name, self._camera_name)
                )
            return

        if wrist_occ or cam_occ:
            if not self._warn_occluded:
                self._warn_occluded = True
                self.get_logger().warning("Skipping occluded mocap segment frame.")
            return

        self._warn_missing_segments = False
        self._warn_occluded = False

        stamp = self.get_clock().now().to_msg()

        wrist_transform = TransformStamped()
        wrist_transform.transform.translation.x = wrist_tf.translation.x
        wrist_transform.transform.translation.y = wrist_tf.translation.y
        wrist_transform.transform.translation.z = wrist_tf.translation.z
        wrist_transform.transform.rotation.x = wrist_tf.rotation.x
        wrist_transform.transform.rotation.y = wrist_tf.rotation.y
        wrist_transform.transform.rotation.z = wrist_tf.rotation.z
        wrist_transform.transform.rotation.w = wrist_tf.rotation.w
        wrist_transform.header.frame_id = self._mocap_parent_frame
        wrist_transform.child_frame_id = self._wrist_truth_frame
        wrist_transform.header.stamp = stamp
        self._tf_broadcaster.sendTransform(wrist_transform)

        camera_transform = TransformStamped()
        camera_transform.transform.translation.x = cam_tf.translation.x
        camera_transform.transform.translation.y = cam_tf.translation.y
        camera_transform.transform.translation.z = cam_tf.translation.z
        camera_transform.transform.rotation.x = cam_tf.rotation.x
        camera_transform.transform.rotation.y = cam_tf.rotation.y
        camera_transform.transform.rotation.z = cam_tf.rotation.z
        camera_transform.transform.rotation.w = cam_tf.rotation.w
        camera_transform.header.frame_id = self._mocap_parent_frame
        camera_transform.child_frame_id = self._camera_truth_frame
        camera_transform.header.stamp = stamp
        self._tf_broadcaster.sendTransform(camera_transform)

    @staticmethod
    def _find_segment(
        msg: MocapState, segment_name: str
    ) -> Tuple[Optional[Transform], bool]:
        for obj in msg.tracked_objects:
            for seg in obj.segments:
                if seg.name == segment_name:
                    return seg.transform, bool(seg.occluded)
        return None, False


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = TrajectoryOverlayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
