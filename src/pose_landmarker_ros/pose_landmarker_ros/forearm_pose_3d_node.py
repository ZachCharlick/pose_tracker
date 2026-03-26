"""Fuse MediaPipe pose landmarks with aligned depth + CameraInfo for 3D forearm pose."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from pose_landmarker_interfaces.msg import PoseLandmark, PoseLandmarksStamped


def _quaternion_from_rotation_matrix(R: np.ndarray) -> Quaternion:
    """Rotation matrix (body basis as columns in camera frame) -> geometry_msgs Quaternion."""
    rvec, _ = cv2.Rodrigues(R)
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-8:
        return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    axis = (rvec.flatten() / angle)
    ha = angle * 0.5
    s = float(np.sin(ha))
    w = float(np.cos(ha))
    return Quaternion(
        x=float(axis[0] * s),
        y=float(axis[1] * s),
        z=float(axis[2] * s),
        w=w,
    )


def _build_rotation_x_forward(x_dir: np.ndarray) -> np.ndarray:
    """Right-handed R; columns are forearm-frame axes in camera frame. +x along bone (distal - proximal)."""
    x = np.asarray(x_dir, dtype=np.float64).reshape(3)
    n = np.linalg.norm(x)
    if n < 1e-9:
        return np.eye(3)
    x = x / n
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(ref, x))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y = np.cross(ref, x)
    yn = np.linalg.norm(y)
    if yn < 1e-9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y = np.cross(ref, x)
        yn = np.linalg.norm(y)
    y = y / yn
    z = np.cross(x, y)
    zn = np.linalg.norm(z)
    if zn < 1e-9:
        return np.eye(3)
    z = z / zn
    return np.column_stack([x, y, z])


class ForearmPose3DNode(Node):
    def __init__(self) -> None:
        super().__init__('forearm_pose_3d_node')

        self.declare_parameter('landmarks_topic', 'pose_landmarks')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('output_pose_topic', 'forearm_pose_camera')
        self.declare_parameter('approx_sync_slop_sec', 0.08)
        self.declare_parameter('sync_queue_size', 50)
        self.declare_parameter('landmark_proximal', 14)
        self.declare_parameter('landmark_distal', 16)
        self.declare_parameter('depth_window_half_size', 1)
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 10.0)
        self.declare_parameter('min_bone_length_m', 0.01)

        self._landmarks_topic = (
            self.get_parameter('landmarks_topic').get_parameter_value().string_value)
        self._depth_topic = (
            self.get_parameter('depth_topic').get_parameter_value().string_value)
        self._camera_info_topic = (
            self.get_parameter('camera_info_topic').get_parameter_value().string_value)
        self._output_topic = (
            self.get_parameter('output_pose_topic').get_parameter_value().string_value)
        slop = self.get_parameter('approx_sync_slop_sec').get_parameter_value().double_value
        qsize = self.get_parameter('sync_queue_size').get_parameter_value().integer_value
        self._idx_a = self.get_parameter('landmark_proximal').get_parameter_value().integer_value
        self._idx_b = self.get_parameter('landmark_distal').get_parameter_value().integer_value
        self._depth_half = self.get_parameter('depth_window_half_size').get_parameter_value().integer_value
        self._min_z = self.get_parameter('min_depth_m').get_parameter_value().double_value
        self._max_z = self.get_parameter('max_depth_m').get_parameter_value().double_value
        self._min_bone = self.get_parameter('min_bone_length_m').get_parameter_value().double_value

        self._bridge = CvBridge()
        self._cam_info: Optional[CameraInfo] = None
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._warned_enc: set = set()

        qos_sensor = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self.create_subscription(
            CameraInfo, self._camera_info_topic, self._on_camera_info, 10)

        self._pub_pose = self.create_publisher(PoseStamped, self._output_topic, 10)

        sub_lm = Subscriber(self, PoseLandmarksStamped, self._landmarks_topic, qos_sensor)
        sub_depth = Subscriber(self, Image, self._depth_topic, qos_sensor)
        self._ats = ApproximateTimeSynchronizer([sub_lm, sub_depth], qsize, slop)
        self._ats.registerCallback(self._on_sync)

        self.get_logger().info(
            f'Sync: "{self._landmarks_topic}" + "{self._depth_topic}" '
            f'(slop={slop}s) -> "{self._output_topic}" (indices {self._idx_a}->{self._idx_b})'
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._cam_info = msg
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if msg.d:
            self._D = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
        else:
            self._D = None

    def _undistort_uv(self, u: float, v: float) -> Tuple[float, float]:
        if self._K is None:
            return u, v
        if self._D is None or self._D.size == 0 or float(np.max(np.abs(self._D))) < 1e-8:
            return u, v
        pts = np.array([[[u, v]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, self._K, self._D, P=self._K)
        return float(und[0, 0, 0]), float(und[0, 0, 1])

    def _map_uv_to_depth(
        self,
        u_color: float,
        v_color: float,
        lm_w: int,
        lm_h: int,
        depth_w: int,
        depth_h: int,
    ) -> Tuple[float, float]:
        ud = u_color * float(depth_w) / float(max(lm_w, 1))
        vd = v_color * float(depth_h) / float(max(lm_h, 1))
        return ud, vd

    def _median_depth_m(
        self, depth_cv: np.ndarray, enc: str, u: float, v: float
    ) -> Optional[float]:
        h, w = depth_cv.shape[0], depth_cv.shape[1]
        uc = int(round(np.clip(u, 0.0, float(w - 1))))
        vc = int(round(np.clip(v, 0.0, float(h - 1))))
        half = max(0, int(self._depth_half))
        u0, u1 = max(0, uc - half), min(w, uc + half + 1)
        v0, v1 = max(0, vc - half), min(h, vc + half + 1)
        patch = depth_cv[v0:v1, u0:u1]

        if enc == '16UC1':
            vals = patch.astype(np.float64).reshape(-1)
            vals = vals[vals > 0]
            if vals.size == 0:
                return None
            z = float(np.median(vals)) * 0.001
        elif enc == '32FC1':
            vals = patch.astype(np.float64).reshape(-1)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size == 0:
                return None
            z = float(np.median(vals))
        else:
            if enc not in self._warned_enc:
                self._warned_enc.add(enc)
                self.get_logger().warning(
                    f'Unsupported depth encoding "{enc}" (expected 16UC1 or 32FC1)'
                )
            return None

        if z < self._min_z or z > self._max_z:
            return None
        return z

    def _deproject(self, u: float, v: float, z: float) -> np.ndarray:
        assert self._K is not None
        fx, fy = float(self._K[0, 0]), float(self._K[1, 1])
        cx, cy = float(self._K[0, 2]), float(self._K[1, 2])
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float64)

    def _landmark_by_index(
        self, msg: PoseLandmarksStamped, idx: int
    ) -> Optional[PoseLandmark]:
        for lm in msg.landmarks:
            if int(lm.index) == idx:
                return lm
        return None

    def _on_sync(self, lm_msg: PoseLandmarksStamped, depth_msg: Image) -> None:
        if self._K is None:
            return

        lma = self._landmark_by_index(lm_msg, self._idx_a)
        lmb = self._landmark_by_index(lm_msg, self._idx_b)
        if lma is None or lmb is None:
            return

        try:
            depth_cv = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding=depth_msg.encoding)
        except Exception as e:
            self.get_logger().warning(f'depth cv_bridge failed: {e}')
            return

        enc = depth_msg.encoding
        dh, dw = depth_cv.shape[0], depth_cv.shape[1]
        lmw = int(lm_msg.image_width) if lm_msg.image_width > 0 else dw
        lmh = int(lm_msg.image_height) if lm_msg.image_height > 0 else dh

        ux_a = float(lma.x) * float(lmw)
        uy_a = float(lma.y) * float(lmh)
        ux_b = float(lmb.x) * float(lmw)
        uy_b = float(lmb.y) * float(lmh)

        uda, vda = self._map_uv_to_depth(ux_a, uy_a, lmw, lmh, dw, dh)
        udb, vdb = self._map_uv_to_depth(ux_b, uy_b, lmw, lmh, dw, dh)

        uda, vda = self._undistort_uv(uda, vda)
        udb, vdb = self._undistort_uv(udb, vdb)

        za = self._median_depth_m(depth_cv, enc, uda, vda)
        zb = self._median_depth_m(depth_cv, enc, udb, vdb)
        if za is None or zb is None:
            return

        p14 = self._deproject(uda, vda, za)
        p16 = self._deproject(udb, vdb, zb)
        pos = 0.5 * (p14 + p16)
        bone = p16 - p14
        bn = np.linalg.norm(bone)
        if bn < self._min_bone:
            return

        R = _build_rotation_x_forward(bone)
        quat = _quaternion_from_rotation_matrix(R)

        frame_id = self._cam_info.header.frame_id if self._cam_info else depth_msg.header.frame_id
        out = PoseStamped()
        out.header.stamp = lm_msg.header.stamp
        out.header.frame_id = frame_id
        out.pose.position.x = float(pos[0])
        out.pose.position.y = float(pos[1])
        out.pose.position.z = float(pos[2])
        out.pose.orientation = quat
        self._pub_pose.publish(out)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = ForearmPose3DNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
