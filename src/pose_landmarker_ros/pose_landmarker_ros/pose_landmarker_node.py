import os
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.publisher import Publisher
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from pose_landmarker_interfaces.msg import PoseLandmark, PoseLandmarksStamped

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def _pose_connection_pairs() -> List[Tuple[int, int]]:
    try:
        conns = mp.solutions.pose.POSE_CONNECTIONS
        out: List[Tuple[int, int]] = []
        for c in conns:
            a, b = c[0], c[1]
            ai = int(a.value) if hasattr(a, 'value') else int(a)
            bi = int(b.value) if hasattr(b, 'value') else int(b)
            out.append((ai, bi))
        return out
    except Exception:
        return []


def _landmark_labels() -> List[str]:
    try:
        return [m.name for m in mp.solutions.pose.PoseLandmark]
    except Exception:
        return [str(i) for i in range(33)]


class PoseLandmarkerNode(Node):
    def __init__(self) -> None:
        super().__init__('pose_landmarker_node')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('landmarks_topic', 'pose_landmarks')
        self.declare_parameter('model_path', 'pose_landmarker_heavy.task')
        self.declare_parameter('publish_annotated_image', False)
        self.declare_parameter('annotated_image_topic', 'pose_image_annotated')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        landmarks_topic = self.get_parameter('landmarks_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self._publish_annot = self.get_parameter(
            'publish_annotated_image').get_parameter_value().bool_value
        annot_topic = self.get_parameter(
            'annotated_image_topic').get_parameter_value().string_value

        if not os.path.isfile(model_path):
            self.get_logger().fatal(
                f'Model file not found: {model_path}. '
                'Pass absolute path via model_path parameter.'
            )
            raise RuntimeError(f'Missing model: {model_path}')

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._last_mp_ms = -1
        self._bridge = CvBridge()
        self._conn_pairs = _pose_connection_pairs()
        self._labels = _landmark_labels()

        self._pub_landmarks = self.create_publisher(PoseLandmarksStamped, landmarks_topic, 10)
        self._pub_image: Optional[Publisher] = None
        if self._publish_annot:
            self._pub_image = self.create_publisher(Image, annot_topic, 10)

        self.create_subscription(Image, image_topic, self._image_callback, 10)

        self.get_logger().info(
            f'Subscribe: {image_topic}, landmarks: {landmarks_topic}, '
            f'annotated: {self._publish_annot} -> {annot_topic if self._publish_annot else "off"}'
        )

    def destroy_node(self) -> bool:
        try:
            self._landmarker.close()
        except Exception:
            pass
        return super().destroy_node()

    def _next_monotonic_ms(self, header: Header) -> int:
        ms = int(header.stamp.sec * 1000 + header.stamp.nanosec // 1_000_000)
        if ms <= self._last_mp_ms:
            ms = self._last_mp_ms + 1
        self._last_mp_ms = ms
        return ms

    def _image_callback(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f'cv_bridge failed: {e}')
            return

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t_ms = self._next_monotonic_ms(msg.header)
        result = self._landmarker.detect_for_video(mp_image, t_ms)

        out_msg = PoseLandmarksStamped()
        out_msg.header = msg.header
        out_msg.image_width = int(w)
        out_msg.image_height = int(h)

        landmarks_list = None
        if result.pose_landmarks:
            landmarks_list = result.pose_landmarks[0]
            for idx, lm in enumerate(landmarks_list):
                pl = PoseLandmark()
                pl.index = idx
                pl.x = float(lm.x)
                pl.y = float(lm.y)
                pl.z = float(lm.z)
                pl.visibility = float(getattr(lm, 'visibility', 0.0))
                pl.presence = float(getattr(lm, 'presence', 0.0))
                out_msg.landmarks.append(pl)

        self._pub_landmarks.publish(out_msg)

        if self._pub_image is not None:
            vis = frame.copy()
            if landmarks_list is not None:
                pts = [
                    (int(lm.x * w), int(lm.y * h))
                    for lm in landmarks_list
                ]
                for a, b in self._conn_pairs:
                    if a < len(pts) and b < len(pts):
                        cv2.line(vis, pts[a], pts[b], (0, 255, 0), 2)
                for idx, (px, py) in enumerate(pts):
                    cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)
                    label = self._labels[idx] if idx < len(self._labels) else str(idx)
                    cv2.putText(
                        vis,
                        label,
                        (px + 4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
            try:
                out_img = self._bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            except Exception as e:
                self.get_logger().warning(f'cv2_to_imgmsg failed: {e}')
                return
            out_img.header = msg.header
            self._pub_image.publish(out_img)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = PoseLandmarkerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
