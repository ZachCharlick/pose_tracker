"""Subscribe to sensor_msgs/Image, apply one augmentation mode, republish as Image."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

# Set with: --ros-args -p mode:=<value>
MODE_SPLOTCHES = 'splotches'
MODE_FLICKER = 'flicker'
MODE_DISTORTION = 'distortion'
_MODES = {MODE_SPLOTCHES, MODE_FLICKER, MODE_DISTORTION}


class ImageAugmentNode(Node):
    def __init__(self) -> None:
        super().__init__('image_augment_node')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('output_topic', '/camera/camera/color/image_augmented')
        self.declare_parameter('mode', MODE_SPLOTCHES)
        self.declare_parameter('effect_seed', 7)

        # Dirty camera splotches — many black squares, uniform side length (px)
        self.declare_parameter('splotches_count', 140)
        self.declare_parameter('splotches_square_side', 6)

        # Flicker — only when idle: each frame roll this to start a black burst (not during a burst).
        self.declare_parameter('flicker_blackout_probability', 0.05)
        self.declare_parameter('flicker_burst_frames_min', 1)
        self.declare_parameter('flicker_burst_frames_max', 5)

        # Distorted / unfocused lens — blur + radial remap
        self.declare_parameter('distortion_blur_sigma', 3.2)
        self.declare_parameter('distortion_radial_k', 0.22)

        in_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        param_mode = self.get_parameter('mode').get_parameter_value().string_value.strip().lower()
        mode = param_mode or MODE_SPLOTCHES
        if mode not in _MODES:
            self.get_logger().error(
                f'Unknown mode "{mode}". Use: {MODE_SPLOTCHES}, {MODE_FLICKER}, '
                f'{MODE_DISTORTION}. Falling back to {MODE_SPLOTCHES}.'
            )
            mode = MODE_SPLOTCHES
        self._mode = mode

        seed = int(self.get_parameter('effect_seed').get_parameter_value().integer_value)
        self._rng = np.random.default_rng(seed)

        self._splotch_n = int(self.get_parameter('splotches_count').get_parameter_value().integer_value)
        self._splotch_square_side = max(
            1, int(self.get_parameter('splotches_square_side').get_parameter_value().integer_value))
        self._flicker_p = float(
            self.get_parameter('flicker_blackout_probability').get_parameter_value().double_value)
        bmin = int(self.get_parameter('flicker_burst_frames_min').get_parameter_value().integer_value)
        bmax = int(self.get_parameter('flicker_burst_frames_max').get_parameter_value().integer_value)
        lo = max(1, min(bmin, bmax))
        hi = max(bmin, bmax, lo)
        lengths = np.arange(lo, hi + 1, dtype=np.int64)
        wts = 1.0 / lengths.astype(np.float64)
        wts /= wts.sum()
        self._flicker_len_support = lengths
        self._flicker_len_pmf = wts
        self._flicker_burst_remaining = 0

        self._blur_sigma = float(
            self.get_parameter('distortion_blur_sigma').get_parameter_value().double_value)
        self._radial_k = float(
            self.get_parameter('distortion_radial_k').get_parameter_value().double_value)

        self._bridge = CvBridge()
        self._splotch_specs: Optional[List[Tuple[int, int, int]]] = None
        self._splotch_hw: Optional[Tuple[int, int]] = None

        self._pub = self.create_publisher(Image, out_topic, 10)
        # Same as original node: depth 10 + default QoS (int form).
        self.create_subscription(Image, in_topic, self._on_image, 10)

        self.get_logger().info(
            f'Subscribe {in_topic} -> publish {out_topic} | mode={self._mode}'
        )

    def _ensure_splotches(self, h: int, w: int) -> None:
        if self._splotch_specs is not None and self._splotch_hw == (h, w):
            return
        n = max(0, self._splotch_n)
        side = self._splotch_square_side
        specs: List[Tuple[int, int, int]] = []
        for _ in range(n):
            cx = int(self._rng.integers(0, w))
            cy = int(self._rng.integers(0, h))
            specs.append((cx, cy, side))
        self._splotch_specs = specs
        self._splotch_hw = (h, w)
        self.get_logger().info(
            f'Locked {len(specs)} square splotches ({side}px side) for {w}x{h} (fixed dead pixels).')

    def _apply_splotches(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self._ensure_splotches(h, w)
        assert self._splotch_specs is not None
        for cx, cy, side in self._splotch_specs:
            half = side // 2
            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(w, x0 + side)
            y1 = min(h, y0 + side)
            if x1 > x0 and y1 > y0:
                frame[y0:y1, x0:x1] = 0

    def _apply_flicker(self, frame: np.ndarray) -> None:
        p = float(np.clip(self._flicker_p, 0.0, 1.0))
        if self._flicker_burst_remaining > 0:
            frame[:] = 0
            self._flicker_burst_remaining -= 1
            return  # do not roll for a new burst until this one ends
        if self._rng.random() < p:
            burst_len = int(self._rng.choice(self._flicker_len_support, p=self._flicker_len_pmf))
            frame[:] = 0
            self._flicker_burst_remaining = burst_len - 1

    def _apply_distortion(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        # Soft defocus
        ksz = max(3, int(self._blur_sigma * 4) | 1)
        blurred = cv2.GaussianBlur(frame, (ksz, ksz), self._blur_sigma)

        y_idx, x_idx = np.indices((h, w)).astype(np.float32)
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        scale_denom = float(max(w, h))
        dx = (x_idx - cx) / scale_denom
        dy = (y_idx - cy) / scale_denom
        r2 = dx * dx + dy * dy
        k = self._radial_k
        # Sample from slightly expanded radius → mild barrel / broken-lens look
        scale = 1.0 / (1.0 + k * r2 + 1e-6)
        map_x = np.clip(cx + (x_idx - cx) * scale, 0, w - 1).astype(np.float32)
        map_y = np.clip(cy + (y_idx - cy) * scale, 0, h - 1).astype(np.float32)
        warped = cv2.remap(
            blurred, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        frame[...] = warped

    def _on_image(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f'cv_bridge decode failed: {e}')
            return

        out = frame.copy()

        if self._mode == MODE_SPLOTCHES:
            self._apply_splotches(out)
        elif self._mode == MODE_FLICKER:
            self._apply_flicker(out)
        else:
            self._apply_distortion(out)

        try:
            out_msg = self._bridge.cv2_to_imgmsg(out, encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f'cv_bridge encode failed: {e}')
            return

        out_msg.header = msg.header
        self._pub.publish(out_msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = ImageAugmentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
