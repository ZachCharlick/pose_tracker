#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import glob
import time
import argparse
import sys

class ImagePublisher(Node):
    def __init__(self, image_folder, fps=30):
        super().__init__('image_publisher')
        self.pub = self.create_publisher(Image, 'camera', 10)
        self.bridge = CvBridge()
        self.images = sorted(glob.glob(f'{image_folder}/*.png'))
        self.fps = fps
        self.get_logger().info(f'Publishing {len(self.images)} images from {image_folder} at {fps} FPS')

    def publish_images(self):
        delay = 1.0 / self.fps
        for img_path in self.images:
            cv_img = cv2.imread(img_path)
            if cv_img is None:
                self.get_logger().warn(f'Could not read image {img_path}')
                continue
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='rgb8')
            self.pub.publish(msg)
            time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description='Publish a folder of images to ROS2 topic')
    parser.add_argument('folder', help='Path to folder containing images')
    parser.add_argument('--fps', type=float, default=30.0, help='Publishing FPS')
    args = parser.parse_args()

    rclpy.init()  # just init normally
    node = ImagePublisher(args.folder, fps=args.fps)
    node.publish_images()
    rclpy.shutdown()

if __name__ == '__main__':
    main()