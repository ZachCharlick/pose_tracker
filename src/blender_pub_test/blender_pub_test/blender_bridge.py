import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import socket
import threading
from std_msgs.msg import Header
import numpy as np
from cv_bridge import CvBridge
from io import BytesIO
from PIL import Image as PILImage

HOST = '127.0.0.1'
PORT = 5006  # separate port for images

class BlenderImageBridge(Node):
    def __init__(self):
        super().__init__('blender_image_bridge')
        self.publisher_ = self.create_publisher(Image, 'camera_image', 10)
        self.bridge = CvBridge()

        thread = threading.Thread(target=self.socket_server, daemon=True)
        thread.start()

    def socket_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            print(f"Listening for images on {HOST}:{PORT}")
            while True:
                conn, _ = s.accept()
                print("Client connected")
                with conn:
                    buffer = b''
                    while True:
                        data = conn.recv(4096)
                        if not data:
                            break
                        buffer += data

                        # Assume each frame is sent with a length prefix (4 bytes)
                        while len(buffer) >= 4:
                            # first 4 bytes = frame length
                            frame_len = int.from_bytes(buffer[:4], 'big')
                            if len(buffer) < 4 + frame_len:
                                break
                            frame_bytes = buffer[4:4+frame_len]
                            buffer = buffer[4+frame_len:]

                            try:
                                pil_img = PILImage.open(BytesIO(frame_bytes)).convert('RGBA')
                                np_img = np.array(pil_img)
                                ros_img = self.bridge.cv2_to_imgmsg(np_img, encoding='rgba8')
                                ros_img.header = Header()
                                ros_img.header.stamp = self.get_clock().now().to_msg()
                                self.publisher_.publish(ros_img)
                                self.get_logger().info(f"Published image {np_img.shape}")
                            except Exception as e:
                                print("Failed to process image:", e)

def main():
    rclpy.init()
    node = BlenderImageBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()