import threading
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
from geometry_msgs.msg import Twist
import signal

class ColourFollower(Node):
    def __init__(self):
        super().__init__('colour_follower')

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription  

        self.sensitivity = 15  
        self.green_found = False
        self.red_found = False
        self.green_area = 0
        self.red_area = 0

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        hsv_red_lower1 = np.array([0, 100, 100])
        hsv_red_upper1 = np.array([self.sensitivity, 255, 255])
        hsv_red_lower2 = np.array([180 - self.sensitivity, 100, 100])
        hsv_red_upper2 = np.array([180, 255, 255])

        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        red_mask1 = cv2.inRange(hsv_image, hsv_red_lower1, hsv_red_upper1)
        red_mask2 = cv2.inRange(hsv_image, hsv_red_lower2, hsv_red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        self.green_found, self.green_area = self.detect_colour_contour(green_mask, image, (0, 255, 0))
        self.red_found, self.red_area = self.detect_colour_contour(red_mask, image, (0, 0, 255))

        self.control_movement()

    def detect_colour_contour(self, mask, image, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(cx), int(cy))
            radius = int(radius)
            cv2.circle(image, center, radius, color, 2)

            return True, area
        return False, 0

    def control_movement(self):
        twist_msg = Twist()

        if self.red_found:
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.get_logger().info("Red Detected - Stopping Robot!")
        elif self.green_found:
            if self.green_area < 5000:  
                twist_msg.linear.x = 0.2
                self.get_logger().info("Following Green - Moving Closer")
            else:
                twist_msg.linear.x = 0.0
                self.get_logger().info("Green Object Reached - Stopping")
        else:
            twist_msg.linear.x = 0.0

        self.publisher.publish(twist_msg)

def main():
    rclpy.init(args=None)
    node = ColourFollower()
    rclpy.spin(node)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
