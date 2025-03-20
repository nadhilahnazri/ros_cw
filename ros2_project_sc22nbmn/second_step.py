# Exercise 2 - detecting two colours, and filtering out the third colour and background.



import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal



class colourIdentifier(Node):
    def __init__(self):
        super().__init__('cI')

        # Initialise the value you wish to use for sensitivity in the colour detection (10 should be enough)
        # Remember to initialise a CvBridge() and set up a subscriber to the image topic you wish to use
        # We covered which topic to subscribe to should you wish to receive image data
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.subscription  # prevent unused variable warning
        self.sensitivity = 10

        # Store the last contour area (for movement detection)
        self.last_contour_area = None

    def callback(self, data):

        # Convert the received image into a opencv image
        # But remember that you should always wrap a call to this conversion method in an exception handler
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return
        
        # Convert the rgb image into a hsv image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Set the upper and lower bounds for the two colours you wish to identify
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        
        hsv_red_lower1 = np.array([0, 100, 100])
        hsv_red_upper1 = np.array([self.sensitivity, 255, 255])
        hsv_red_lower2 = np.array([180 - self.sensitivity, 100, 100])
        hsv_red_upper2 = np.array([180, 255, 255])

        hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])

        # Filter out everything but particular colours using the cv2.inRange() method
        # Do this for each colour
        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)
        red_mask1 = cv2.inRange(hsv_image, hsv_red_lower1, hsv_red_upper1)
        red_mask2 = cv2.inRange(hsv_image, hsv_red_lower2, hsv_red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # To combine the masks you should use the cv2.bitwise_or() method
        # You can only bitwise_or two images at once, so multiple calls are necessary for more than two colours
        combined_mask = cv2.bitwise_or(green_mask, cv2.bitwise_or(blue_mask, red_mask))

        # Apply the mask to the original image using the cv2.bitwise_and() method
        # As mentioned on the worksheet the best way to do this is to...
        #bitwise and an image with itself and pass the mask to the mask parameter (rgb_image,rgb_image, mask=mask)
        # As opposed to performing a bitwise_and on the mask and the image.
        green_image = cv2.bitwise_and(image, image, mask=green_mask)
        blue_image = cv2.bitwise_and(image, image, mask=blue_mask)
        red_image = cv2.bitwise_and(image, image, mask=red_mask)
        combined_image = cv2.bitwise_and(image, image, mask=combined_mask)

        # Detect object movement using contours
        movement_status = self.detect_movement(green_mask, image)

        #Show the resultant images you have created. You can show all of them or just the end result if you wish to.
        cv2.imshow('Original Feed', image)
        cv2.imshow('Green Objects', green_image)
        cv2.imshow('Blue Objects', blue_image)
        cv2.imshow('Red Objects', red_image)
        cv2.imshow('Combined Detection', combined_image)

        # Display movement status
        self.get_logger().info(movement_status)

        cv2.waitKey(3)

def detect_movement(self, mask, image):
        """Determines if the TurtleBot is moving closer or farther from the object based on contour size."""
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assuming it's the ball)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # Draw bounding rectangle and enclosing circle
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(cx), int(cy))
            radius = int(radius)
            cv2.circle(image, center, radius, (0, 0, 255), 2)  # Red circle

            # Compare contour area to detect movement
            if self.last_contour_area is not None:
                if area > self.last_contour_area:
                    movement = "TurtleBot Moving Closer"
                elif area < self.last_contour_area:
                    movement = "TurtleBot Moving Away"
                else:
                    movement = "TurtleBot is Stationary"
            else:
                movement = "Tracking Started"

            self.last_contour_area = area  # Update last area

            return f"Ball Detected - {movement}, Contour Area: {area:.2f}"

        return "No Ball Detected"

# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main():
    def signal_handler(sig, frame):
        rclpy.shutdown()
        cv2.destroyAllWindows()

    
    # Instantiate your class
    # And rclpy.init the entire node
    rclpy.init(args=None)
    cI = colourIdentifier()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(cI,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            continue
    except ROSInterruptException:
        pass
    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()
    
# Check if the node is executing in the main path
if __name__ == '__main__':
    main()