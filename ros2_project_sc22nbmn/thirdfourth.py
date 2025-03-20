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

    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV bounds for Green and Red
    hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
    hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

    hsv_red_lower1 = np.array([0, 100, 100])
    hsv_red_upper1 = np.array([self.sensitivity, 255, 255])
    hsv_red_lower2 = np.array([180 - self.sensitivity, 100, 100])
    hsv_red_upper2 = np.array([180, 255, 255])

    # Create masks
    green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
    red_mask1 = cv2.inRange(hsv_image, hsv_red_lower1, hsv_red_upper1)
    red_mask2 = cv2.inRange(hsv_image, hsv_red_lower2, hsv_red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Find contours
    self.green_found, self.green_area = self.detect_colour_contour(green_mask, image, (0, 255, 0))  # Green
    self.red_found, self.red_area = self.detect_colour_contour(red_mask, image, (0, 0, 255))  # Red

    # Call movement function
    self.control_movement()

def detect_colour_contour(self, mask, image, color):
    """Finds the largest contour of a given color and returns its area."""
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # Draw bounding box

        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(cx), int(cy))
        radius = int(radius)
        cv2.circle(image, center, radius, color, 2)  # Draw enclosing circle

        return True, area  # Detected
    return False, 0  # Not detected

def control_movement(self):
    twist_msg = Twist()

    # Stop if red is detected
    if self.red_found:
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.get_logger().info("Red Detected - Stopping Robot!")
    elif self.green_found:
        # Move toward the object if it's small (far away)
        if self.green_area < 5000:  
            twist_msg.linear.x = 0.2  # Move forward
            twist_msg.angular.z = 0.0
            self.get_logger().info("Following Green - Moving Closer")
        else:
            # Stop when close enough
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.get_logger().info("Green Object Reached - Stopping")
    else:
        # If no green or red, stop
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.get_logger().info("No Target Found - Stopping")

    # Publish movement command
    self.publisher.publish(twist_msg)


