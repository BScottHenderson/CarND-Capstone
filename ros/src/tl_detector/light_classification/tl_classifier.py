from styx_msgs.msg import TrafficLight

import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

            UNKNOWN
            GREEN
            YELLOW
            RED

        Find RGV to HSV conversions via this web site: http://colorizer.org/
        """
        #TODO implement light color prediction
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green  = np.array([ 95,  0,  0])
        upper_green  = np.array([140,255,255])
        green_mask   = cv2.inRange(hsv, lower_green, upper_green)

        lower_yellow = np.array([ 45,  0,  0])
        upper_yellow = np.array([ 70,255,255])
        yellow_mask  = cv2.inRange(hsv, lower_yellow, upper_yellow)

        lower_red    = np.array([  0,  0,  0])
        upper_red    = np.array([ 10,255,255])
        red_mask     = cv2.inRange(hsv, lower_red, upper_red)

        if self.check_image(image, green_mask):
            light_color = TrafficLight.GREEN
        elif self.check_image(image, yellow_mask):
            light_color = TrafficLight.YELLOW
        elif self.check_image(image, red_mask):
            light_color = TrafficLight.RED
        else:
            light_color = TrafficLight.UNKNOWN

        return light_color

    def check_image(self, image, mask):
        img = cv2.bitwise_and(image, image, mask=mask)
        img = cv2.medianBlur(img, 5)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=20, maxRadius=30)

        return True if circles is not None else False
