#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

from scipy.spatial import KDTree


STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        # Telemetry data
        self.current_pose     = None
        self.base_waypoints   = None
        self.waypoints_2d     = None    # Waypoints in 2D (z coordinate removed).
        self.waypoints_tree   = None    # KDTree of 2D waypoints.
        self.traffic_lights   = []
        self.camera_image     = None
        self.camera_image_raw = None

        rospy.Subscriber('/current_pose',   PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane,        self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_light_cb)   # Simulator data - not available on the actual vehicle.
        rospy.Subscriber('/image_color',            Image,             self.image_cb)           # Data from the car's camera.
        # Raw camera image - use this for classifier instead of '/image_color' because the translation
        # to a particular color scheme may result is data loss. The classifier needs as much data as
        # it can get.
        rospy.Subscriber('/image_raw',              Image,             self.image_raw_cb)

        # Read the permanent (x, y) coordinates of each traffic light's stop line.
        config_string         = rospy.get_param('/traffic_light_config')
        self.config           = yaml.load(config_string)

        # Publish the index of the waypoint for nearest upcoming red light's stop line.
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge           = CvBridge()              # Image conversion
        self.light_classifier = TLClassifier()
        self.listener         = tf.TransformListener()

        self.state            = TrafficLight.UNKNOWN
        self.last_state       = TrafficLight.UNKNOWN
        self.last_wp          = -1
        self.state_count      = 0

        rospy.spin()

    #
    # Callbacks
    #

    def pose_cb(self, msg):
        # Save the current vehicle position.
        self.current_pose = msg

    def waypoints_cb(self, waypoints):
        # Save waypoints for later use.
        # This list includes all waypoints for the track - the '/base_waypoints' publisher publishes only once.
        self.base_waypoints = waypoints
        # self.waypoints_cycle = cycle(self.base_waypoints.waypoints)
        rospy.logwarn('tl_detector: Received {} waypoints.'.format(len(self.base_waypoints.waypoints)))
        # 2D version of base waypoints - z-coordinate removed.
        self.waypoints_2d  = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in self.base_waypoints.waypoints]
        # kd-tree for quick nearest-neighbor lookup (scipy.spatial)
        self.waypoint_tree = KDTree(self.waypoints_2d)
        rospy.logwarn('tl_detector: Translated waypoints to 2D and created KDTree.')

    def traffic_light_cb(self, msg):
        self.traffic_lightss = msg.lights   # Array of TrafficLight objects.
        rospy.logwarn('tl_detector: Received {} traffic lights.'.format(len(self.traffic_lights)))

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera
        """
        self.camera_image = msg
        light_wp, state   = self.process_traffic_lights()

        """
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is used.
        """
        if self.state != state:
            self.state_count = 0
            self.state       = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            # Reached the threshold, use the new light wp index.
            # :TODO Modify this code to handle TrafficLight.YELLOW
            self.last_state = self.state
            light_wp        = light_wp if state == TrafficLight.RED else -1
            self.last_wp    = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            # Not at threshold yet, maintain previous light wp.
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def image_raw_cb(self, msg):
        self.camera_image_raw = msg

    #
    # Helper Functions
    #

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            x, y (float): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        # Return one point                             ^
        # Get the index of the point                      ^
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        rospy.logwarn('tl_detector: get_light_state: ({}, {}) {}'.format(
            light.pose.pose.position.x, light.pose.pose.position.y, light.state))
        """
        The simulator provides light state but for the real car we must use a classifier to determine light state.

        if not self.camera_image:
            return TrafficLight.UNKNOWN

        # Convert from rospy Image to OpenCV.
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        return self.light_classifier.get_classification(cv_image)
        """
        return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx   = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.current_pose:
            car_wp_idx = self.get_closest_waypoint(self.current_pose.pose.position.x, self.current_pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.base_waypoints.waypoints)   # Distance in indices between vehicle wp and closest light wp.
            for i, light in enumerate(self.traffic_lights):
                # Get stop line waypoint index.
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index.
                d = temp_wp_idx - car_wp_idx
                if 0 <= d and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
