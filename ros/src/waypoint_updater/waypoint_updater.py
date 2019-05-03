#!/usr/bin/env python

import numpy as np
import math
from itertools import cycle, islice

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint

from scipy.spatial import KDTree

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

PUBLISH_RATE  = 50  # Waypoint publish rate in Hz.
LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
MAX_DECEL     = 0.5 # Deceleration limit.
MAX_SPEED_METERS_PER_SEC = 10*0.447 # 10 mph


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.decel_limit = rospy.get_param('~decel_limit', -5)

        rospy.Subscriber('/current_pose',      PoseStamped,  self.pose_cb)
        rospy.Subscriber('/base_waypoints',    Lane,         self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint',  Int32,        self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32,        self.obstacle_cb)

        rospy.Subscriber('/current_velocity',  TwistStamped, self.current_velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.current_pose     = None # Current vehicle position.
        self.current_velocity = None # Current vehicle velocity.
        self.base_waypoints   = None # A list of all waypoints for the track.
        # self.waypoints_cycle  = None # Waypoints cycle (wrap-around).
        self.waypoints_2d     = None # Waypoints in 2D (z coordinate removed).
        self.waypoint_tree    = None # KDTree of 2D waypoints.
        self.stopline_wp_idx  = -1   # Waypoint index for a red light stop line.
        self.obstacle_wp_idx  = -1   # Waypoint index for an obstacle.

        # Publish waypoints until shut down.
        self.loop()

    def loop(self):
        """
        Publish waypoints until shut down.
        """
        rate = rospy.Rate(PUBLISH_RATE)
        while not rospy.is_shutdown():
            # Make sure we have a current car position and the base waypoints.
            if not None in (self.current_pose, self.base_waypoints, self.waypoints_2d, self.waypoint_tree):
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            # Sleep for a bit.
            rate.sleep()

    def get_closest_waypoint_idx(self):
        """
        Get the closest waypoint that is ahead of the current vehicle position.
        """
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        # Return one point                             ^
        # Get the index of the point                      ^

        # Is the closest waypoint ahead or behind the vehicle?
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord    = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest coords.
        closest_vect = np.array(closest_coord)
        prev_vect    = np.array(prev_coord)
        pose_vect    = np.array([x, y])
        val = np.dot(closest_vect - prev_vect, pose_vect - closest_vect)

        # If the closest waypoint is behind the vehicle, just take the next waypoint.
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def publish_waypoints(self, closest_idx):
        """
        The list published to /final_waypoints should include just a fixed number of waypoints
        currently ahead of the vehicle:

            The first waypoint in the list published to /final_waypoints should be the first
            waypoint that is currently ahead of the car.

            The total number of waypoints ahead of the vehicle that should be included in the
            /final_waypoints list is provided by the LOOKAHEAD_WPS variable.
        """
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        # lane.header = self.base_waypoints.header

        # Get the next LOOKAHEAD_WPS waypoints in front of the current vehicle position.
        closest_idx  = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]
        # waypoints = list(islice(self.waypoints_cycle, closest_idx, closest_idx + LOOKAHEAD_WPS))
        # Using list(islice(cycle(waypoints))) should work. But it doesn't.

        # If we have not detected a traffic light or we have but it's still farther away
        # than our lookahead buffer (LOOKAHEAD_WPS), then just return waypoints without
        # adjusted speed.
        if self.stopline_wp_idx == -1 or self.stopline_wp_idx >= farthest_idx:
            lane.waypoints = waypoints
        # We have detected a traffic light within our lookahead buffer, so slow down.
        else:
            lane.waypoints = self.decelerate_waypoints(waypoints, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        new_waypoints = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            # Set the stop index two waypoints back from the stopline so the car
            # stops with the front of the car behind the line. If we omit the '-2'
            # term the car will stop with the center of the car on the stopline.
            # The 'closest_idx' term is included because the waypoint list we're
            # using here is a slice of the base waypoints starting at closest_idx.
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            # Set deceleration - use sqrt() so that we slow down more as we
            # get closer to the stopline waypoint. Could just multiply by a
            # constant so that we'd get linear deceleration.
            vel  = math.sqrt(2 * MAX_DECEL * dist)  # Use self.decel_limit?
            vel  = 0 if vel < 1 else vel

            # Remain at the speed limit (base waypoint velocity) until our
            # calculated velocity (which inludes deceleration) becomes smaller
            # - which will happen as the car gets closer to the stop line.
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)

            new_waypoints.append(p)

        return new_waypoints

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
        rospy.loginfo('Received {} waypoints.'.format(len(self.base_waypoints.waypoints)))
        # 2D version of base waypoints - z-coordinate removed.
        self.waypoints_2d  = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in self.base_waypoints.waypoints]
        # kd-tree for quick nearest-neighbor lookup (scipy.spatial)
        self.waypoint_tree = KDTree(self.waypoints_2d)
        rospy.loginfo('Translated waypoints to 2D and created KDTree.')

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        self.obstacle_wp_idx = msg.data

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

    #
    # Helper Functions
    #

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        """
        Accumulate piece-wise distance for each segment betweeen 'wp1' and 'wp2'.
        """
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
