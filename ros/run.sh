#!/usr/bin/env bash
catkin_make
source devel/setup.bash
roslaunch launch/styx.launch
