#!/usr/bin/env bash

git pull
catkin_make
source devel/setup.bash
roslaunch launch/styx.launch
