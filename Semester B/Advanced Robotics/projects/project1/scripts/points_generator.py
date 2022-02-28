#!/usr/bin/env python

import rospy
import random
from ar_week5_test.msg import *


# This node generates random values for initial and final position, velocity, and time every 20 seconds
# Publishs a ROS Topic using the cubic_traj_params message
# Positions and velocities do not exceed a maximum/minimum value of +/- 10
# The time should always be 0, and tf=s
def main():
    pub = rospy.Publisher('initial_data', cubic_traj_params,
                          queue_size=10)
    rospy.init_node('points_generator')
    rate = rospy.Rate(
        0.05)

    print
    "Random Generator Node: publishing messages on topic 'initial_data'"

    while not rospy.is_shutdown():
        p0 = random.uniform(-10, 10)
        pf = random.uniform(-10, 10)
        v0 = random.uniform(-10, 10)
        vf = random.uniform(-10, 10)

        t0 = 0
        dt = random.uniform(5, 10)
        tf = t0 + dt

        pub.publish(p0, pf, v0, vf, t0, tf)
        rate.sleep()


# Initializer
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
